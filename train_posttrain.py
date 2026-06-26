import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import copy
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from posevla.modeling_posevla import PoseVLAConfig, PoseVLAPolicy
from data.ds_train.robot.dataset_hdf5_action import VLAConsumerDataset, make_dataset
from data.collators import CollatorForActionConsumerDataset as DataCollatorForPI0ConsumerDataset
import torch
import wandb
from tqdm import tqdm
import random
from pathlib import Path
from transformers import logging
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from utils.logger import register_features_types, save_wandb, initialize_wandb
import matplotlib.pyplot as plt
from utils.vis import plot_all_joints


logging.set_verbosity_error()   # 只显示错误

def set_global_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cast_training_params(model, dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def convert_omegaconf_to_native(obj):
    """Recursively convert OmegaConf types to native Python types."""
    from omegaconf import DictConfig, ListConfig
    
    if isinstance(obj, ListConfig):
        return [convert_omegaconf_to_native(item) for item in obj]
    elif isinstance(obj, DictConfig):
        return {key: convert_omegaconf_to_native(value) for key, value in obj.items()}
    else:
        return obj


def load_action_expert(policy, action_expert_path):
    from safetensors.torch import load_file

    model_file = Path(action_expert_path) / "model.safetensors"
    state_dict = load_file(str(model_file), device="cpu")
    prefix = "model.paligemma_with_expert.gemma_expert."

    expert_state = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        expert_key = key[len(prefix):]
        if expert_key in {"model.embed_tokens.weight", "lm_head.weight"}:
            continue
        expert_state[expert_key] = value

    missing, unexpected = policy.model.paligemma_with_expert.gemma_expert.load_state_dict(
        expert_state,
        strict=False,
    )
    print(f"load pretrain action expert from: {action_expert_path}")
    if missing:
        print(f"  action expert missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  action expert unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

@hydra.main(
        version_base=None,
        config_path="./config",
    config_name="base_posttrain",
    )
def train(cfg: DictConfig) -> None:
    # for better performance, but reduce reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl")
    accelerator_project_config = ProjectConfiguration(total_limit=cfg.checkpoints_total_limit)
    accelerator = Accelerator(
                              dispatch_batches=False,
                              mixed_precision=cfg.training.mixed_precision,
                              gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=cfg.deepspeed),
                              project_config=accelerator_project_config,
                              )
    if not cfg.debug:

        if accelerator.is_main_process:
            if cfg.resume_ckpt:
                cfg_prev = OmegaConf.load(Path(cfg.resume_ckpt).parent / "base.yaml")
                cfg = OmegaConf.merge(cfg_prev, {"resume_ckpt": cfg.resume_ckpt})
                cfg.ckpt_save_dir = cfg.resume_ckpt
                cfg = initialize_wandb(cfg) # load wandb.run.id
            else:
                cfg = initialize_wandb(cfg)
                wandb.config.update(OmegaConf.to_container(cfg))
                os.makedirs(cfg.ckpt_save_dir, exist_ok=True)
                save_wandb(wandb, cfg.ckpt_save_dir)
                OmegaConf.save(cfg, Path(cfg.ckpt_save_dir) / "base.yaml")

        accelerator.wait_for_everyone()


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32

    ############# Config Model ############
    accelerator.print("Initialize Model")
    posevla_config_kwargs = dict(
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1, # not used
        optimizer_lr = cfg.training.optimizer_lr,
        optimizer_betas = cfg.training.optimizer_betas,
        optimizer_eps = cfg.training.optimizer_eps,
        optimizer_weight_decay= cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps = cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps = cfg.training.scheduler_decay_steps,
        scheduler_decay_lr = cfg.training.scheduler_decay_lr,
        is_knowledge_insulation=cfg.training.is_knowledge_insulation,
        pi05=cfg.training.pi05,
        vis_attn=cfg.training.vis_attn,
        add_extra_token=cfg.training.add_extra_token,
        add_image_token=cfg.training.add_image_token,
        add_prior=cfg.training.add_prior,
    )
    posevla_config = PoseVLAConfig(**posevla_config_kwargs)

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        policy = PoseVLAPolicy.from_pretrained(os.path.join(cfg.resume_ckpt, "model"), local_files_only=True)
    else:
        # posevlm + pi0_expert
        policy = PoseVLAPolicy.from_pretrained(cfg.model.pretrained_model_path, config=posevla_config, strict=False)
        print(f"load pretrain vlm model from: {cfg.model.pretrained_model_path}")
        load_action_expert(policy, cfg.model.action_expert_path)


    del policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens
    del policy.model.paligemma_with_expert.gemma_expert.lm_head

    gc.collect()

    if cfg.co_training.vlm_training:
        policy.resize_embeddings()

    policy.to(weight_dtype)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    ############# Config Optimizer ############
    accelerator.print("Initialize Optimizer")
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(posevla_config, policy)
    total_train_steps = cfg.training.max_training_steps * accelerator.num_processes * cfg.training.grad_accumulation_steps
    optimizer = posevla_config.get_optimizer_preset().build(filter(lambda p: p.requires_grad, policy.parameters()))
    posevla_config.scheduler_warmup_steps *= accelerator.num_processes
    posevla_config.scheduler_decay_steps *= accelerator.num_processes
    lr_scheduler = posevla_config.get_scheduler_preset().build(optimizer, total_train_steps)

    ############# Config Dataset and Dataloader ############
    accelerator.print("Initialize Dataset and Dataloader")
    # Dataset and DataLoaders creation:

    # Load vlm dataset:
    if cfg.co_training.vlm_training:
        from importlib import import_module

        dataset_vlm = import_module("dataset_vlm")
        VLMConsumerDataset = dataset_vlm.VLMConsumerDataset
        DataCollatorForPI0VLMConsumerDataset = dataset_vlm.DataCollatorForPI0VLMConsumerDataset

        train_vlm_dataset = VLMConsumerDataset(config=cfg)
        data_vlm_collator = DataCollatorForPI0VLMConsumerDataset(config=cfg)
        train_vlm_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_vlm_dataset, collate_fn=data_vlm_collator)

    # Load action dataset:
    if cfg.co_training.action_training:
        if cfg.dataset.type == "hdf5":
            # HDF5 format
            train_dataset = VLAConsumerDataset(config=cfg)
            data_collator = DataCollatorForPI0ConsumerDataset(config=cfg)
            train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset, collate_fn=data_collator)

        elif cfg.dataset.type == "lerobot":
            # Lerobot format
            train_dataset = make_dataset(posevla_config, cfg)
            train_dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
        else:
            raise NotImplementedError("Only support `hdf5` and `lerobot` dataset formats now.")


    ############# Preapare `accelerator` ############
    # Prepare everything with our `accelerator`.
    # will cast the parameters of model into mix-precision type in deepspeed mode
    # vlm + action
    if cfg.co_training.vlm_training and not cfg.co_training.action_training:
        print("Prepare vlm dataloader")
        policy, optimizer, train_vlm_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_vlm_dataloader, lr_scheduler
        )
    elif not cfg.co_training.vlm_training and cfg.co_training.action_training:
        print("Prepare action dataloader")
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
    elif cfg.co_training.vlm_training and cfg.co_training.action_training:
        print("Prepare vlm + action dataloader")
        policy, optimizer, train_vlm_dataloader, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_vlm_dataloader, train_dataloader, lr_scheduler
        )
    else:
        raise NotImplementedError("Only support `vlm` and `action` training now.")

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming optimizer and scheduler from checkpoint {cfg.resume_ckpt}")
        accelerator.load_state(os.path.join(cfg.resume_ckpt, "state", "training_state.pth"))
        accelerator.wait_for_everyone()

    total_batch_size = (
            cfg.training.batch_size
            * accelerator.num_processes
            * cfg.training.grad_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Total parameters: {num_total_params} M")
    accelerator.print(f"  Trainable parameters: {num_learnable_params} M")
    if cfg.co_training.vlm_training:
        accelerator.print(f"  Num examples VLM = {len(train_vlm_dataset)}")
    if cfg.co_training.action_training:
        accelerator.print(f"  Num examples VLA = {len(train_dataset)}")
    accelerator.print(f"  Instantaneous batch size per device = {cfg.training.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {cfg.training.max_training_steps}")
    accelerator.print(f"  is_knowledge_insulation: {cfg.training.is_knowledge_insulation}")
    accelerator.print(f"  pi05: {cfg.training.pi05}")
    accelerator.print(f"  vis_attn: {cfg.training.vis_attn}")
    accelerator.print(f"  add_extra_token: {cfg.training.add_extra_token}")
    accelerator.print(f"  add_image_token: {cfg.training.add_image_token}")
    accelerator.print(f"  add_prior: {cfg.training.add_prior}")
    accelerator.print(f"  weighted_sample: {cfg.training.weighted_sample}")

    train_loss = []
    train_flow_loss = []
    train_align_loss = []
    train_ntp_loss = []

    val_loss = []
    angle_dist = []
    if cfg.co_training.vlm_training:
        train_vlm_loader_iter = cycle(train_vlm_dataloader)
        # val_vlm_loader_iter = iter(val_vlm_dataloader)
    if cfg.co_training.action_training:
        train_loader_iter = cycle(train_dataloader)
        # val_loader_iter = iter(val_dataloader)
    current_step = int(cfg.resume_ckpt.split("/")[-1]) + 1 if cfg.resume_ckpt else 0
    progress_bar = tqdm(range(current_step, cfg.training.max_training_steps), ncols=100, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    progress_bar.update(current_step)

    # for n_iter in tqdm(range(cfg.training.max_training_steps), desc="Training", ncols=100, disable=not accelerator.is_local_main_process):
    for n_iter in range(current_step, cfg.training.max_training_steps):
        policy.train()
        with accelerator.accumulate(policy):
            if cfg.co_training.vlm_training:
                batch = next(train_vlm_loader_iter)
            if cfg.co_training.action_training:
                batch = next(train_loader_iter)

            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(weight_dtype)

            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = policy.parameters()
                accelerator.clip_grad_norm_(params_to_clip, cfg.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if (n_iter + 1) % (cfg.training.train_frequency * cfg.training.grad_accumulation_steps) == 0 and accelerator.is_main_process and not cfg.debug:
                total_norm = 0.0
                for p in policy.parameters():
                    if p.requires_grad:
                        param_norm = p.data.norm(2)  # L2 norm
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                wandb.log({"train/total_weight_norm": total_norm}, step=n_iter)

            train_loss.append(loss.item())
            if cfg.co_training.vlm_training:
                train_ntp_loss.append(output_dict["ntp_loss"])
            if cfg.co_training.action_training:
                train_flow_loss.append(output_dict["flow_loss"])

            if (n_iter+1) % (cfg.training.train_frequency * cfg.training.grad_accumulation_steps) == 0 and accelerator.is_main_process and not cfg.debug:
                wandb.log(
                    {
                        "train/loss": np.mean(train_loss),
                        "train/flow_loss": np.mean(train_flow_loss) if len(train_flow_loss) > 0 else 0,
                        "train/align_loss": np.mean(train_align_loss) if len(train_align_loss) > 0 else 0,
                        "train/ntp_loss": np.mean(train_ntp_loss) if len(train_ntp_loss) > 0 else 0,
                        "train/iter": n_iter,
                        "train/lr":optimizer.param_groups[0]["lr"],
                    }
                )
                train_loss = []
                train_flow_loss = []
                train_align_loss = []
                train_ntp_loss = []

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": np.mean(train_loss) if len(train_loss) > 0 else 0})

        # validation
        if (n_iter + 1) % (cfg.training.eval_frequency * cfg.training.grad_accumulation_steps) == 0:
        # if True:
            accelerator.wait_for_everyone()
            accelerator.print(f"\n Evaluate at n_iter {n_iter}.")
            eval_model = accelerator.unwrap_model(policy)
            eval_model.eval()
            for i in tqdm(range(cfg.training.max_evaluation_steps), desc="Validation", ncols=100, disable=not accelerator.is_local_main_process):
                if cfg.co_training.vlm_training:
                    batch = next(train_vlm_loader_iter)
                if cfg.co_training.action_training:
                    batch = next(train_loader_iter)

                with torch.no_grad():
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(weight_dtype)

                    output_dict = eval_model(batch)

                    if cfg.co_training.vlm_training:
                        ce_loss = output_dict['loss']
                        output = eval_model.forward_evaluate_ntp(batch)
                        gt_texts, pred_texts = output["gt"], output["pred"]

                        all_loss = accelerator.gather_for_metrics((ce_loss))

                        dist = torch.tensor(0.0, device=ce_loss.device)
                        all_loss = all_loss.mean()

                    if cfg.co_training.action_training:
                        flow_loss = output_dict['loss']
                        output = eval_model.forward_evaluate(batch)
                        gt_actions, pred_actions = output["gt"], output["pred"]
                        all_predictions, all_targets, all_loss = accelerator.gather_for_metrics((pred_actions, gt_actions, flow_loss))

                        eval_action_dim = 16 if cfg.dataset.action_type == "eep" else 14
                        dist = (
                            all_predictions[:, :, :eval_action_dim]
                            - all_targets[:, :, :eval_action_dim]
                        ).abs().mean()
                        all_loss = all_loss.mean()

                    val_loss.append(all_loss.item())
                    angle_dist.append(dist.item())
            eval_model.train()
            if accelerator.is_main_process:
                if not cfg.debug:
                    wandb.log(
                        {
                            "val/loss": np.mean(val_loss),
                            "val/angle dist": np.mean(angle_dist),
                            "val/iter": n_iter,
                        }
                    )
                    val_loss = []
                    angle_dist = []

                    if cfg.co_training.vlm_training:
                        try:
                            from utils.vis import plot_all_flows, text_to_flow
                        except ImportError as exc:
                            raise ImportError(
                                "VLM logging requires plot_all_flows/text_to_flow in utils.vis."
                            ) from exc

                        pred_flow_norm = text_to_flow(pred_texts[0])
                        gt_flow_norm = text_to_flow(gt_texts[0])

                        # columns = ["Text"]
                        fig = plot_all_flows(batch["observation.images.top_head"][0], gt_flow_norm, pred_flow_norm, batch['task'][0], wandb=True)
                        wandb.log({"all_flow_subplot": wandb.Image(fig),
                                   "text_flow": wandb.Table(data=[[pred_texts[0], gt_texts[0]]], columns=["Text_pred", "Text_gt"])
                                   })


                        plt.close()

                    if cfg.co_training.action_training:
                        fig = plot_all_joints(all_targets[0].float().detach().cpu().numpy(),
                                              all_predictions[0].float().detach().cpu().numpy(), wandb=True)
                        wandb.log({"all_angles_subplot": wandb.Image(fig)})
                        plt.close()


            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

        # save checkpoint
        if (n_iter+1) % (cfg.training.ckpt_frequency * cfg.training.grad_accumulation_steps) == 0 and not cfg.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_model = accelerator.unwrap_model(policy)
                # require copy or clone to avoid shared memory of tensors. https://github.com/huggingface/safetensors/issues/202
                model_to_save = copy.deepcopy(ckpt_model)
                
                # Fix: Convert ALL OmegaConf types to native Python types to avoid draccus serialization error
                # This is needed because deepcopy preserves OmegaConf types which draccus can't serialize
                for attr_name in dir(model_to_save.config):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(model_to_save.config, attr_name)
                            if not callable(attr_value):
                                converted_value = convert_omegaconf_to_native(attr_value)
                                setattr(model_to_save.config, attr_name, converted_value)
                        except:
                            pass
                
                # Explicitly convert known fields
                if hasattr(model_to_save.config, 'optimizer_betas'):
                    model_to_save.config.optimizer_betas = tuple(model_to_save.config.optimizer_betas)
                
                register_features_types()
                model_to_save.save_pretrained(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "model"))
                del model_to_save
                gc.collect()

            # also save the state
            if cfg.save_training_state:
                # if you use deepspeed, recommend to save state via accelerator.save_state
                # and no need to use accelerator.is_main_process before saving state.
                # ref to https://github.com/huggingface/diffusers/issues/2606
                accelerator.save_state(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "state", "training_state.pth"))
                accelerator.print(f"Saved state checkpoint at epoch {n_iter}.")

            torch.cuda.empty_cache()
    accelerator.end_training()


if __name__ == "__main__":
    train()

