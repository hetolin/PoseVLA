# ===== Environment variables (must be set before importing other libraries) =====
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1"  # IMPORTANT!!! MUST BE COMMENTED, SET to 0
# os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

# ===== Standard library =====
import copy
import gc
import random
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ===== Third-party libraries =====
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import logging

# ===== Local modules: model =====
from pi0.modeling_pi0 import PI0Config, PI0Policy, bin_tokenizer

# ===== Local modules: dataset / dataloader factory =====
from data.factory import build_action_dataloader, build_vlm_dataloaders

# ===== Local modules: utilities and evaluation =====
from eval_gemini import compute_metrics_summary, evaluate_sample
from utils.mapping_token import decode_text_to_scene_with_tokenizer
from utils.logger import initialize_wandb, register_features_types, save_wandb
from utils.vis import (
    plot_all_joints,
    vis_atten_map,
    visualize_2d_3d_all,
)

# ===== Global logging configuration =====
# logging.set_verbosity_info()    # show info and warnings
logging.set_verbosity_error()     # only show errors

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

def safe_mean(x):
    return np.mean(x) if len(x) > 0 else 0.
@hydra.main(
        version_base=None,
        config_path="./config",
        config_name="base",
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
                              # dispatch_batches=False,
                              mixed_precision=cfg.training.mixed_precision,
                              gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
                              # kwargs_handlers=[ddp_kwargs],
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=cfg.deepspeed),
                              project_config=accelerator_project_config,
                              # cpu=False
                              )
    if not cfg.debug:
        # if accelerator.is_local_main_process:
        if accelerator.is_main_process:
            if cfg.resume_ckpt:
                cfg_prev = OmegaConf.load(Path(cfg.resume_ckpt).parent / "base.yaml")
                cfg = OmegaConf.merge(cfg_prev, {"resume_ckpt": cfg.resume_ckpt})
                cfg.ckpt_save_dir = os.path.dirname(cfg.resume_ckpt)
                cfg = initialize_wandb(cfg) # load wandb.run.id
            else:
                cfg = initialize_wandb(cfg)
                wandb.config.update(OmegaConf.to_container(cfg))
                os.makedirs(cfg.ckpt_save_dir, exist_ok=True)
                save_wandb(wandb, cfg.ckpt_save_dir)
                OmegaConf.save(cfg, Path(cfg.ckpt_save_dir) / "base.yaml")

        accelerator.wait_for_everyone()
    # print(f"Training Pi0 Model on `{list(cfg.dataset.sample_weights.keys())}` from `{cfg.dataset.hdf5_dir}`")
    # set_global_seed(cfg.seed) # TODO: should be comment, as it will reduce the performance

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32

    # with accelerator.main_process_first():

    ############# Config Model ############
    accelerator.print("Initialize Model")
    pi0_config = PI0Config(
        tokenizer_model_path=(cfg.model.tokenizer_model_path),
        n_action_steps=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1,
        chunk_size=cfg.dataset.action_chunk_size + cfg.dataset.img_history_size - 1, # not used
        optimizer_lr = cfg.training.optimizer_lr,
        optimizer_betas = tuple(cfg.training.optimizer_betas),
        optimizer_eps = cfg.training.optimizer_eps,
        optimizer_weight_decay= cfg.training.optimizer_weight_decay,
        scheduler_warmup_steps = cfg.training.scheduler_warmup_steps,
        scheduler_decay_steps = cfg.training.scheduler_decay_steps,
        scheduler_decay_lr = cfg.training.scheduler_decay_lr,
        is_knowledge_insulation=cfg.training.is_knowledge_insulation,
        resize_imgs_with_padding=(cfg.dataset.image_size, cfg.dataset.image_size),
        pi05=cfg.training.pi05,
        vis_attn=cfg.training.vis_attn,
        add_extra_token=cfg.training.add_extra_token,
        add_image_token=cfg.training.add_image_token,
        add_prior=cfg.training.add_prior,
        # 仅在主流程（from_pretrained 完整覆盖权重）下安全；
        # 若使用 PI0Policy(pi0_config) 从零构建 + load_pretrained_vlm，请保持 False。
        skip_init_weights=False)

    if cfg.resume_ckpt:
        accelerator.print(f"Resuming from checkpoint {cfg.resume_ckpt}")
        policy = PI0Policy.from_pretrained(os.path.join(cfg.resume_ckpt, "model"), local_files_only=True)
    else:
        '''pi0/pi05'''
        # policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)
        # print(f"load pretrain model from: {cfg.model.pretrained_model_path}")

        '''paligemma+no_expert'''
        policy = PI0Policy(pi0_config)
        policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")
        print(f"load pretrain VLM from: pretrain/paligemma-3b-pt-224")

        '''vlm3d/4d+no expert'''
        # policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)
        # print(f"load pretrain model from: {cfg.model.pretrained_model_path}")
        # _policy = PI0Policy(pi0_config)
        # policy.model.paligemma_with_expert.gemma_expert = _policy.model.paligemma_with_expert.gemma_expert

        '''vlm3d: pi0 action expert + paligemma VLM'''
        # policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config, strict=False)
        # print(f"load pretrain model from: {cfg.model.pretrained_model_path}")
        # policy.load_pretrained_vlm("pretrain/paligemma-3b-pt-224")
        # print(f"load pretrain VLM from: pretrain/paligemma-3b-pt-224")

        # train only VLM
        # policy = PI0Policy.from_pretrained(cfg.model.pretrained_model_path, config=pi0_config)
        # policy.load_pretrained_vlm("google/paligemma-3b-pt-224", local_files_only=False)
        # print(f"load pretrain VLM model from: google/paligemma-3b-pt-224, action expert from pi0")

    del policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens
    del policy.model.paligemma_with_expert.gemma_expert.lm_head

    gc.collect()
    policy.to(weight_dtype)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    ############# Config Optimizer ############
    accelerator.print("Initialize Optimizer")
    total_train_steps = cfg.training.max_training_steps * accelerator.num_processes * cfg.training.grad_accumulation_steps
    optimizer = pi0_config.get_optimizer_preset().build(filter(lambda p: p.requires_grad, policy.parameters()))
    pi0_config.scheduler_warmup_steps *= accelerator.num_processes
    pi0_config.scheduler_decay_steps *= accelerator.num_processes
    lr_scheduler = pi0_config.get_scheduler_preset().build(optimizer, total_train_steps)

    ############# Config Dataset and Dataloader ############
    accelerator.print("Initialize Dataset and Dataloader")
    # Dataset and DataLoaders creation:

    # Load vlm dataset
    if cfg.co_training.vlm_training:
        train_vlm_dataloader, val_vlm_dataloader, train_vlm_dataset = build_vlm_dataloaders(
            cfg, pi0_config, bin_tokenizer
        )

    # Load action dataset
    if cfg.co_training.action_training:
        train_dataloader, train_dataset = build_action_dataloader(cfg, pi0_config)


    ############# Preapare `accelerator` ############
    # Prepare everything with our `accelerator`.
    # will cast the parameters of model into mix-precision type in deepspeed mode
    # vlm + action
    if cfg.co_training.vlm_training and not cfg.co_training.action_training:
        print("Prepare vlm dataloader")
        policy, optimizer, train_vlm_dataloader, val_vlm_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_vlm_dataloader, val_vlm_dataloader, lr_scheduler
        )
    elif not cfg.co_training.vlm_training and cfg.co_training.action_training:
        print("Prepare action dataloader")
        policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_dataloader, lr_scheduler
        )
    elif cfg.co_training.vlm_training and cfg.co_training.action_training:
        print("Prepare vlm + action dataloader")
        policy, optimizer, train_vlm_dataloader, val_vlm_dataloader, train_dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_vlm_dataloader, val_vlm_dataloader, train_dataloader, lr_scheduler
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
    accelerator.print(f"  statistics_path_6d_dataset: {cfg.statistics_path_6d_dataset}")
    accelerator.print(f"  max_norm_6d_dataset: {cfg.max_norm_6d_dataset}")
    accelerator.print(f"  uniform_mapping_6d_dataset: {cfg.uniform_mapping_6d_dataset}")

    train_loss = []
    train_metrics = ["loss"]
    train_log_dict = {name: {m: [] for m in train_metrics} for name in cfg.co_training.keys()}
    val_metrics = ["loss", "dist", "gt", "pred", "attn", "mAP"]
    val_log_dict = {name: {m: [] for m in val_metrics} for name in cfg.co_training.keys()}

    if cfg.co_training.vlm_training:
        train_vlm_loader_iter = cycle(train_vlm_dataloader)
        val_vlm_loader_iter = cycle(val_vlm_dataloader)
    if cfg.co_training.action_training:
        train_loader_iter = cycle(train_dataloader)
        # val_loader_iter = iter(val_dataloader)
    current_step = int(cfg.resume_ckpt.split("/")[-1]) + 1 if cfg.resume_ckpt else 0
    progress_bar = tqdm(range(cfg.training.max_training_steps), ncols=100, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")
    progress_bar.update(current_step)

    for n_iter in range(current_step, cfg.training.max_training_steps):
        policy.train()
        with accelerator.accumulate(policy):
            train_configs = []
            if cfg.co_training.vlm_training:
                train_configs.append(("vlm_training", train_vlm_loader_iter))
            if cfg.co_training.action_training:
                train_configs.append(("action_training", train_loader_iter))

            loss_dict = {}
            for name, loader_iter in train_configs:
                loss_dict[name] = {}
                if not getattr(cfg.co_training, name):
                    continue

                batch = next(loader_iter)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(weight_dtype)
                output = policy.forward(batch)
                loss_dict[name] = output["loss"]

                if name == "vlm_training":
                    train_log_dict[name]["loss"].append(output["ntp_loss"])
                elif name == "action_training":
                    train_log_dict[name]["loss"].append(output["flow_loss"])

            if cfg.training.is_knowledge_insulation:
                loss = loss_dict.get("vlm_training", 0.) + loss_dict.get("action_training", 0.)
            else:
                loss = loss_dict.get("vlm_training", 0.) + 10.0 * loss_dict.get("action_training", 0.)
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
            if (n_iter+1) % (cfg.training.train_frequency * cfg.training.grad_accumulation_steps) == 0 and accelerator.is_main_process and not cfg.debug:

                wandb.log(
                    {
                        "train/loss": safe_mean(train_loss),
                        "train/ntp_loss": safe_mean(train_log_dict["vlm_training"]["loss"]),
                        "train/flow_loss": safe_mean(train_log_dict["action_training"]["loss"]),
                        "train/iter": n_iter,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                )
                train_loss = []
                train_log_dict = {name: {m: [] for m in train_metrics} for name in cfg.co_training.keys()}

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": safe_mean(train_loss)})

        # validation
        if (n_iter + 1) % (cfg.training.eval_frequency * cfg.training.grad_accumulation_steps) == 0:
            accelerator.wait_for_everyone()
            accelerator.print(f"\n Evaluate at n_iter {n_iter}.")
            eval_model = accelerator.unwrap_model(policy)
            eval_model.eval()
            for i in tqdm(range(cfg.training.max_evaluation_steps), desc="Validation", ncols=100, disable=not accelerator.is_local_main_process):
                train_configs = []
                if cfg.co_training.vlm_training:
                    train_configs.append(("vlm_training", val_vlm_loader_iter))
                if cfg.co_training.action_training:
                    train_configs.append(("action_training", train_loader_iter))

                for name, loader_iter in train_configs:
                    if not getattr(cfg.co_training, name):
                        continue

                    with torch.no_grad():
                        batch = next(loader_iter)
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.to(weight_dtype)
                        output = eval_model(batch)

                        if name == "vlm_training":
                            loss = output['loss']
                            output_res = eval_model.forward_evaluate_ntp(batch)
                            pred, gt = output_res["pred"], output_res["gt"]
                            val_log_dict[name]["pred"] = pred
                            val_log_dict[name]["gt"] = gt

                            all_loss = accelerator.gather_for_metrics((loss))
                            pred_res = decode_text_to_scene_with_tokenizer(pred[0], bin_tokenizer)
                            if cfg.data_3d:
                                val_log_dict[name]["mAP"].append(evaluate_sample(pred_res, batch, camera_idx=0))
                            dist = torch.tensor(0.0, device=loss.device)
                            all_loss = all_loss.mean()

                        elif name == "action_training":
                            loss = output['loss']
                            output_res = eval_model.forward_evaluate(batch)
                            pred, gt = output_res["pred"], output_res["gt"]
                            val_log_dict[name]["pred"] = pred
                            val_log_dict[name]["gt"] = gt
                            if pi0_config.vis_attn:
                                attn = output_res["attn"]
                                val_log_dict[name]["attn"] = attn

                            all_predictions, all_targets, all_loss = accelerator.gather_for_metrics((pred, gt, loss))
                            # dist = (all_predictions - all_targets).abs().mean()
                            dist = (all_predictions[:, :, :14] - all_targets[:, :, :14]).abs().mean()
                            all_loss = all_loss.mean()

                        val_log_dict[name]["loss"].append(all_loss.item())
                        val_log_dict[name]["dist"].append(dist.item())

            eval_model.train()
            if accelerator.is_main_process and not cfg.debug:
                if cfg.data_3d:
                    metric_3d = compute_metrics_summary(val_log_dict["vlm_training"]["mAP"], "./train_AP", True)
                wandb.log(
                    {
                        "val/vlm_loss": safe_mean(val_log_dict["vlm_training"]["loss"]),
                        "val/vlm_dist": safe_mean(val_log_dict["vlm_training"]["dist"]),
                        "val/vlm_mAP": metric_3d["mean_AP"] if cfg.data_3d else 0,
                        "val/action_loss": safe_mean(val_log_dict["action_training"]["loss"]),
                        "val/action_dist": safe_mean(val_log_dict["action_training"]["dist"]),
                        "val/iter": n_iter,
                    }
                )

                # [(b, hist, 3, h, w), ..., (b, hist, 3, h, w)] -> [(3, h, w), ..., (3, h, w)]
                images = {"image0": batch['observation.images.top_head'][0, -1, ...],
                          "image1": batch['observation.images.hand_left'][0, -1, ...],
                          "image2": batch['observation.images.hand_right'][0, -1, ...]}
                intrinsics = {"image0": batch.get("observation.images.top_head.intrinsics", [None])[0],
                              "image1": batch.get("observation.images.hand_left.intrinsics", [None])[0],
                              "image2": batch.get("observation.images.hand_right.intrinsics", [None])[0]}

                if cfg.co_training.vlm_training:
                    pred_text = val_log_dict["vlm_training"]["pred"][0]
                    gt_text = val_log_dict["vlm_training"]["gt"][0]

                    pred_res = decode_text_to_scene_with_tokenizer(pred_text, bin_tokenizer)
                    gt_res = decode_text_to_scene_with_tokenizer(gt_text, bin_tokenizer)
                    fig = visualize_2d_3d_all(images, gt_res, pred_res, intrinsics, batch['task'][0], wandb=True)

                    wandb.log({"all_flow_subplot": wandb.Image(fig),
                               "text_flow": wandb.Table(data=[[pred_text, gt_text]], columns=["Text_pred", "Text_gt"])
                               })

                if cfg.co_training.action_training:
                    fig = plot_all_joints(val_log_dict["action_training"]["gt"][0].float().detach().cpu().numpy(),
                                          val_log_dict["action_training"]["pred"][0].float().detach().cpu().numpy(), wandb=True)
                    wandb.log({"all_angles_subplot": wandb.Image(fig)})
                    plt.close()

                    if pi0_config.vis_attn:
                        fig = vis_atten_map(val_log_dict["action_training"]["attn"], list(images.values()), wandb=True)
                        wandb.log({f"all_attns_subplot": wandb.Image(fig)})

                val_log_dict = {name: {m: [] for m in val_metrics} for name in cfg.co_training.keys()}

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
                # register_features_types()

                # safe_serialization=False:
                # https://github.com/huggingface/transformers/issues/27293
                # https://zhuanlan.zhihu.com/p/705206155
                model_to_save.save_pretrained(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "model"), safe_serialization=False)
                model_to_save.language_tokenizer.save_pretrained(os.path.join(cfg.ckpt_save_dir, f"{n_iter}", "model"), safe_serialization=False)
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