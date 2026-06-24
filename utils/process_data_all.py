import sys

sys.path.append("./")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml
import shutil


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_qpos, left_endpose = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
            root["/endpose/left_endpose"][()],
        )
        right_gripper, right_qpos, right_endpose = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
            root["/endpose/right_endpose"][()],
        )

        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_qpos, left_endpose, right_gripper, right_qpos, right_endpose, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        # Convert RGB back to BGR before encoding (dataset expects BGR format)
        img_bgr = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        success, encoded_image = cv2.imencode(".jpg", img_bgr)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return padded_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_qpos_all, left_endpose_all, right_gripper_all, right_qpos_all, right_endpose_all, \
            image_dict = load_hdf5(os.path.join(path, f"episode{i}.hdf5"))
        state_qpos_list = []
        state_eep_list = []
        actions_qpos = []
        actions_eep = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_qpos, left_endpose, right_gripper, right_qpos, right_endpose = (
                left_gripper_all[j],
                left_qpos_all[j],
                left_endpose_all[j],
                right_gripper_all[j],
                right_qpos_all[j],
                right_endpose_all[j],
            )

            state_qpos = np.concatenate((left_qpos, [left_gripper], right_qpos, [right_gripper]), axis=0).astype(np.float32)  # joint
            state_eep = np.concatenate((left_endpose, [left_gripper], right_endpose, [right_gripper]), axis=0).astype(np.float32)  # endpose

            if j != left_gripper_all.shape[0] - 1:

                state_qpos_list.append(state_qpos)
                state_eep_list.append(state_eep)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_rgb = cv2.cvtColor(camera_high, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                camera_high_resized = cv2.resize(camera_high_rgb, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_rgb = cv2.cvtColor(camera_right_wrist, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                camera_right_wrist_resized = cv2.resize(camera_right_wrist_rgb, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_rgb = cv2.cvtColor(camera_left_wrist, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                camera_left_wrist_resized = cv2.resize(camera_left_wrist_rgb, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                actions_qpos.append(state_qpos)
                actions_eep.append(state_eep)

        if not os.path.exists(os.path.join(save_path, f"episode_{i}")):
            os.makedirs(os.path.join(save_path, f"episode_{i}"))
        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("actions_qpos", data=np.array(actions_qpos))
            f.create_dataset("actions_eep", data=np.array(actions_eep))
            obs = f.create_group("observations")
            obs.create_dataset("state_qpos", data=np.array(state_qpos_list))
            obs.create_dataset("state_eep", data=np.array(state_eep_list))

        # Save videos for three camera views
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        video_high_path = os.path.join(save_path, f"episode_{i}", f"cam_high.mp4")
        video_right_path = os.path.join(save_path, f"episode_{i}", f"cam_right_wrist.mp4")
        video_left_path = os.path.join(save_path, f"episode_{i}", f"cam_left_wrist.mp4")
        
        if len(cam_high) > 0:
            height, width = cam_high[0].shape[:2]
            video_writer = cv2.VideoWriter(video_high_path, fourcc, fps, (width, height))
            for frame in cam_high:
                video_writer.write(frame)
            video_writer.release()
        
        if len(cam_right_wrist) > 0:
            height, width = cam_right_wrist[0].shape[:2]
            video_writer = cv2.VideoWriter(video_right_path, fourcc, fps, (width, height))
            for frame in cam_right_wrist:
                video_writer.write(frame)
            video_writer.release()
        
        if len(cam_left_wrist) > 0:
            height, width = cam_left_wrist[0].shape[:2]
            video_writer = cv2.VideoWriter(video_left_path, fourcc, fps, (width, height))
            for frame in cam_left_wrist:
                video_writer.write(frame)
            video_writer.release()

        begin += 1
        print(f"proccess {i} success!")

    return begin




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("--task_name", type=str, default="adjust_bottle")
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--expert_data_num", type=int, default=50)
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../robotwin_raw_depth", str(task_name), str(task_config), "data")

    print(f"read data from path: {load_dir}")

    begin = data_transform(
        load_dir,
        expert_data_num,
        f"./processed_data/{task_name}-{task_config}-{expert_data_num}",
    )

    tokenizer, text_encoder = None, None
    for idx in range(expert_data_num):
        print(f"Processing Language: {idx}", end="\r")
        data_file_path = (f"../../robotwin_raw_dataset/{task_name}/{task_config}/instructions/episode{idx}.json")
        target_dir = (f"processed_data/{task_name}-{task_config}-{expert_data_num}/episode_{idx}")
        os.makedirs(target_dir, exist_ok=True)
        save_dir = os.path.join(target_dir, f"instructions.json")
        shutil.copy2(data_file_path, save_dir)

