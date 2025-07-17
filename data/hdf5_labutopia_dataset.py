import os
import fnmatch
import yaml
import h5py
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5LabutopiaDataset:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # 读取 base.yaml 配置
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.h5_file = h5py.File(h5_path, 'r')
        self.episode_map = []
        for episode_name in self.h5_file.keys():
            n_frames = self.h5_file[episode_name]['actions'].shape[0]
            self.episode_map.append((episode_name, n_frames))

        episode_lens = [n for _, n in self.episode_map]
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.episode_map)

    def get_dataset_name(self):
        return "labutopia"

    def get_item(self, index=None, state_only=False):
        while True:
            if index is None:
                idx = np.random.choice(len(self.episode_map), p=self.episode_sample_weights)
            else:
                idx = index
            episode_name, n_frames = self.episode_map[idx]
            episode = self.h5_file[episode_name]
            if n_frames < self.CHUNK_SIZE + self.IMG_HISORY_SIZE:
                index = np.random.randint(0, len(self.episode_map))
                continue
            if state_only:
                return self.parse_hdf5_file_state_only(episode)
            else:
                return self.parse_hdf5_file(episode, n_frames)

    def parse_hdf5_file(self, episode, n_frames):
        # 随机采样一个step
        step_id = np.random.randint(self.IMG_HISORY_SIZE-1, n_frames-self.CHUNK_SIZE)
        cam1 = episode['camera_1_rgb'][step_id-self.IMG_HISORY_SIZE+1:step_id+1]
        cam3 = episode['camera_3_rgb'][step_id-self.IMG_HISORY_SIZE+1:step_id+1]
        cam1 = cam1.astype(np.float32) / 255.0
        cam3 = cam3.astype(np.float32) / 255.0

        agent_pose = episode['agent_pose'][step_id:step_id+1]
        actions = episode['actions'][step_id:step_id+self.CHUNK_SIZE]
        instruction = episode['language_instruction'][step_id:step_id+1]

        def fill_in_state(values):
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(values.shape[-1])
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        state = fill_in_state(agent_pose)
        actions = fill_in_state(actions)
        state_std = np.std(agent_pose, axis=0)
        state_mean = np.mean(agent_pose, axis=0)
        state_norm = np.sqrt(np.mean(agent_pose**2, axis=0))
        state_indicator = np.ones_like(state)

        valid_len = self.IMG_HISORY_SIZE
        cam_mask = np.array([True]*valid_len)
        cam_high = cam1
        cam_high_mask = cam_mask
        cam_left_wrist = np.zeros_like(cam1)
        cam_left_wrist_mask = np.zeros_like(cam_mask, dtype=bool)
        cam_right_wrist = cam3
        cam_right_wrist_mask = cam_mask

        meta = {
            "dataset_name": "labutopia",
            "#steps": n_frames,
            "step_id": step_id,
            "instruction": instruction
        }

        return {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }

    def parse_hdf5_file_state_only(self, episode):
        agent_pose = episode['agent_pose'][:]
        actions = episode['actions'][:]
        def fill_in_state(values):
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(values.shape[-1])
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        state = fill_in_state(agent_pose)
        actions = fill_in_state(actions)
        return {
            "state": state,
            "action": actions
        }

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

if __name__ == "__main__":
    # 示例用法
    ds = HDF5LabutopiaDataset(
        h5_path="/home/ubuntu/Documents/LabSim/outputs/collect/2025.07.16/00.19.30_Level3_PourLiquid_Single_Material/dataset/episode_data.hdf5",
    )
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        sample = ds.get_item(i)
        print(sample.keys())
        break