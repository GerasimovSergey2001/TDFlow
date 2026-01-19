from huggingface_hub import hf_hub_download
import torch
import numpy as np
from stable_baselines3 import TD3
from torch.utils.data import Dataset
from tqdm import tqdm


class PointMassMazeDataset(Dataset):

    def __init__(self, file_path='data/point_mass_maze/rnd/buffer/', task='reach_top_left', device='auto'):
        self.task = task

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        model_path = hf_hub_download(repo_id ="SergeiGerasimov/TDFlow", filename=f"td3_point_mass_expert_{task}.zip")
        model =  TD3.load(model_path)
        self.policy = self.configure_policy(model)
        self.file_path = file_path
        self.s = []
        self.a = []
        self.s_next = []
        self.a_next = []
        for i in tqdm(range(0, 10_000), desc="Dataset Processing"):
            idx = f"{i}"
            while len(idx)<6:
                idx = '0'+idx
            data = np.load(file_path+f'episode_{idx}_1000.npz')
            self.s.append(data['observation'][:-1])
            self.s_next.append(data['observation'][1:])
            self.a.append(data['action'][1:])
            self.a_next.append(self.policy(data['observation'][1:]))

        self.s = torch.from_numpy(np.concat(self.s)).to(torch.float32)
        self.s_next = torch.from_numpy(np.concat(self.s_next)).to(torch.float32)
        self.a = torch.from_numpy(np.concat(self.a)).to(torch.float32)
        self.a_next = torch.from_numpy(np.concat(self.a_next)).to(torch.float32)

    def configure_policy(self, model):
        model.policy.to(self.device)
        model.policy.set_training_mode(False)

        def policy(s):
            obs_tensor = torch.as_tensor(s, device=self.device, dtype=torch.float32)
            obs_dict = {
                'position': obs_tensor[:, :2],
                'velocity': obs_tensor[:, 2:]
            }

            with torch.no_grad():

                actions = model.policy.actor(obs_dict)
            
            return actions.cpu().numpy()
            
        return policy
    
    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.s_next[idx], self.a_next[idx]