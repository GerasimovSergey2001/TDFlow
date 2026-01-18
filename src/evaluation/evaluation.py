import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import ot
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.model.base_models.unet_mlp import TDFlowUnet
from src.model.flow_matching import ConditionalFlowMatching
from src.evaluation.utils import force_set_state, make_env


class Evaluator:
    def __init__(self, model="td2_cfm", task='reach_top_left', epoch=None, 
                 num_samples=64, gamma=0.99, device='auto', disable=False):
        self.repo_id = "SergeiGerasimov/TDFlow"
        if device=='auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.disable = disable
        model_path = f"{model}_model_{task}.pth" if epoch is None else f"{model}_model_{task}_epoch_{epoch}.pth"
        checkpoint_path = hf_hub_download(repo_id=self.repo_id, filename=model_path)

        self.fm = ConditionalFlowMatching(TDFlowUnet(), obs_dim=(4,))
        self.fm.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        self.fm = self.fm.to(device)
        self.fm.freeze_model()

        self.task = task
        self.num_samples = num_samples
        self.device = device
        
        self.vec_env = VecMonitor(DummyVecEnv([lambda: make_env(task) for _ in range(1)]))
        td3_link = hf_hub_download(repo_id ="SergeiGerasimov/TDFlow", filename=f"td3_point_mass_expert_{task}.zip")
        self.td3 = TD3.load(td3_link, device=self.device)
        self.gamma = gamma

    def sample_states(self):
        self.actions = []
        self.trajectories = []
        self.rewards = []
        self.stopping_times = []
        self.initial_cond = []
        for _ in tqdm(range(self.num_samples), desc="Generating expert trajectories", disable=self.disable):
            x = np.random.uniform(-0.29, -0.15)
            y = np.random.uniform(0.15, 0.29)
            vx = 0.0 #np.random.uniform(-0.01, 0.01)
            vy = 0.0 #np.random.uniform(-0.01, 0.01)
            # initial_action = np.random.uniform(-1.0, 1.0, size=(1, 2))
            obs = force_set_state(self.vec_env, target_state=np.array([x,  y, vx, vy]))
            initial_action, _ = self.td3.predict(obs, deterministic=True)
            state = [np.concat([obs['position'], obs['velocity']], axis=-1)] # set initial state
            actions = [initial_action] # set initial action
            obs, r, terminated, _ = self.vec_env.step(actions[0])
            state.append(np.concat([obs['position'], obs['velocity']], axis=-1))
            reward = [r]
            cond = np.concatenate([np.array([[x,  y, vx, vy]]), actions[0]], axis=-1)
            self.initial_cond.append(cond)
            for t in range(1000):      
                action, _ = self.td3.predict(obs, deterministic=True)
                obs, r, terminated, _ = self.vec_env.step(action)
                reward.append(r)
                self.actions.append(action)
                state.append(np.concat([obs['position'], obs['velocity']], axis=-1))
                if terminated:
                    break
            self.rewards.append(reward)
            T = np.minimum(np.random.geometric(1-self.gamma, 2048), len(state)-1)
            self.trajectories.append(state)
            self.rewards.append(reward)
            self.stopping_times.append(T)
        return np.array([np.array(state)[T].squeeze() for state, T in zip(self.trajectories, self.stopping_times)])
    
    def sample_model_states(self):
        self.fm.eval()
        self.initial_cond = torch.from_numpy(np.array(self.initial_cond)).squeeze().to(torch.float32)
        self.initial_cond = self.initial_cond[:,None, :].repeat(1, 2048, 1).to(self.device)
        t_batch = torch.ones(2048).to(self.device)
        gen_samples = []
        for i in tqdm(range(self.num_samples), desc="Generating model samples", disable=self.disable):
            gen_samples.append(self.fm.sample(2048, self.initial_cond[i], t_batch).cpu().numpy())
        return np.array(gen_samples)

    def compute_emd(self, states_target, states_model):
        M = ot.dist(states_target, states_model, metric='euclidean')
        a, b = np.ones((2048,)) / 2048, np.ones((2048,)) / 2048
        value = ot.emd2(a, b, M)
        return value
    
    def get_reward(self, state):
        wrapper = self.vec_env.envs[0].unwrapped
        dm_env = getattr(wrapper, '_env', wrapper)

        obs = force_set_state(self.vec_env, target_state=state)

        dm_env.physics.forward() 

        reward = dm_env.task.get_reward(dm_env.physics)
        
        return reward
        
    def evaluate(self):
        states_target = self.sample_states()
        states_model = self.sample_model_states()
        all_emds = []
        msev = []
        for i in tqdm(range(self.num_samples), desc="Computing EMDs", disable=self.disable):
            value = self.compute_emd(states_target[i], states_model[i])
            all_emds.append(value)
            r_mc = np.sum([r*(self.gamma**j) for j, r in enumerate(self.rewards[i])])
            r_model = np.mean([self.get_reward(states_model[i][j]) for j in range(len(states_model[i]))])/(1-self.gamma)
            msev.append((r_model-r_mc)**2)

        # nll
        idx = torch.randperm(self.num_samples)[0:64]
        states_nll = torch.from_numpy(states_target[:, :, :4]).view(-1,4).to(self.device).to(torch.float32)
        cond = self.initial_cond.view(-1, 6).to(self.device).to(torch.float32)
        idx = torch.randperm(cond.shape[0])[:2048]
        nll = -torch.mean(self.fm.logp(x1=states_nll[idx], cond=cond[idx], n_samples=5))/4.0
        return {
            'EMD':np.mean(all_emds), 
            'MSE(V)':np.mean(msev),
            'NLL':nll.item()
            }