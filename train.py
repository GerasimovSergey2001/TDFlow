import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from copy import deepcopy

from src.model.flow_matching import ConditionalFlowMatching
from src.model.base_models.unet_mlp import TDFlowUnet
from src.training import TDFlowTrainer
from src.datasets import PointMassMazeDataset

task = 'reach_top_left'

def main():
    dataset = PointMassMazeDataset(task=task)
    train_loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True)
    gamma = 0.99
    num_epochs = 500
    ema = 1e-3
    optimizer_config = {
        'lr':1e-4,
        'weight_decay': 1e-3
    }
    velocity = TDFlowUnet()
    fm = ConditionalFlowMatching(velocity, obs_dim=(4, ))
    fm_target = deepcopy(fm)
    trainer = TDFlowTrainer(
        fm=fm,
        fm_target=fm_target,
        train_loader=train_loader,
        optimizer_config=optimizer_config,
        gamma=gamma,
        ema=ema,
        device='auto'
    )
    trainer.fit(num_epochs)
    torch.save(trainer.fm.model.state_dict(), 'td2_cfm_model.pth')

if __name__ == '__main__':
    main()