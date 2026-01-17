import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
from copy import deepcopy
import wandb

from src.model.flow_matching import ConditionalFlowMatching
from src.model.base_models.unet_mlp import TDFlowUnet
from src.training import TDFlowTrainer
from src.datasets import PointMassMazeDataset



def main(task = 'reach_top_left', num_epochs=100):
    dataset = PointMassMazeDataset(task=task)
    train_loader = DataLoader(dataset=dataset, batch_size=1024, shuffle=True)
    gamma = 0.99
    # num_epochs = 500
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
        device='auto',
        task=task
    )
    try:
        trainer.fit(num_epochs)
    finally:
        wandb.finish()
    torch.save(trainer.fm.model.state_dict(), f'checkpoints/td2_cfm_model_{task}.pth')
    torch.save(trainer.fm.model.state_dict(), f'checkpoints/td2_cfm_target_model_{task}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='reach_top_left')
    parser.add_argument('--num_epochs', type=int, default=100)    
    args = parser.parse_args()
    
    print(f"Starting training with args: {args}")
    main(
        task=args.task, 
        num_epochs=args.num_epochs
        )