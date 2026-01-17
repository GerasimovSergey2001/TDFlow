import torch
import torch.nn as nn

import wandb
from tqdm import tqdm

class TDFlowTrainer:
    def __init__(self, 
                 fm, 
                 fm_target, 
                 train_loader,
                 optimizer_config,
                 gamma, 
                 ema,
                 project_name="TDFlow-Project",
                 device = 'auto'
                 ):
        
        if device=='auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.fm = fm.to(self.device)
        self.fm_target = fm_target.to(self.device)
        self.fm_target.freeze_model()
        self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(self.fm.parameters(), **optimizer_config)
        self.project_name = project_name    
        self.gamma, self.ema = gamma, ema
        self.batch_size = self.train_loader.batch_size

        wandb.init(project=project_name, config={
            "gamma": gamma,
            "ema": ema,
            "optimizer": "AdamW",
            **optimizer_config
        })
    
    def fit(self, num_epochs=500):
        self.global_step = 0
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            epoch_l1, epoch_l2, epoch_loss = 0, 0, 0
            for s, a, s_next, a_next in pbar:
                self.global_step += 1
                s, a, s_next, a_next = s.to(self.device), a.to(self.device), s_next.to(self.device), a_next.to(self.device)
                cond = torch.cat([s, a], dim=-1)
                cond_next = torch.cat([s_next, a_next], dim=-1)

                t_batch = torch.rand(self.batch_size, device=self.device)
                x0 = torch.randn((self.batch_size, s.shape[1]), device=self.device)

                x_target = self.fm_target.sample(self.batch_size, cond_next, t_batch)
                velocity_target = self.fm_target.velocity(t_batch, x_target, cond_next)

                l1 = self.fm.criterion(t_batch, x0, s, cond)    
                l2 = self.second_term_criterion(self.fm, velocity_target, t_batch, x_target, cond)
                loss = (1-self.gamma)*l1 + self.gamma*l2
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_parameters()

                epoch_l1 += l1.item()
                epoch_l2 += l2.item()
                epoch_loss += loss.item()
                
                pbar.set_postfix({"loss": loss.item()})

                avg_loss = epoch_loss / len(self.train_loader)
                pbar.set_postfix({"loss": loss.item(), "step": self.global_step})

            wandb.log({
                "epoch": epoch,
                "l1_flow_matching": epoch_l1 / len(self.train_loader),
                "l2_bootstrap": epoch_l2 / len(self.train_loader),
                "total_loss": avg_loss,
                "global_step": self.global_step
                }, 
                step=self.global_step)

    def second_term_criterion(self, fm, target, t, x, cond):
        v = fm.velocity(t, x, cond)
        dim = tuple(torch.arange(1, len(x.shape)))
        return torch.mean((v - target).pow(2).sum(dim=dim)) 


    def update_parameters(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.fm_target.parameters(), self.fm.parameters()):
                target_param.data.mul_(1.0-self.ema ).add_(online_param.data, alpha=self.ema)
        