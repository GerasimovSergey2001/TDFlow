import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return x + self.net(x)
    
class TDFlowUnet(nn.Module):
    def __init__(self, state_dim=4, cond_dim=6): # s dim = 4, a dim = 2 => cond_dim = 6
        super().__init__()
        self.time_enc = SinusoidalPosEmb(256)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 256),
            Mish(),
            nn.Linear(256, 256)
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 512), 
            Mish(),
            nn.Linear(512, 512), 
            Mish(),
            nn.Linear(512, 512)
        )
        
        self.down1 = nn.Linear(state_dim+512 + 256, 512) # Cond + Time
        self.res1 = ResBlock(512)
        
        self.mid = ResBlock(512)
        
        self.up1 = nn.Linear(512 + 512, 512) # Bottleneck + Skip connection
        self.res2 = ResBlock(512)

        self.act = Mish()

        self.final = nn.Linear(512, state_dim)
    
    def forward(self, t, x, cond):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        cond_emb = self.cond_encoder(cond)
        t_emb = self.time_mlp(self.time_enc(t))

        x = torch.cat([x, t_emb, cond_emb], dim=-1)

        down1 = self.act(self.down1(x))
        res1 = self.res1(down1)
        mid = self.mid(res1)
        up1 = self.act(self.up1(torch.cat([res1, mid], dim=-1)))
        res2 = self.res2(up1)
        return self.final(res2)
