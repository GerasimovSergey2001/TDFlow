import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim=64, num_layers=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Sequential(nn.Linear(inp_dim+1, hidden_dim), nn.GELU())
        self.modules = []
        for _ in range(num_layers):
            self.modules.append(nn.Linear(hidden_dim, hidden_dim))
            self.modules.append(nn.GELU())
        self.mlp = nn.Sequential(*self.modules)
        self.out = nn.Linear(hidden_dim, inp_dim)

    def forward(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        t = t.unsqueeze(0).reshape(-1,1)
        x = torch.cat([x, t], dim=-1)
        x = self.input(x)
        x = self.mlp(x)
        return self.out(x)
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info