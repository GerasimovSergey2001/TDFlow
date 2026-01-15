import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class FlowMatchingBase(nn.Module):
    def __init__(self, model, obs_dim=(2,), sigma_min=1e-6, n_samples=10):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.obs_dim = obs_dim
        self.n_samples = n_samples
    
    def process_timesteps(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        if t.shape[0]!=x.shape[0] or len(t.shape)!=1:
            raise ValueError("Timesteps shape should (batch_size, )")
        return t

    def forward(self, t, x0, x1):
        """ 
        Computes velocity v from the equation dphi(t, x) = v(t, phi(t, x))dt. 
        """
        t = self.process_timesteps(t, x0)
        x = self.conditional_flow(t, x0, x1)
        return self.model(t, x)

    def velocity(self, t, x0):
        t = self.process_timesteps(t, x0)
        return self.model(t, x0)
    
    def reversed_velocity_with_div(self, t, state):
        s = 1-t
        x, logp = state
        x_ = x.detach().clone().requires_grad_(True)
        div_estimates = []
        with torch.set_grad_enabled(True):
            for i in range(self.n_samples):
                v = self.model(s, x_)
                is_last = (i == self.n_samples - 1)
                div_estimates.append(
                    self.approx_div(v, x_, retain_graph=not is_last)
                )
        
        mean_div = torch.stack(div_estimates).mean(dim=0)
        return ((-v).detach(), mean_div)
    
    def sigma(self, t, x1=None):
        return (1 - (1-self.sigma_min)*t)
    
    def dsigma_dt(self, t, x1):
        return - (1-self.sigma_min)
    
    def mu(self, t, x1):
        return t*x1
    
    def dmu_dt(self, t, x1):
        return x1

    def conditional_flow(self, t, x, x1):
        """
        Computes \phi(t,x) = \sigma(t, x1)x + \mu(t, x1), where \phi(0,x) = x = x0
        
        :param t: timestep. Float in [0,1].
        :param x0: starting point sampled from N(0, I).
        :param x1: observation
        """
        dims = [1]*(len(x.shape)-1)
        t = t.view(-1, *dims)
        return self.sigma(t, x1)*x + self.mu(t, x1)
    
    def conditional_velocity(self, t, x, x1, eps=1e-7):
        dims = [1]*(len(x.shape)-1)
        t = t.view(-1, *dims)
        return self.dsigma_dt(t, x1)/(self.sigma(t, x1) + eps)*(x-self.mu(t, x1)) + self.dmu_dt(t, x1)
    
    def target_velocity(self, t, x, x1):
        return self.conditional_velocity(t, self.conditional_flow(t, x, x1), x1)
    
    def criterion(self, t, x0, x1):
       v = self.forward(t, x0, x1)
       target = self.target_velocity(t, x0, x1)
       dim = tuple(torch.arange(1, len(x0.shape)))
       return torch.mean((v - target).pow(2).sum(dim=dim))
    
    def sample(self, n_samples, method='dopri5', rtol=1e-5, atol=1e-5):
        self.device = next(self.parameters()).device
        x0 = torch.randn([n_samples]+list(self.obs_dim), device=self.device)
        t = torch.linspace(0,1,2, device=self.device)
        with torch.no_grad():
            return odeint(self.model, x0, t, rtol=rtol, atol=atol, method=method)[-1,:,:]
        

    def approx_div(self, f_x, x, retain_graph=True):
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=True, retain_graph=retain_graph)[0]
        return (e_dzdx*z).view(z.shape[0], -1).sum(dim=1)

        
    def logp(self, x1, n_samples=50, rtol=1e-05, atol=1e-05):
        self.device = next(self.parameters()).device
        self.n_samples = n_samples
        t = torch.linspace(0, 1, 2, device=self.device )
        phi, f = odeint(
            self.reversed_velocity_with_div, 
            (x1, torch.zeros((x1.shape[0], 1))), 
            t, 
            rtol=rtol,
            atol=atol,
            adjoint_params = self.model.parameters(),
            )
        phi, f = phi[-1].detach().cpu(), f[-1].detach().cpu().flatten()
        logp_noise = -0.5 * (phi.pow(2).sum(1) + phi.shape[1] * torch.log(torch.tensor(2 * torch.pi)))
        return logp_noise - f
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.model.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
    


class ConditionalFlowMatching(nn.Module):
    def __init__(self, model, obs_dim=(2,), sigma_min=1e-6, n_samples=10):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.obs_dim = obs_dim
        self.n_samples = n_samples
    
    def process_timesteps(self, t, x):
        if len(t.shape) == 0:
            t = t.repeat(x.shape[0])
        if t.shape[0]!=x.shape[0] or len(t.shape)!=1:
            raise ValueError("Timesteps shape should be (batch_size, )")
        return t

    def forward(self, t, x0, x1, cond=None):
        """ 
        Computes velocity v from the equation dphi(t, x) = v(t, phi(t, x))dt. 
        """
        t = self.process_timesteps(t, x0)
        x = self.conditional_flow(t, x0, x1)
        return self.model(t, x, cond)

    def velocity(self, t, x, cond=None):
        t = self.process_timesteps(t, x)
        if cond is not None:
            return self.model(t, x, cond)
        else:
            return self.model(t, x)
    
    def sigma(self, t, x1=None):
        return (1 - (1-self.sigma_min)*t)
    
    def dsigma_dt(self, t, x1=None):
        return - (1-self.sigma_min)
    
    def mu(self, t, x1):
        return t*x1
    
    def dmu_dt(self, t, x1):
        return x1

    def conditional_flow(self, t, x0, x1):
        """
        Computes \phi(t,x|x1) = \sigma(t, x1)x + \mu(t, x1), where \phi(0,x|x1) = x = x0
        
        :param t: timestep. Float in [0,1].
        :param x0: starting point sampled from N(0, I).
        :param x1: observation
        """
        dims = [1]*(len(x0.shape)-1)
        t = t.view(-1, *dims)
        return self.sigma(t, x1)*x0 + self.mu(t, x1)
    
    def conditional_velocity(self, t, x, x1, eps=1e-7):
        dims = [1]*(len(x.shape)-1)
        t = t.view(-1, *dims)
        return self.dsigma_dt(t, x1)/(self.sigma(t, x1) + eps)*(x-self.mu(t, x1)) + self.dmu_dt(t, x1)
    
    def target_velocity(self, t, x0, x1):
        return self.conditional_velocity(t, self.conditional_flow(t, x0, x1), x1)
    
    def criterion(self, t, x0, x1, cond=None):
       v = self.forward(t, x0, x1, cond)
       target = self.target_velocity(t, x0, x1)
       dim = tuple(torch.arange(1, len(x0.shape)))
       return torch.mean((v - target).pow(2).sum(dim=dim))
    
    def sample(self, n_samples, cond=None, t=1, method='dopri5', rtol=1e-5, atol=1e-5):
        self.device = next(self.parameters()).device
        x0 = torch.randn([n_samples]+list(self.obs_dim), device=self.device)
        t = torch.linspace(0, t, 2, device=self.device)
        with torch.no_grad():
            return odeint(
                lambda t, x0: self.model(t, x0, cond), 
                x0, t, 
                rtol=rtol, 
                atol=atol, 
                method=method, 
                adjoint_params = self.model.parameters()
                )[-1,:,:]
    
    def reversed_velocity_with_div(self, t, state):
        s = 1-t
        x, cond, logp = state[0], state[1], state[2]
        x_ = x.detach().clone().requires_grad_(True)
        div_estimates = []
        with torch.set_grad_enabled(True):
            for i in range(self.n_samples):
                v = self.model(s, x_, cond)
                is_last = (i == self.n_samples - 1)
                div_estimates.append(
                    self.approx_div(v, x_, retain_graph=not is_last)
                )
        
        mean_div = torch.stack(div_estimates).mean(dim=0)
        return ((-v).detach(), torch.zeros_like(cond), mean_div)
        

    def approx_div(self, f_x, x, retain_graph=True):  # доделать позже
        z = torch.randint(low=0, high=2, size=x.shape).to(x) * 2 - 1
        e_dzdx = torch.autograd.grad(f_x, x, z, create_graph=True, retain_graph=retain_graph)[0]
        return (e_dzdx*z).view(z.shape[0], -1).sum(dim=1)

        
    def logp(self, x1, cond=None, n_samples=50, rtol=1e-05, atol=1e-05): # доделать позже
        self.device = next(self.parameters()).device
        self.n_samples = n_samples
        t = torch.linspace(0, 1, 2, device=self.device )
        f0 = torch.zeros((x1.shape[0], 1), device=self.device)

        def reversed_velocity_with_div(t, state):
            s = 1-t
            x, logp = state
            x_ = x.detach().clone().requires_grad_(True)
            div_estimates = []
            with torch.set_grad_enabled(True):
                for i in range(self.n_samples):
                    v = self.model(s, x_, cond)
                    is_last = (i == self.n_samples - 1)
                    div_estimates.append(self.approx_div(v, x_, retain_graph=not is_last))
            mean_div = torch.stack(div_estimates).mean(dim=0)
            return ((-v).detach(), mean_div)
        
        phi, f = odeint(
            reversed_velocity_with_div, 
            (x1, f0), 
            t, 
            rtol=rtol,
            atol=atol,
            adjoint_params = self.model.parameters(),
            )
        phi, f = phi[-1].detach().cpu(), f[-1].detach().cpu().flatten()
        logp_noise = -0.5 * (phi.pow(2).sum(1) + phi.shape[1] * torch.log(torch.tensor(2 * torch.pi)))
        return logp_noise - f
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.model.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info