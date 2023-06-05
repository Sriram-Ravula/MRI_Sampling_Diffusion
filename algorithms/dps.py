import edm.dnnlib as dnnlib

import pickle
from datasets.mri_dataloaders import get_mvue
from problems.fourier_multicoil import MulticoilForwardMRINoMask
import torch
import numpy as np
from tqdm import tqdm

def normalize(x, x_min, x_max):
    """
    Scales x to appx [-1, 1]
    """
    out = (x - x_min) / (x_max - x_min)
    return 2*out - 1

def unnormalize(x, x_min, x_max):
    """
    Takes input in appx [-1,1] and unscales it
    """
    out = (x + 1) / 2
    return out * (x_max - x_min) + x_min

class Dummy:
    def __init__(self):
        self.s_maps = None

class DPS:
    def __init__(self, hparams, args, c, device=None):
        self.device = device
        self.hparams = hparams
        self.args = args

        self.likelihood_step_size = self.hparams.net.likelihood_step_size
        self.steps = self.hparams.net.steps
        self.rho = 7.0
        self.S_churn = self.hparams.net.S_churn
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1.0

        self._init_net()
        self.c = c.detach().clone() #[N, 1, H, W]
        self.H_funcs = Dummy()
    
    def __call__(self, x_mod, y):
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])
        
        mask = self.c.clone()
        maps = self.H_funcs.s_maps.clone()

        ref = mask * y

        #set up forward operator
        FS = MulticoilForwardMRINoMask(maps)
        A = lambda x: mask * FS(x)

        #grab mvue from undersampled measurements and estimate normalisation
        # factors with it
        estimated_mvue = torch.tensor(
                    get_mvue(ref.cpu().numpy(),
                    maps.cpu().numpy()), device=ref.device)#[N, H, W] complex
        estimated_mvue = torch.view_as_real(estimated_mvue) #[N, H, W, 2] float
        norm_mins = torch.amin(estimated_mvue, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
        norm_maxes = torch.amax(estimated_mvue, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]

        #default labels are zero unless specified
        # [N, label_dim]
        class_labels = None
        if self.net.label_dim:
            class_labels = torch.zeros((ref.shape[0], self.net.label_dim), device=ref.device)

        # Time step discretization.
        step_indices = torch.arange(self.steps, dtype=torch.float64, device=ref.device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = x_mod.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:]))): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            x_hat = x_hat.requires_grad_() #starting grad tracking with the noised img

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Likelihood step
            denoised_unscaled = unnormalize(denoised, norm_mins, norm_maxes) #we need to undo the scaling to [-1,1] first
            Ax = A(denoised_unscaled)
            residual = ref - Ax
            sse_per_samp = torch.sum(torch.square(torch.abs(residual)), dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
            sse = torch.sum(sse_per_samp)
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
            x_next = x_next - (self.likelihood_step_size / torch.sqrt(sse_per_samp)) * likelihood_score

            # Cleanup
            x_next = x_next.detach()
            x_hat = x_hat.detach()
        
        return unnormalize(x_next, norm_mins, norm_maxes)

    def set_c(self, c):
        self.c = c.detach().clone().type_as(self.c).to(self.c.device)
        
    def _init_net(self):
        with dnnlib.util.open_url(self.hparams.net.config_dir, verbose=self.hparams.verbose) as f:
            net = pickle.load(f)['ema'].to(self.device)
        self.net = net
        
        self.sigma_min = max(self.net.sigma_min, 0.002)
        self.sigma_max = min(self.net.sigma_max, 80)

        return
