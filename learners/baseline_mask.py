import torch
import numpy as np
import sigpy.mri

class Baseline_Mask:
    def __init__(self, hparams, device, num_acs_lines=20):
        super().__init__()

        #Expects all info to be in hparams.mask
        #hparams.mask.sample_pattern in ["horizontal", "vertical", "3D"]
        #hparams.mask.R - NOTE this will be the true R with center sampled by default!
        self.hparams = hparams
        self.device = device

        self.num_acs_lines = num_acs_lines #number of lines to keep for 1D, side length ratio of central square for 3D

        self._init_mask()

    def _init_mask(self):
        #define the number of parameters
        if self.hparams.mask.sample_pattern in ['horizontal', 'vertical']:
            m = self.hparams.data.image_size
        elif self.hparams.mask.sample_pattern == '3D':
            m = self.hparams.data.image_size ** 2
        else:
            raise NotImplementedError("Fourier sampling pattern not supported!")

        #now check for any smart initializations
        center_line_idx = np.arange((self.hparams.data.image_size - self.num_acs_lines) // 2,
                            (self.hparams.data.image_size + self.num_acs_lines) // 2)

        if self.hparams.mask.sample_pattern == '3D':
            c = sigpy.mri.poisson(img_shape=(self.hparams.data.image_size, self.hparams.data.image_size),
                                    accel=self.hparams.mask.R,
                                    seed=self.hparams.seed-1) #NOTE function errors with seg fault for seed 2023. Using -1 to fix
            c = torch.tensor(c)
            c = torch.view_as_real(c)[:,:,0]
            center_line_idx = np.meshgrid(center_line_idx, center_line_idx)
            c[center_line_idx] = 1.
            c = c.flatten()

        elif self.hparams.mask.sample_pattern in ['horizontal', 'vertical']:
            outer_line_idx = np.setdiff1d(np.arange(self.hparams.data.image_size), center_line_idx)

            #account for the center lines when sampling the rest of the equispaced to match proper R
            outer_R = np.round((m - self.num_acs_lines) / (m/self.hparams.mask.R - self.num_acs_lines))
            random_line_idx = outer_line_idx[::int(outer_R)]
            c = torch.zeros(self.hparams.data.image_size)
            c[center_line_idx] = 1.
            c[random_line_idx] = 1.

        self.weights = c.to(self.device).type(torch.float32)
    
    def _reshape_mask(self, raw_mask):
        if self.hparams.mask.sample_pattern == '3D':
            out_mask = raw_mask.view(self.hparams.data.image_size, self.hparams.data.image_size)

        elif self.hparams.mask.sample_pattern == 'horizontal':
            out_mask = raw_mask.unsqueeze(1).repeat(1, self.hparams.data.image_size)

        elif self.hparams.mask.sample_pattern == 'vertical':
            out_mask = raw_mask.unsqueeze(0).repeat(self.hparams.data.image_size, 1)

        return out_mask

    def sample_mask(self):
        return self._reshape_mask(self.weights)
        