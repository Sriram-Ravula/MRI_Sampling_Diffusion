import torch
import torch.fft as torch_fft
import torch.nn as nn

class MulticoilForwardMRINoMask(nn.Module):
    def __init__(self, s_maps):
        """
        Args:
            s_maps: [N, C, H, W] complex
        """
        super(MulticoilForwardMRINoMask, self).__init__()

        self.s_maps = s_maps
    
    def ifft(self, x):
        return self._ifft(x)

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x

    # Centered, orthogonal fft in torch >= 1.7
    def _fft(self, x):
        x = torch_fft.fftshift(x, dim=(-2, -1))
        x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        return x

    def forward(self, image):
        """
        Args:
            image:  [N, 2, H, W] float second channel is (Re, Im)

        Returns:
            ksp_coils: [N, C, H, W] torch.complex64/128 in kspace domain
        """
        #convert to a complex tensor
        x = torch.complex(image[:,0], image[:,1])

        # Broadcast pointwise multiply
        coils = x[:, None] * self.s_maps

        # Convert to k-space data
        ksp_coils = self._fft(coils)

        # Return k-space
        return ksp_coils