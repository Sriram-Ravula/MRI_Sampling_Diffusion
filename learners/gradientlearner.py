import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.fft as torch_fft
import json

import os
import sys
import torch.utils.tensorboard as tb
import yaml

from algorithms.dps import DPS
from problems.fourier_multicoil import MulticoilForwardMRINoMask
from datasets import get_dataset, split_dataset

from utils.exp_utils import save_images, save_to_pickle, load_if_pickled
from utils.metric_utils import Metrics

from learners.probabilistic_mask import Probabilistic_Mask
from learners.baseline_mask import Baseline_Mask

class MaskLearner:
    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.device = self.hparams.device

        #check if we have a probabilistic c
        self.prob_c = not(self.args.baseline)

        #running parameters
        self._init_c()
        self._init_dataset()
        self.A = None #placeholder for forward operator - None since each sample has different coil map

        if not(self.args.baseline) and not(self.args.test):
            self._init_meta_optimizer()
        
        if self.args.mask_path is not None:
            self.c.weights.requires_grad_(False)
            
            checkpoint = load_if_pickled(os.path.join(args.mask_path))
            self.c.weights.copy_(checkpoint["c_weights"].to(self.device))
            
            self._print_if_verbose("Restoring Mask from given path")
            self.c.weights.requires_grad_(True)

        self.global_epoch = 0
        
        #track the best validation weights and restore them before test time
        self.best_val_psnr = 0
        self.best_val_weights = None

        if self.prob_c:
            tau = getattr(self.hparams.mask, 'tau', 0.5)
            self.cur_mask_sample = self.c.sample_mask(tau).unsqueeze(0).unsqueeze(0) #draw a binary mask sample and reshape [H, W] --> [1, 1, H, W]
            c_shaped = self.cur_mask_sample.detach().clone()
        else:
            c_shaped = self.c.sample_mask().unsqueeze(0).unsqueeze(0) #[H, W] --> [1, 1, H, W]

        if self.hparams.net.model == 'dps':
            self.recon_alg = DPS(self.hparams, self.args, c_shaped, self.device)
        else:
            raise NotImplementedError("Reconstruction algorithm not supported!")

        #logging and metrics
        self.metrics = Metrics(hparams=self.hparams)
        self.log_dir = os.path.join(self.hparams.save_dir, self.args.doc)
        self.image_root = os.path.join(self.log_dir, 'images')
        self.tb_root = os.path.join(self.log_dir, 'tensorboard')

        self._make_log_folder()
        self._save_config()

        self.tb_logger = tb.SummaryWriter(log_dir=self.tb_root)

        #We need all the stuff made before we can resume
        if self.args.resume:
            self._resume()
            return

        #take a snap of the initialization
        if not self.hparams.debug and self.hparams.save_imgs:
            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone().squeeze()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_binary.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Prob_00", "Sample_00", "Max_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (c_shaped_binary.count_nonzero() / c_shaped_binary.numel())
                self._print_if_verbose("INITIAL SPARSITY (SAMPLE MASK): " + str(sparsity_level.item()))

                sparsity_level = 1 - (c_shaped_max.count_nonzero() / c_shaped_max.numel())
                self._print_if_verbose("INITIAL SPARSITY (MAX MASK): " + str(sparsity_level.item()))
            else:
                c_shaped = self.c.sample_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = c_shaped.unsqueeze(0).cpu()

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Actual_00"], c_path)

                #NOTE sparsity level is the proportion of zeros in the image
                sparsity_level = 1 - (c_shaped.count_nonzero() / c_shaped.numel())
                self._print_if_verbose("INITIAL SPARSITY: " + str(sparsity_level.item()))

    def test(self):
        """
        Run through the test set.
        We want to save the metrics for each individual sample here!
        We also want to save images of every sample, reconstruction, measurement, recon_meas,
            and the c.
        """
        self._print_if_verbose("TESTING")

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            #grab a new mask(s) for every sample
            if self.prob_c:
                bs = item['gt_image'].shape[0]
                tau = getattr(self.hparams.mask, 'tau', 0.5)
                self.cur_mask_sample = torch.stack([self.c.sample_mask(tau).unsqueeze(0) for _ in range(bs)], dim=0) #[N, 1, H, W]
                c_shaped = self.cur_mask_sample.detach().clone()
            else:
                c_shaped = self.c.sample_mask().unsqueeze(0).unsqueeze(0)
            self.recon_alg.set_c(c_shaped)
            
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #logging and saving
            scan_idxs = item['scan_idx']
            slice_idxs = item['slice_idx']
            x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
            self._save_all_images(x_hat, x, y, x_idx, "test", save_masks_manual=(True if i==0 else False))

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._add_metrics_to_tb("test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

        #grab the raw metrics dictionary and save it
        test_metrics = self.metrics.test_metrics['iter_'+str(self.global_epoch)]
        save_to_pickle(test_metrics, os.path.join(self.log_dir, "test_"+str(self.global_epoch)+".pkl"))

        return

    def run_meta_opt(self):
        for iter in tqdm(range(self.hparams.opt.num_iters)):
            #checkpoint
            if iter % self.hparams.opt.checkpoint_iters == 0:
                self._checkpoint()

            #train
            self._run_outer_step()
            self._add_metrics_to_tb("train")
            
            #validate
            if (iter + 1) % self.hparams.opt.val_iters == 0:
                self._run_validation()
                self._add_metrics_to_tb("val")

            self.global_epoch += 1

        #test
        self._run_test()
        self._add_metrics_to_tb("test")

        self._checkpoint()
    
    def _dps_loss(self, item, net):
        #(0) Grab the necessary variables
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float, x*
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W] complex, FSx*
        if len(y.shape) > 4:
            y = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex, S
        
        #Fix - make single-coil measurements
        s_maps = torch.ones_like(s_maps)[:, 0].unsqueeze(1)
        self.A = MulticoilForwardMRINoMask(s_maps) #FS, [N, 2, H, W] float --> [N, C, H, W] complex
        y = self.A(x)
        ref = self.cur_mask_sample * y #[N, C, H, W] complex, PFSx*
    
        with torch.no_grad():
            estimated_mvue = torch.sum(self._ifft(ref) * torch.conj(s_maps), axis=1) / torch.sqrt(torch.sum(torch.square(torch.abs(s_maps)), axis=1))
            estimated_mvue = torch.view_as_real(estimated_mvue)
            norm_mins = torch.amin(estimated_mvue, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
            norm_maxes = torch.amax(estimated_mvue, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
        
        #Also grab normalisation factors for ground truth MVUE
        x_mins = torch.amin(x, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
        x_maxes = torch.amax(x, dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
        
        class_labels = None
        if net.label_dim:
            class_labels = torch.zeros((x.shape[0], net.label_dim), device=self.device)#[N, label_dim]
        
        #(3) Grab the noise and noise the image
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp() #P_std=1.2, P_mean=-1.2
        
        #scale the gt mvue to [-1, 1] before noising
        x_scaled = (x - x_mins) / (x_maxes - x_mins)
        x_scaled = 2*x_scaled - 1

        n = torch.randn_like(x) * sigma
        x_t = x_scaled + n

        x_t = x_t.requires_grad_() #track gradients for DPS
        
        #(4) Grab the unconditional denoise estimate and the likelihood grad
        #\hat{x}_0^t
        x_hat_0 = net(x_t, sigma, class_labels)

        x_hat_0_unscaled = (x_hat_0 + 1) / 2
        x_hat_0_unscaled = x_hat_0_unscaled * (norm_maxes - norm_mins) + norm_mins
        
        # Likelihood gradient
        Ax = self.cur_mask_sample * self.A(x_hat_0_unscaled) #PFS\hat{x}_0^t
        residual = ref - Ax
        sse_per_samp = torch.sum(torch.square(torch.abs(residual)), dim=(1,2,3), keepdim=True) #[N, 1, 1, 1]
        sse = torch.sum(sse_per_samp)
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_t, create_graph=True)[0] #create a graph to track grads of likelihood
        
        #Final Denoised prediction
        x_hat = x_hat_0 - self.hparams.net.training_step_size * likelihood_score
        
        x_hat = (x_hat + 1) / 2
        x_hat = x_hat * (norm_maxes - norm_mins) + norm_mins
        
        #(5) Update Step
        self.opt.zero_grad()
        
        if self.hparams.mask.meta_loss_type == "l2":
            meta_loss = torch.sum(torch.square(x_hat - x))
        else:
            raise NotImplementedError("META LOSS NOT IMPLEMENTED!")
        meta_loss.backward()
        
        self.opt.step()
        
        #(6) Log Things
        with torch.no_grad():
            grad_metrics_dict = {"sigma": sigma.flatten().cpu().numpy(),
                                 "meas_sse": sse_per_samp.flatten().cpu().numpy(),
                                 "meta_loss": np.array([meta_loss.item()] * x.shape[0]),
                                 "likelihood_grad_norm": torch.norm(likelihood_score, p=2, dim=(1,2,3)).detach().cpu().numpy()}
            self.metrics.add_external_metrics(grad_metrics_dict, self.global_epoch, "train")
        
        return x_hat, x, y

    def _run_outer_step(self):
        self._print_if_verbose("\nTRAINING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.train_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            tau = getattr(self.hparams.mask, 'tau', 0.5)
            self.cur_mask_sample = torch.stack([self.c.sample_mask(tau).unsqueeze(0) for _ in range(bs)], dim=0) #[N, 1, H, W]
            c_shaped = self.cur_mask_sample.detach().clone()
            self.recon_alg.set_c(c_shaped)
            
            x_hat, x, y = self._dps_loss(item, self.recon_alg.net)
            self._add_batch_metrics(x_hat, x, y, "train")

            #logging and saving
            if i == 0:
                scan_idxs = item['scan_idx']
                slice_idxs = item['slice_idx']
                x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
                self._save_all_images(x_hat, x, y, x_idx, "train")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "train")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "train"), "\n")

    def _run_validation(self):
        self._print_if_verbose("\nVALIDATING\n")

        for i, (item, x_idx) in tqdm(enumerate(self.val_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            tau = getattr(self.hparams.mask, 'tau', 0.5)
            self.cur_mask_sample = torch.stack([self.c.sample_mask(tau).unsqueeze(0) for _ in range(bs)], dim=0) #[N, 1, H, W]
            c_shaped = self.cur_mask_sample.detach().clone()
            self.recon_alg.set_c(c_shaped)
            
            #Grab the recon
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "val")

            #logging and saving
            if i == 0:
                scan_idxs = item['scan_idx']
                slice_idxs = item['slice_idx']
                x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
                self._save_all_images(x_hat, x, y, x_idx, "val")

        self.metrics.aggregate_iter_metrics(self.global_epoch, "val")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "val"), "\n")
        
        #track the best validation stats
        cur_val_psnr = self.metrics.get_all_metrics(self.global_epoch, "val")['mean_psnr']
        if cur_val_psnr > self.best_val_psnr:
            self.best_val_psnr = cur_val_psnr
            self.best_val_weights = self.c.weights.clone().detach()
            self._print_if_verbose("BEST VALIDATION PSNR: ", cur_val_psnr)

    def _run_test(self):
        self._print_if_verbose("\nTESTING\n")
        
        #Restore best weights
        self.c.weights.requires_grad_(False)
        self.c.weights.copy_(self.best_val_weights)
        self._print_if_verbose("Restoring best validation weights")

        for i, (item, x_idx) in tqdm(enumerate(self.test_loader)):
            #grab a new mask(s) for every sample
            bs = item['gt_image'].shape[0]
            tau = getattr(self.hparams.mask, 'tau', 0.5)
            self.cur_mask_sample = torch.stack([self.c.sample_mask(tau).unsqueeze(0) for _ in range(bs)], dim=0) #[N, 1, H, W]
            c_shaped = self.cur_mask_sample.detach().clone()
            self.recon_alg.set_c(c_shaped)
            
            #Grab the recon            
            x_hat, x, y = self._shared_step(item)
            self._add_batch_metrics(x_hat, x, y, "test")

            #logging and saving
            scan_idxs = item['scan_idx']
            slice_idxs = item['slice_idx']
            x_idx = [str(scan_id.item())+"_"+str(slice_id.item()) for scan_id, slice_id in zip(scan_idxs, slice_idxs)]
            self._save_all_images(x_hat, x, y, x_idx, "test", save_masks_manual=(True if i==0 else False))

        self.metrics.aggregate_iter_metrics(self.global_epoch, "test")
        self._print_if_verbose("\n", self.metrics.get_all_metrics(self.global_epoch, "test"), "\n")

    def _shared_step(self, item):
        x = item['gt_image'].to(self.device) #[N, 2, H, W] float second channel is (Re, Im)
        y = item['ksp'].type(torch.cfloat).to(self.device) #[N, C, H, W, 2] float last channel is ""
        s_maps = item['s_maps'].to(self.device) #[N, C, H, W] complex

        #set coil maps and forward operator including current coil maps
        self.recon_alg.H_funcs.s_maps = s_maps
        self.A = MulticoilForwardMRINoMask(s_maps)

        #Get the reconstruction
        x_mod = torch.randn_like(x)
        x_hat = self.recon_alg(x_mod, y) #[N, 2, H, W] float

        return x_hat, x, y

    # Centered, orthogonal ifft in torch >= 1.7
    def _ifft(self, x):
        x = torch_fft.ifftshift(x, dim=(-2, -1))
        x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch_fft.fftshift(x, dim=(-2, -1))
        return x 

    @torch.no_grad()
    def _add_batch_metrics(self, x_hat, x, y, iter_type):
        #calc the measurement loss in fully-sampled kspace
        resid = self.A(x_hat) - y
        real_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise SSE

        #calc the measurement loss in the observed indices                                           
        if self.prob_c:
            c_shaped = self.cur_mask_sample.detach().clone() #[N, 1, H, W]
        else:
            c_shaped = self.c.sample_mask().unsqueeze(0).unsqueeze(0) #[1, 1, H, W]
        resid = c_shaped * resid
        weighted_meas_loss = torch.sum(torch.square(torch.abs(resid)), dim=[1,2,3]) #get element-wise SSE with mask

        #calc the ground truth L2 and L1 error
        resid = x_hat - x
        gt_mse = torch.mean(torch.square(resid), dim=[1,2,3]) #element-wise MSE in pixel-space
        gt_mae = torch.mean(torch.abs(resid), dim=[1,2,3]) #element-wise mean MAE 

        extra_metrics_dict = {"real_meas_sse": real_meas_loss.cpu().numpy().flatten(),
                            "weighted_meas_sse": weighted_meas_loss.cpu().numpy().flatten(),
                            "gt_mse": gt_mse.cpu().numpy().flatten(),
                            "gt_mae": gt_mae.cpu().numpy().flatten()}
        
        if self.prob_c:
            prob_mask = self.c.get_prob_mask()
            extra_metrics_dict["mean_prob"] = np.array([torch.mean(prob_mask).item()] * x.shape[0])

            max_mask = self.c.get_max_mask()
            sparsity_level_max = 1 - (max_mask.count_nonzero() / max_mask.numel())
            extra_metrics_dict["sparsity_level_max"] = np.array([sparsity_level_max.item()] * x.shape[0]) #ugly artifact

            cur_mask = self.cur_mask_sample.detach().clone() #[N, 1, H, W]
            sparsity_level_sample = 1 - (cur_mask.count_nonzero(dim=(1,2,3)) / (cur_mask.shape[-1]*cur_mask.shape[-2])) #[N]
            extra_metrics_dict["sparsity_level_sample"] = sparsity_level_sample.cpu().numpy()
        else:
            sparsity_level = 1 - (c_shaped.count_nonzero() / c_shaped.numel())
            extra_metrics_dict["sparsity_level"] = np.array([sparsity_level.item()] * x.shape[0]) 

        self.metrics.add_external_metrics(extra_metrics_dict, self.global_epoch, iter_type)
        self.metrics.calc_iter_metrics(x_hat, x, self.global_epoch, iter_type)

    @torch.no_grad()
    def _save_all_images(self, x_hat, x, y, x_idx, iter_type, save_masks_manual=False):
        if self.hparams.debug or (not self.hparams.save_imgs):
            return
        elif iter_type == "train" and not (self.global_epoch % self.hparams.opt.checkpoint_iters == 0 or
                 self.global_epoch == self.hparams.opt.num_iters - 1):
            return

        #(1) Save samping masks
        if iter_type == "train" or save_masks_manual:
            if self.prob_c:
                c_shaped = self.c.get_prob_mask()
                c_shaped_binary = self.cur_mask_sample.detach().clone()
                c_shaped_max = self.c.get_max_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = torch.stack([c_shaped.unsqueeze(0).cpu(), c_shaped_max.unsqueeze(0).cpu()])

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Prob_" + str(self.global_epoch), 
                                          "Max_" + str(self.global_epoch)], c_path)
                self._save_images(c_shaped_binary.cpu(), 
                                  ["Sample_"+str(self.global_epoch)+"_"+str(j) for j in range(c_shaped_binary.shape[0])],
                                  c_path)
            else:
                c_shaped = self.c.sample_mask()

                c_path = os.path.join(self.image_root, "learned_masks")
                c_out = c_shaped.unsqueeze(0).cpu()

                if not os.path.exists(c_path):
                    os.makedirs(c_path)
                self._save_images(c_out, ["Actual_" + str(self.global_epoch)], c_path)

        #(2) Save reconstructions at every iteration
        meas_recovered_path = os.path.join(self.image_root, iter_type + "_recon_meas", "epoch_"+str(self.global_epoch))
        recovered_path = os.path.join(self.image_root, iter_type + "_recon", "epoch_"+str(self.global_epoch))

        x_hat_vis = torch.norm(x_hat, dim=1).unsqueeze(1) #[N, 1, H, W]
        x_resid = torch.norm(x_hat - x, dim=1).unsqueeze(1) #save the residual image
        x_resid_stretched = (x_resid - torch.amin(x_resid, dim=(1,2,3), keepdim=True)) / \
                                (torch.amax(x_resid, dim=(1,2,3), keepdim=True) - torch.amin(x_resid, dim=(1,2,3), keepdim=True))

        if not os.path.exists(recovered_path):
            os.makedirs(recovered_path)
        self._save_images(x_hat_vis, x_idx, recovered_path)
        self._save_images(x_resid, [idx + "_resid" for idx in x_idx], recovered_path)
        self._save_images(x_resid_stretched, [idx + "_resid_stretched" for idx in x_idx], recovered_path)
        
        #grab the dict and save the stats for the recons
        metric_dict = self.metrics.get_dict(iter_type)['iter_' + str(self.global_epoch)]
        psnr_array = metric_dict['psnr'][-len(x_idx):]
        ssim_array = metric_dict['ssim'][-len(x_idx):]
        sample_metric_dicts = [{"Slice": idx, "PSNR": psnr_array[i], "SSIM": ssim_array[i]} for i, idx in enumerate(x_idx)]
        metric_path = os.path.join(recovered_path, "sample_metrics.json")
        with open(metric_path, 'a') as f:
            json.dump(sample_metric_dicts, f, indent=4)
            
        avg_metric_dict = [{"MEAN PSNR": np.mean(metric_dict['psnr']), "MEAN SSIM": np.mean(metric_dict['ssim']),
                            "STD PSNR": np.std(metric_dict['psnr']), "STD SSIM": np.std(metric_dict['ssim'])}]
        avg_metric_path = os.path.join(recovered_path, "avg_sample_metrics.json")
        with open(avg_metric_path, 'w') as f:
            json.dump(avg_metric_dict, f, indent=4)

        fake_maps = torch.ones_like(x)[:,0,:,:].unsqueeze(1) #[N, 1, H, W]
        recon_meas = MulticoilForwardMRINoMask(fake_maps)(x_hat)
        recon_meas = torch.abs(recon_meas)

        if not os.path.exists(meas_recovered_path):
            os.makedirs(meas_recovered_path)
        self._save_images(recon_meas, x_idx, meas_recovered_path)

        #(3) Save ground truth only once
        if "test" in iter_type or self.global_epoch == 0:
            true_path = os.path.join(self.image_root, iter_type)
            meas_path = os.path.join(self.image_root, iter_type + "_meas")

            x_vis = torch.norm(x, dim=1).unsqueeze(1) #[N, 1, H, W]

            if not os.path.exists(true_path):
                os.makedirs(true_path)
            self._save_images(x_vis, x_idx, true_path)

            gt_meas = MulticoilForwardMRINoMask(fake_maps)(x)
            gt_meas = torch.abs(gt_meas)

            if not os.path.exists(meas_path):
                os.makedirs(meas_path)
            self._save_images(gt_meas, x_idx, meas_path)

    def _init_dataset(self):
        train_set, test_set = get_dataset(self.hparams)
        split_dict = split_dataset(train_set, test_set, self.hparams)
        train_dataset = split_dict['train']
        val_dataset = split_dict['val']
        test_dataset = split_dict['test']

        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.data.train_batch_size, shuffle=True,
                                num_workers=1, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.data.val_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.hparams.data.test_batch_size, shuffle=False,
                                num_workers=1, drop_last=True)

    def _init_c(self):
        num_acs_lines = getattr(self.hparams.mask, 'num_acs_lines', 20)

        if self.prob_c:
            self.c = Probabilistic_Mask(self.hparams, self.device, num_acs_lines)
        else:
            self.c = Baseline_Mask(self.hparams, self.device, num_acs_lines)

        return

    def _init_meta_optimizer(self):
        opt_type = self.hparams.opt.optimizer
        lr = self.hparams.opt.lr

        if opt_type == 'adam':
            meta_opt = torch.optim.Adam([{'params': self.c.weights}], lr=lr)
        elif opt_type == 'sgd':
            meta_opt = torch.optim.SGD([{'params': self.c.weights}], lr=lr)
        else:
            raise NotImplementedError("Optimizer not supported!")

        if self.hparams.opt.decay:
            meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_opt, self.hparams.opt.lr_decay)
        else:
            meta_scheduler = None

        self.opt =  meta_opt
        self.scheduler = meta_scheduler

    def _checkpoint(self):
        if self.hparams.debug:
            return

        save_dict = {
            "c_weights": self.c.weights.detach().cpu(),
            "global_epoch": self.global_epoch,
            "opt_state": self.opt.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None
        }
        metrics_dict = {
            'train_metrics': self.metrics.train_metrics,
            'val_metrics': self.metrics.val_metrics,
            'test_metrics': self.metrics.test_metrics,
            'train_metrics_aggregate': self.metrics.train_metrics_aggregate,
            'val_metrics_aggregate': self.metrics.val_metrics_aggregate,
            'test_metrics_aggregate': self.metrics.test_metrics_aggregate,
        }
        save_to_pickle(save_dict, os.path.join(self.log_dir, "checkpoint.pkl"))
        save_to_pickle(metrics_dict, os.path.join(self.log_dir, "metrics.pkl"))

    def _resume(self):
        self._print_if_verbose("RESUMING FROM CHECKPOINT")

        checkpoint = load_if_pickled(os.path.join(self.log_dir, "checkpoint.pkl"))
        metrics = load_if_pickled(os.path.join(self.log_dir, "metrics.pkl"))

        self.c.weights.copy_(checkpoint["c_weights"].to(self.device))
        self.c.weights.requires_grad_()

        if not(self.args.baseline) and not(self.args.test):
            self.opt.load_state_dict(checkpoint['opt_state'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.global_epoch = checkpoint['global_epoch']

        self.metrics.resume(metrics)

        self._print_if_verbose("RESUMING FROM EPOCH " + str(self.global_epoch))

    def _make_log_folder(self):
        if not self.hparams.debug:
            if os.path.exists(self.log_dir):
                sys.exit("Folder exists. Program halted.")
            else:
                os.makedirs(self.log_dir)
                os.makedirs(self.image_root)
                os.makedirs(self.tb_root)

    def _save_config(self):
        if not self.hparams.debug:
            with open(os.path.join(self.log_dir, 'config.yml'), 'w') as f:
                yaml.dump(self.hparams, f, default_flow_style=False)
            
            with open(os.path.join(self.log_dir, 'args.yml'), 'w') as f:
                yaml.dump(self.args, f, default_flow_style=False)

    def _add_metrics_to_tb(self, iter_type):
        if not self.hparams.debug:
            self.metrics.add_metrics_to_tb(self.tb_logger, self.global_epoch, iter_type)

    def _save_images(self, images, img_indices, save_path):
        if not self.hparams.debug and self.hparams.save_imgs:
            save_images(images, img_indices, save_path)

    def _print_if_verbose(self, *text):
        if self.hparams.verbose:
            print("".join(str(t) for t in text))
