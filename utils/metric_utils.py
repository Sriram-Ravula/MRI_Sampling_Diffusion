import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

@torch.no_grad()
def get_ssim(x_hat, x):
    """
    Calculates SSIM(x_hat, x)
    """
    ssim_vals = []
    for i in range(x_hat.shape[0]):
        im1 = x_hat[i,0].detach().cpu().numpy()
        im2 = x[i,0].detach().cpu().numpy()
        range_ = np.amax(im2) - np.amin(im2)
        
        ssim_val = ssim(im1, im2, data_range=range_)
        ssim_vals.append(ssim_val)

    return np.array(ssim_vals)

@torch.no_grad()
def get_psnr(x_hat, x):
    """
    Calculates PSNR(x_hat, x)
    """
    psnr_vals = []
    for i in range(x_hat.shape[0]):
        im1 = x_hat[i,0].detach().cpu().numpy()
        im2 = x[i,0].detach().cpu().numpy()
        range_ = np.amax(im2) - np.amin(im2)
        
        psnr_val = psnr(im2, im1, data_range=range_)
        psnr_vals.append(psnr_val)

    return np.array(psnr_vals)

@torch.no_grad()
def get_all_metrics(x_hat, x):
    """
    function for getting all image reference metrics and returning in a dict
    """

    metrics = {}

    metrics['ssim'] = get_ssim(x_hat, x)
    metrics['psnr'] = get_psnr(x_hat, x)

    return metrics

class Metrics:
    """
    A class for storing and aggregating metrics during a run.
    Metrics are stored as numpy arrays.
    """
    def __init__(self, hparams):
        #dicts for olding raw, image-by-image stats for each iteration.
        #e.g. self.train_metrics['iter_0']['psnr'] = [0.9, 0.1, 0.3] means that at train iteration 0, the images had psnrs of 0.9, 0.1, 0.3
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        #dicts for holding summary stats for each iteration.
        #e.g. self.train_metrics_aggregate['iter_0']['mean_psnr'] = 0.5 means that training iter 0 had a mean train psnr of 0.5
        self.train_metrics_aggregate = {}
        self.val_metrics_aggregate = {}
        self.test_metrics_aggregate = {}

        self.hparams = hparams
    
    def resume(self, checkpoint):
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        self.test_metrics = checkpoint['test_metrics']
        self.train_metrics_aggregate = checkpoint['train_metrics_aggregate']
        self.val_metrics_aggregate = checkpoint['val_metrics_aggregate']
        self.test_metrics_aggregate = checkpoint['test_metrics_aggregate']

        return

    def __init_iter_dict(self, cur_dict, iter_num, should_exist=False):
        """
        Helper method for initializing an iteratio metric dict if it doesn't exist
        """
        iterkey = 'iter_' + str(iter_num)

        if should_exist:
            assert iterkey in cur_dict
        elif iterkey not in cur_dict:
            cur_dict[iterkey] = {}

        return

    def __append_to_iter_dict(self, cur_dict, iter_metrics, iter_num):
        """
        Helper method for appending values to a given iteration metric dict
        """
        iterkey = 'iter_' + str(iter_num)
        for key, value in iter_metrics.items():
            if key not in cur_dict[iterkey]:
                cur_dict[iterkey][key] = value
            else:
                cur_dict[iterkey][key] = np.append(cur_dict[iterkey][key], value)

        return

    def __retrieve_dict(self, iter_type, dict_type='raw'):
        """
        Helper method for validating and retrieving the correct dictionary (train, val, or test)
        """
        assert iter_type in ['train', 'val', 'test']
        assert dict_type in ['raw', 'aggregate']

        if dict_type == 'raw':
            if iter_type == 'train':
                cur_dict = self.train_metrics
            elif iter_type == 'val':
                cur_dict = self.val_metrics
            elif iter_type == 'test':
                cur_dict = self.test_metrics

        elif dict_type == 'aggregate':
            if iter_type == 'train':
                cur_dict = self.train_metrics_aggregate
            elif iter_type == 'val':
                cur_dict = self.val_metrics_aggregate
            elif iter_type == 'test':
                cur_dict = self.test_metrics_aggregate

        return cur_dict

    def get_dict(self, iter_type, dict_type='raw'):
        """Public-facing getter than calls retrieve_dict"""
        return self.__retrieve_dict(iter_type, dict_type)

    def get_metric(self, iter_num, iter_type, metric_key):
        """
        Getter method for retrieving a mean aggregated metric for a certain iteration.
        """
        cur_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')

        metric_key = "mean_" + metric_key
        iterkey = 'iter_' + str(iter_num)
        if iterkey not in cur_dict or metric_key not in cur_dict[iterkey]:
            out_metric = None
        else:
            out_metric = cur_dict[iterkey][metric_key]

        return out_metric

    def get_all_metrics(self, iter_num, iter_type):
        """
        Getter method for retrieiving all the aggregated metrics for a certain iteration.
        """
        cur_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')

        iterkey = 'iter_' + str(iter_num)
        if iterkey not in cur_dict:
            out_dict = None
        else:
            out_dict = cur_dict[iterkey]

        return out_dict

    def calc_iter_metrics(self, x_hat, x, iter_num, iter_type='train'):
        """
        Function for calculating and adding metrics from one iteration to the master.

        Args:
            x_hat: the proposed image(s). torch tensor with shape [N, C, H, W]
            x: ground truth image(s). torch tensor with shape [N, C, H, W]
            iter_num: the global iteration number. int
            iter_type: 'train', 'test', or 'val' - the type of metrics we are calculating
        """
        cur_dict = self.__retrieve_dict(iter_type) #validate and retrieve the right dict

        if x_hat.shape[1] == 2:
            x_hat_ = torch.norm(x_hat, dim=-3).unsqueeze(-3)
            x_ = torch.norm(x, dim=-3).unsqueeze(-3)
        else:
            x_hat_ = x_hat.clone()
            x_ = x.clone()
        iter_metrics = get_all_metrics(x_hat_, x_) #calc the metrics

        self.__init_iter_dict(cur_dict, iter_num) #check that the iter dict is initialized

        self.__append_to_iter_dict(cur_dict, iter_metrics, iter_num) #add the values to the iter dict

        return

    def add_external_metrics(self, external_metrics, iter_num, iter_type='train'):
        """
        Function for adding a given dict of metrics to the given iteration.
        """
        cur_dict = self.__retrieve_dict(iter_type) #validate and retrieve the right dict

        self.__init_iter_dict(cur_dict, iter_num) #check that the iter dict is initialized

        self.__append_to_iter_dict(cur_dict, external_metrics, iter_num) #add the values to the iter dict

        return

    def aggregate_iter_metrics(self, iter_num, iter_type='train'):
        """
        Called at the end of an iteration/epoch to find summary stats for all the metrics.
        """
        agg_dict = self.__retrieve_dict(iter_type, dict_type='aggregate') #validate and retrieve the right dicts
        raw_dict = self.__retrieve_dict(iter_type, dict_type='raw')

        self.__init_iter_dict(agg_dict, iter_num) #check that the iter dict is initialized
        self.__init_iter_dict(raw_dict, iter_num, should_exist=True) #make sure the corresponding dict exists in the raw

        #go through all the metrics in the raw data and aggregate them
        iterkey = 'iter_' + str(iter_num)
        for key, value in raw_dict[iterkey].items():
            mean_key = "mean_" + key
            std_key = "std_" + key
            mean_value = np.mean(value)
            std_value = np.std(value)
            agg_dict[iterkey][mean_key] = mean_value
            agg_dict[iterkey][std_key] = std_value

        return

    def add_metrics_to_tb(self, tb_logger, step, iter_type='train'):
        """
        Run through metrics and log everything there.
        For each type of metric, we want to log the train, val, and test metrics on the same plot.
        Intended to be called at the end of each type of iteration (train, test, val)
        """
        assert iter_type in ['train', 'val', 'test']

        raw_dict = self.__retrieve_dict(iter_type, dict_type='raw')
        agg_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')

        iterkey ='iter_' + str(step)

        if iterkey not in raw_dict:
            print("\ncurrent iteration has not yet been logged\n")
            return

        for metric_type, metric_value in raw_dict[iterkey].items():
            for i, val in enumerate(metric_value):
                tb_logger.add_scalars("raw " + metric_type, {iter_type: val}, step*len(metric_value) + i)

        for metric_type, metric_value in agg_dict[iterkey].items():
            tb_logger.add_scalars(metric_type, {iter_type: metric_value}, step)

        return
