import torch
import torchvision
import random
import numpy as np
import argparse
import yaml
from datetime import datetime
import os
import pickle

def save_image(image, path):
    """
    Save a pytorch image as a png file.
    """
    x_png = image.detach().cpu()

    if x_png.shape[0] == 2:
        x_png = torch.linalg.norm(x_png, dim=0, keepdim=True)
    
    torchvision.utils.save_image(x_png, path)

def save_images(images, labels, save_prefix):
    """
    Save a batch of images (in a dictionary) to png files and .pt files.
    """
    for image_num, image in zip(labels, images):
        if isinstance(image_num, torch.Tensor):
            save_image(image, os.path.join(save_prefix, str(image_num.item())+'.png'))
            torch.save(image, os.path.join(save_prefix, str(image_num.item())+'.pt'))
        elif isinstance(image_num, int):
            save_image(image, os.path.join(save_prefix, str(image_num)+'.png'))
            torch.save(image, os.path.join(save_prefix, str(image_num)+'.pt'))
        elif isinstance(image_num, str):
            save_image(image, os.path.join(save_prefix, image_num +'.png'))
            torch.save(image, os.path.join(save_prefix, image_num +'.pt'))
        else:
            raise NotImplementedError("Bad type given to save_images for labels.")

def save_to_pickle(data, pkl_filepath):
    """
    Save the data to a pickle file
    """
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_if_pickled(pkl_filepath):
    """
    Load if the pickle file exists and return. 
    """
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data

def set_all_seeds(random_seed: int):
    """
    Sets random seeds in numpy, torch, and random.

    Args:
        random_seed: The seed to set.
                     Type: int.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    return

def dict2namespace(config: dict):
    """
    Converts a given dictionary to an argparse namespace object.

    Args:
        config: The dictionary to convert to namespace.
                Type: dict.

    Returns:
        namespace: The converted namespace.
                   Type: Namespace.
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_config(config_path):
    """
    Parses experiment configuration yaml file.
    Sets up the device.
       
    Returns a namespace with config parameters.
    """
    with open(config_path, 'r') as f:
        hparams = yaml.safe_load(f)

    #set up the devices - account for multi_GPU DP mode
    if hparams['use_gpu']:
        num = hparams['gpu_num']
        if num == -1:
            hparams['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if torch.cuda.device_count() > 1 and hparams['verbose']:
                print("Let's use ", torch.cuda.device_count(), " GPUs!")
        else:
            hparams['device'] = torch.device('cuda:'+str(num)) if torch.cuda.is_available() else torch.device('cpu:0')
    else:
        hparams['gpu_num'] = None
        hparams['device'] = torch.device('cpu:0')

    #create namespace, print, and wrap up
    HPARAMS = dict2namespace(hparams)
    print(yaml.dump(HPARAMS, default_flow_style=False))

    return HPARAMS

def parse_args(docstring=""):
    """
    Gets command line arguments
    """
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--doc', type=str, default=now, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')

    parser.add_argument('--test', action='store_true', help='set to run a test')

    parser.add_argument('--resume', action='store_true', help="whether to resume from the last checkpoint")
    
    parser.add_argument('--mask_path', type=str, default=None, required=False,  help='Path to probabilistic mask to restore')

    parser.add_argument('--baseline', action='store_true', help='whether to run a baseline with a fixed mask')

    args = parser.parse_args()

    return args