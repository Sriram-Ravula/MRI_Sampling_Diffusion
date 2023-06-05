import os
import torch
import numpy as np
import glob

from datasets.mri_dataloaders import BrainMultiCoil, KneesMultiCoil

def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)

def get_dataset(config):
    if config.data.dataset == 'Brain-Multicoil':
        train_files = get_all_files(config.data.train_input_dir, pattern='*T2*.h5')
        test_files = get_all_files(config.data.test_input_dir, pattern='*T2*.h5')

        #grab the slice maps and slice counts if they are available
        train_num_slices = getattr(config.data, 'train_num_slices_path', None)
        train_slice_mapper = getattr(config.data, 'train_slice_mapper_path', None)

        test_num_slices = getattr(config.data, 'test_num_slices_path', None)
        test_slice_mapper = getattr(config.data, 'test_slice_mapper_path', None)

        load_slice_info = getattr(config.data, 'load_slice_info', False)
        save_slice_info = getattr(config.data, 'save_slice_info', False)

        #If batch size is one, no point in padding k-space
        train_pad_kspace = 28
        if config.data.train_batch_size == 1 and config.data.val_batch_size == 1:
            train_pad_kspace = False
        
        test_pad_kspace = 28
        if config.data.test_batch_size == 1:
            test_pad_kspace = False

        dataset = BrainMultiCoil(train_files,
                                input_dir=config.data.train_input_dir,
                                maps_dir=config.data.train_maps_dir,
                                image_size = config.data.image_size,
                                num_slices=train_num_slices,
                                slice_mapper=train_slice_mapper,
                                load_slice_info=load_slice_info,
                                save_slice_info=save_slice_info,
                                kspace_pad=train_pad_kspace)

        test_dataset = BrainMultiCoil(test_files,
                                input_dir=config.data.test_input_dir,
                                maps_dir=config.data.test_maps_dir,
                                image_size = config.data.image_size,
                                num_slices=test_num_slices,
                                slice_mapper=test_slice_mapper,
                                load_slice_info=load_slice_info,
                                save_slice_info=save_slice_info,
                                kspace_pad=test_pad_kspace)

    elif config.data.dataset == 'Knee-Multicoil':
        bad_slices = ['file1001022.h5', 'file1000262.h5', 'file1000633.h5', 'file1000794.h5', 'file1000882.h5']

        train_files = get_all_files(config.data.train_input_dir, pattern='*.h5')
        test_files = get_all_files(config.data.test_input_dir, pattern='*.h5')

        train_files = [f for f in train_files if os.path.basename(f) not in bad_slices]

        #grab the slice maps and slice counts if they are available
        train_num_slices = getattr(config.data, 'train_num_slices_path', None)
        train_slice_mapper = getattr(config.data, 'train_slice_mapper_path', None)

        test_num_slices = getattr(config.data, 'test_num_slices_path', None)
        test_slice_mapper = getattr(config.data, 'test_slice_mapper_path', None)

        load_slice_info = getattr(config.data, 'load_slice_info', False)
        save_slice_info = getattr(config.data, 'save_slice_info', False)

        dataset = KneesMultiCoil(train_files,
                                input_dir=config.data.train_input_dir,
                                maps_dir=config.data.train_maps_dir,
                                image_size = config.data.image_size,
                                num_slices=train_num_slices,
                                slice_mapper=train_slice_mapper,
                                load_slice_info=load_slice_info,
                                save_slice_info=save_slice_info)

        test_dataset = KneesMultiCoil(test_files,
                                input_dir=config.data.test_input_dir,
                                maps_dir=config.data.test_maps_dir,
                                image_size = config.data.image_size,
                                num_slices=test_num_slices,
                                slice_mapper=test_slice_mapper,
                                load_slice_info=load_slice_info,
                                save_slice_info=save_slice_info)

    else:
        raise NotImplementedError("Dataset not supported!")

    return dataset, test_dataset

def split_dataset(train_set, test_set, hparams):
    """
    Split a given dataset into train, val, and test sets.
    """
    num_train = hparams.data.num_train
    num_val = hparams.data.num_val
    num_test = hparams.data.num_test

    tr_indices = list(range(len(train_set)))
    te_indices = list(range(len(test_set)))

    print("Train Dataset Size: ", len(train_set))
    print("Test Dataset Size: ", len(test_set))

    random_state = np.random.get_state()
    np.random.seed(hparams.seed)
    np.random.shuffle(tr_indices)
    np.random.seed(hparams.seed)
    np.random.shuffle(te_indices)
    np.random.set_state(random_state)

    train_indices = tr_indices[:num_train]
    val_indices = tr_indices[num_train:num_train+num_val]
    test_indices = te_indices[:num_test]

    train_dataset = torch.utils.data.Subset(train_set, train_indices)
    val_dataset = torch.utils.data.Subset(train_set, val_indices)
    test_dataset = torch.utils.data.Subset(test_set, test_indices)

    out_dict = {'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset}
    

    return out_dict
