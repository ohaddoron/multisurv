import sys
import os

import pandas as pd
import torch

if torch.cuda.is_available():
    print('>>> PyTorch detected CUDA <<<')
#
# Make modules in "src" dir visible
if os.getcwd() not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

import multisurv.utils as utils
from multisurv.model import Model

DATA = utils.INPUT_DATA_DIR
MODELS = utils.TRAINED_MODEL_DIR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_modalities = ['clinical', 'mRNA', 'miRNA', 'DNAm', 'CNV']

dataloaders = utils.get_dataloaders(data_location=DATA,
                                    labels_file='data/custom_labels.tsv',
                                    modalities=data_modalities,
                                    wsi_patch_size=299,
                                    n_wsi_patches=5,
                                    # batch_size=4
                                    #                                     batch_size=20,
                                    #                                     batch_size=64,
                                    #                                     batch_size=32,
                                    #                                     exclude_patients=exclude_cancers,
                                    )

multisurv = Model(dataloaders=dataloaders,
                  #                   fusion_method='attention',
                  #                   output_intervals=interval_cuts,
                  device=device)

print('Trainable blocks:')
layer = None

for name, child in multisurv.model.named_children():
    for name_2, params in child.named_parameters():
        if name is not layer:
            print(f'   {name}: {params.requires_grad}')
        layer = name

picked_lr = 5e-3

run_tag = utils.compose_run_tag(model=multisurv, lr=picked_lr,
                                dataloaders=dataloaders,
                                log_dir='.training_logs/',
                                suffix='')
fit_args = {
    'lr': picked_lr,
    'num_epochs': 75,
    'info_freq': 1,
    #     'info_freq': None,
    #     'lr_factor': 0.25,
    #     'scheduler_patience': 5,
    'lr_factor': 0.5,
    'scheduler_patience': 10,
    'log_dir': os.path.join('.training_logs/', run_tag),
}

multisurv.fit(**fit_args)
