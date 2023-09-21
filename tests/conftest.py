from pathlib import Path

import pytest
import torch

from multisurv import utils
from multisurv.model import Model

DATA = utils.INPUT_DATA_DIR
data_modalities = ('clinical', 'mRNA')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='session')
def dataloaders():
    yield utils.get_dataloaders(data_location=DATA,
                                labels_file=Path(__file__).parent.parent.joinpath('data/labels.tsv').as_posix(),
                                modalities=data_modalities,
                                wsi_patch_size=299,
                                n_wsi_patches=5,
                                )


@pytest.fixture(scope='session')
def multisurv(dataloaders):
    yield Model(dataloaders=dataloaders,
                device=device)
