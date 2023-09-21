import pytest
import torch


@pytest.mark.parametrize('data_modalities', [('clinical', 'mRNA')])
def test_get_data_loaders(data_modalities, dataloaders):
    sample = next(iter(dataloaders['train']))

    assert isinstance(sample, list)
    assert isinstance(sample[0], dict)
    assert isinstance(sample[1], torch.Tensor) and isinstance(sample[2], torch.Tensor)
