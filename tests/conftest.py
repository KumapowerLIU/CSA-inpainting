import pytest
import torch

from csa_inpainting.config import Config


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(2)
    torch.manual_seed(7)


@pytest.fixture
def tiny_config():
    return Config.from_dict(
        {
            "model": {"width": 4, "depth": 6, "attention_chunk_size": 11},
            "data": {"image_size": 64, "batch_size": 2, "workers": 0},
            "train": {
                "device": "cpu",
                "epochs": 4,
                "decay_start": 2,
                "discriminator_width": 4,
                "feature_discriminator_width": 16,
            },
        }
    )


@pytest.fixture
def batch():
    images = torch.rand(2, 3, 64, 64) * 2 - 1
    masks = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    masks[0, :, 16:48, 16:48] = True
    masks[1, :, 8:40, 24:56] = True
    return images, masks
