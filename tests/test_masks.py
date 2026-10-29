import random

import pytest
import torch

from csa_inpainting.masks import random_mask


@pytest.mark.parametrize(("height", "width"), [(64, 64), (128, 256), (256, 256)])
def test_random_masks_have_bounded_area_and_different_shapes(height, width):
    masks = [random_mask(height, width, random.Random(seed)) for seed in range(32)]
    for mask in masks:
        assert mask.dtype == torch.bool
        assert mask.shape == (1, height, width)
        fraction = mask.float().mean().item()
        assert 0.1 <= fraction <= 0.4
    assert len({mask.numpy().tobytes() for mask in masks}) == len(masks)
    # Exercise irregular silhouettes as well as changes in placement/area.
    non_rectangular = []
    centroids = []
    for mask in masks:
        ys, xs = mask[0].nonzero(as_tuple=True)
        bounding_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        non_rectangular.append(int(mask.sum()) < int(bounding_area))
        centroids.append(torch.stack((ys.float().mean(), xs.float().mean())))
    assert any(non_rectangular)
    spread = torch.stack(centroids).amax(dim=0) - torch.stack(centroids).amin(dim=0)
    assert bool((spread > torch.tensor([height, width]) * 0.2).all())


def test_random_mask_uses_only_supplied_rng():
    first = random_mask(64, 64, random.Random(811), 0.2, 0.3)
    random.seed(999)
    torch.manual_seed(999)
    second = random_mask(64, 64, random.Random(811), 0.2, 0.3)
    assert torch.equal(first, second)
    assert 0.2 <= first.float().mean() <= 0.3


def test_narrow_fraction_range_and_small_non_square_image():
    # Exactly 13 missing pixels are admissible, which is not a rectangular area
    # in a 3x5 image. The bounded fallback must still honor the requested area.
    mask = random_mask(3, 5, random.Random(12), 0.86, 0.87)
    assert mask.sum() == 13


@pytest.mark.parametrize(
    ("height", "width", "minimum", "maximum"),
    [(1, 64, 0.1, 0.4), (64, 1, 0.1, 0.4), (64, 64, 0.5, 0.4), (3, 5, 0.201, 0.202)],
)
def test_invalid_random_mask_request(height, width, minimum, maximum):
    with pytest.raises(ValueError):
        random_mask(height, width, random.Random(0), minimum, maximum)
