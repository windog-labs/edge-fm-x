from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from vlaforge.adapters.rdt.rdt_host_pipeline import RDTInputProcessor, pad_recorded_image


@pytest.mark.parametrize('height,width', [(2, 5), (5, 2), (3, 3)])
def test_square_padding_retains_all_original_pixels_and_channel_order(height, width):
    pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    result = np.asarray(pad_recorded_image(Image.fromarray(pixels), (21, 47, 93)))
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    assert np.array_equal(result[top:top + height, left:left + width], pixels)
    mask = np.ones((side, side), dtype=bool)
    mask[top:top + height, left:left + width] = False
    assert (result[mask] == (21, 47, 93)).all()


@pytest.mark.parametrize('dataset', [{'image_aspect_ratio': 'crop'}, {'auto_adjust_image_brightness': True}])
def test_other_image_profiles_cannot_silently_use_recorded_adapter(dataset):
    with pytest.raises(ValueError, match='padding without brightness'):
        RDTInputProcessor(None, None, SimpleNamespace(args={'dataset': dataset}), 1024, device='cpu')


def processor():
    return RDTInputProcessor(None, SimpleNamespace(image_mean=[0.5, 0.5, 0.5]),
        SimpleNamespace(args={'dataset': {'image_aspect_ratio': 'pad'}}), 1024, device='cpu')


@pytest.mark.parametrize('state', [np.zeros(14, dtype=np.float32), np.zeros((1, 14), dtype=np.float64),
    np.full((1, 14), float('nan'), dtype=np.float32)])
def test_recorded_state_is_not_reshaped_cast_or_repaired(state):
    with pytest.raises(ValueError, match='state must remain'):
        processor().prepare({'arrays': {'proprio': state}, 'instruction': 'Recorded task'})


@pytest.mark.parametrize('instruction', ['', '   ', None])
def test_missing_instruction_is_not_replaced(instruction):
    with pytest.raises(ValueError, match='recorded instruction'):
        processor().prepare({'arrays': {'proprio': np.zeros((1, 14), dtype=np.float32)}, 'instruction': instruction})


@pytest.mark.parametrize('image', [np.zeros((2, 3, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8),
    np.zeros((0, 3, 3), dtype=np.uint8)])
def test_camera_arrays_are_not_inferred_or_rescaled(image):
    with pytest.raises(ValueError, match='actual decoded uint8'):
        processor().prepare({'arrays': {'proprio': np.zeros((1, 14), dtype=np.float32), 'image_0': image},
                             'instruction': 'Recorded task'})
