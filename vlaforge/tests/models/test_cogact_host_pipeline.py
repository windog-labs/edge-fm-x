from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from vlaforge.adapters.cogact.cogact_host_pipeline import count_official_draws, load_recorded_samples
from vlaforge.validation.host_pipeline import digest


def write(path, data):
    path.write_text(json.dumps(data))


def recorded(tmp_path):
    observations, series = tmp_path / 'observations', tmp_path / 'reference'
    image_root, sample_root = observations / 'frame-3', series / 'sample-0'
    image_root.mkdir(parents=True)
    sample_root.mkdir(parents=True)
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    Image.fromarray(rgb).save(image_root / 'observation.png')
    record = {'schema': 'vlaforge.cogact.public-observation/1', 'simulation': False,
        'frame': 3, 'instruction': 'Recorded instruction', 'source_lock_sha256': 'a' * 64,
        'image_sha256': digest(image_root / 'observation.png'), 'image_shape': list(rgb.shape),
        'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'unnorm_key': 'recorded-profile'}
    write(image_root / 'input.json', record)
    manifest = {'schema': 'vlaforge.cogact.real-observation-series/1', 'count': 1,
        'source_lock_sha256': 'a' * 64, 'frames': [{**record, 'directory': 'frame-3'}]}
    write(observations / 'manifest.json', manifest)
    np.savez(sample_root / 'inputs.npz', tokens=np.array([[1, 2]], dtype=np.int64),
        dino=np.zeros((1, 3, 2, 2), dtype=np.float32), siglip=np.ones((1, 3, 2, 2), dtype=np.float32),
        initial_noise=np.zeros((1, 16, 7), dtype=np.float32), step_noise=np.zeros((10, 2, 16, 7), dtype=np.float32),
        rng_states=np.zeros((12, 16), dtype=np.uint8), rng_before=np.zeros(16, dtype=np.uint8))
    reference = {'status': 'real_series_partition_verified',
        'series_manifest_sha256': digest(observations / 'manifest.json'),
        'runs': [{'index': 0, 'frame': 3, 'input_sha256': digest(sample_root / 'inputs.npz')}]}
    write(series / 'report.json', reference)
    return observations, series, record, manifest, reference, rgb


def test_recorded_loader_preserves_raw_pixels_and_typed_noise(tmp_path):
    observations, series, record, manifest, _, rgb = recorded(tmp_path)
    actual, samples = load_recorded_samples(observations, series)
    assert actual == manifest and samples[0]['identity'] == record
    assert np.array_equal(samples[0]['rgb'], rgb)
    assert samples[0]['original']['tokens'].dtype == torch.int64
    assert samples[0]['original']['rng_states'].dtype == torch.uint8
    assert samples[0]['original']['step_noise'].dtype == torch.float32


@pytest.mark.parametrize('change', ['image', 'text', 'frame', 'pixels', 'noise', 'escape'])
def test_recorded_input_corruption_is_rejected(tmp_path, change):
    observations, series, record, manifest, reference, _ = recorded(tmp_path)
    if change == 'image':
        Image.fromarray(np.ones((3, 4, 3), dtype=np.uint8)).save(observations / 'frame-3/observation.png')
    elif change == 'text':
        record['instruction'] = 'Different instruction'
    elif change == 'frame':
        reference['runs'][0]['frame'] = 4
    elif change == 'pixels':
        record['rgb_sha256'] = manifest['frames'][0]['rgb_sha256'] = '0' * 64
    elif change == 'noise':
        np.savez(series / 'sample-0/inputs.npz', initial_noise=np.ones(1))
    elif change == 'escape':
        manifest['frames'][0]['directory'] = '../reference'
    write(observations / 'frame-3/input.json', record)
    write(observations / 'manifest.json', manifest)
    reference['series_manifest_sha256'] = digest(observations / 'manifest.json')
    write(series / 'report.json', reference)
    with pytest.raises(ValueError):
        load_recorded_samples(observations, series)


def test_observer_preserves_actual_random_values_and_generator_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        before = torch.get_rng_state()
        first = torch.randn(2, 3)
        second = torch.randn_like(first)
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        with count_official_draws() as calls:
            observed_first = torch.randn(2, 3)
            observed_second = torch.randn_like(observed_first)
        assert torch.equal(first, observed_first) and torch.equal(second, observed_second)
        assert torch.equal(after, torch.get_rng_state())
        assert [item['operation'] for item in calls] == ['randn', 'randn_like']


def test_observer_restores_original_random_functions_on_exception():
    original = torch.randn, torch.randn_like
    with pytest.raises(RuntimeError, match='injected'):
        with count_official_draws():
            raise RuntimeError('injected failure')
    assert (torch.randn, torch.randn_like) == original


def test_observer_rejects_foreign_thread_before_consuming_rng():
    before = torch.get_rng_state()
    with count_official_draws() as calls, ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='another thread'):
            pool.submit(torch.randn, 3).result()
    assert calls == [] and torch.equal(before, torch.get_rng_state())
