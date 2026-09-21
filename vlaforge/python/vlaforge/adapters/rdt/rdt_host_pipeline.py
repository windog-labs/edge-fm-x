"""Recorded AgileX inputs through official RDT modules or a generated Session.

The input adapter follows the pinned AgileX image/state and T5 tokenizer calls.
Every processed port must match the archived official inputs before measurement.
The official sampler keeps its original scheduler and random draw; the saved
generator state reproduces the exact noise supplied explicitly to the Session.
"""

import json
from pathlib import Path

from vlaforge.adapters.rdt.rdt_assets import file_identity, verify_assets, verify_source
from vlaforge.adapters.rdt.rdt_inputs import load_recorded_input
from vlaforge.adapters.rdt.rdt_reference import RDTConfig, _import_source, check_environment, load_rdt
from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext
from vlaforge.validation.host_pipeline import digest
from vlaforge.validation.native_session import NativeTensorSession


def read(path):
    return json.loads(Path(path).read_text())


def pad_recorded_image(image, background):
    """Preserve upstream integer centering and original channel ordering."""
    from PIL import Image

    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    result = Image.new(image.mode, (side, side), background)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


class RDTInputProcessor:
    def __init__(self, tokenizer, image_processor, wrapper, max_length, *, device):
        self.tokenizer, self.image_processor, self.wrapper = tokenizer, image_processor, wrapper
        self.max_length, self.device = max_length, device
        dataset = wrapper.args['dataset']
        if dataset.get('auto_adjust_image_brightness', False) or dataset.get('image_aspect_ratio', 'pad') != 'pad':
            raise ValueError('this recorded profile requires upstream padding without brightness adjustment')

    def prepare(self, sample):
        import numpy as np
        import torch
        from PIL import Image

        arrays = sample['arrays']
        if (arrays['proprio'].shape != (1, 14) or arrays['proprio'].dtype != np.float32
                or not np.isfinite(arrays['proprio']).all()):
            raise ValueError('recorded native AgileX state must remain finite [1,14] FP32')
        if not isinstance(sample['instruction'], str) or not sample['instruction'].strip():
            raise ValueError('a recorded instruction is required')
        images = []
        background = tuple(int(value * 255) for value in self.image_processor.image_mean)
        for index in range(6):
            value = arrays[f'image_{index}']
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or 0 in value.shape:
                raise ValueError('six actual decoded uint8 camera views are required')
            image = pad_recorded_image(Image.fromarray(value), background)
            images.append(self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0])
        pixels = torch.stack(images).to(self.device, dtype=torch.bfloat16)
        text = self.tokenizer([sample['instruction']], max_length=self.max_length,
            padding='longest', truncation=True, return_attention_mask=True,
            add_special_tokens=True, return_tensors='pt')
        joints = torch.from_numpy(arrays['proprio'].copy()).to(self.device).unsqueeze(0)
        state, mask = self.wrapper._format_joint_to_state(joints)
        return {'token_ids': text['input_ids'].to(self.device).contiguous(),
                'text_mask': text['attention_mask'].to(self.device).contiguous(),
                'pixels': pixels.contiguous(),
                'unified_state': state.to(self.device, dtype=torch.bfloat16)[:, -1:, :].contiguous(),
                'action_mask': mask.to(self.device, dtype=torch.bfloat16).unsqueeze(1).contiguous(),
                'control_frequency': torch.tensor([sample['control_frequency']]).to(self.device).contiguous(),
                'noise': sample['noise'].to(self.device).contiguous()}


class RDTHostPipeline:
    def __init__(self, config):
        import torch
        import yaml
        from transformers import AutoTokenizer, SiglipImageProcessor

        for name, expected in config['files_sha256'].items():
            if digest(name) != expected:
                raise ValueError('original RDT evidence changed: ' + name)
        self.backend = config['backend']
        self.context = NumericalContext.from_dict(config['numerical_context'])
        self.context.require_current()
        source, assets, series = (Path(config[name]) for name in ('upstream', 'assets', 'series'))
        source_identity = verify_source(source)
        environment = check_environment('torch210-cu128')
        self.loaded, self.native, self.loops = None, None, []
        if self.backend == 'official-pytorch':
            self.loaded = load_rdt(RDTConfig(source, assets, 'cuda:0', 'torch210-cu128'))
            tokenizer = self.loaded.text_embedder.tokenizer
            image_processor = self.loaded.vision_tower.image_processor
            wrapper = object.__new__(self.loaded.wrapper_class)
            wrapper.args = self.loaded.source_config
            asset_identity = self.loaded.provenance['assets']
        elif self.backend == 'generated-session':
            asset_identity = verify_assets(assets)
            modules = _import_source(source)
            wrapper = object.__new__(modules['wrapper'].RoboticDiffusionTransformerModel)
            wrapper.args = yaml.safe_load((source / 'configs/base.yaml').read_text())
            tokenizer = AutoTokenizer.from_pretrained(assets / 'text', local_files_only=True,
                model_max_length=read(assets / 'policy/config.json')['max_lang_cond_len'])
            image_processor = SiglipImageProcessor.from_pretrained(assets / 'vision', local_files_only=True)
            if digest(config['library_receipt']) != config['library_receipt_sha256']:
                raise ValueError('native build receipt identity differs')
            receipt = read(config['library_receipt'])
            if receipt['status'] != 'built_not_executed' or digest(config['cuda_runtime']) != config['cuda_runtime_sha256']:
                raise ValueError('native build or CUDA runtime differs')
            bundle = Path(receipt['bundle'])
            manifest = load_bundle_manifest(bundle / 'bundle.json')
            manifest.verify_files(bundle)
            selections = [item for item in manifest.generated_sources if item.role == 'loop_execution']
            if len(selections) != 1:
                raise ValueError('explicit native loop selection required')
            self.loops = read(bundle / selections[0].path)['loops']
            if any(loop['policy'] != config['policy'] for loop in self.loops):
                raise ValueError('compiled loop policy differs')
            module = parse_canonical_json((bundle / manifest.semantic_ir.path).read_text())
            self.native = NativeTensorSession(receipt['library'], module, library_sha256=receipt['library_sha256'],
                bundle=bundle, bundle_sha256=receipt['bundle_sha256'], cuda_runtime=config['cuda_runtime'])
        else:
            raise ValueError('unsupported RDT host backend')
        self.wrapper = wrapper
        self.processor = RDTInputProcessor(tokenizer, image_processor, wrapper,
            read(assets / 'policy/config.json')['max_lang_cond_len'], device='cuda:0')
        observations, reference = read(series / 'observations.json'), read(series / 'reference.json')
        if (observations['status'] != 'distinct_recorded_observations_prepared'
                or reference['status'] != 'complete_official_reference_series'
                or len(observations['samples']) != len(reference['samples'])):
            raise ValueError('complete recorded RDT observation/reference series required')
        self.samples, identities = [], []
        for index, (observation, ref) in enumerate(zip(observations['samples'], reference['samples'], strict=True)):
            if observation['index'] != ref['index'] or ref['index'] != index:
                raise ValueError('recorded RDT sample ordering changed')
            pack = series / 'observations' / f'sample-{index}'
            if file_identity(pack / 'manifest.json') != observation['input_manifest']:
                raise ValueError('recorded input manifest changed')
            arrays, manifest = load_recorded_input(pack, source_root=source)
            if any(ref[name] != observation[name] or ref[name] != manifest[name] for name in ('step', 'seed')):
                raise ValueError('recorded step/seed pairing changed')
            folder = series / 'references' / f'sample-{index}'
            for name, expected in ref['files'].items():
                if file_identity(folder / name) != expected:
                    raise ValueError('RDT original reference changed: ' + name)
            saved = torch.load(folder / 'official-scheduler.pt', map_location='cpu', weights_only=True)
            noise = saved['noise']
            if (noise.dtype != torch.bfloat16 or tuple(noise.shape) != (1, 64, 128)
                    or not torch.isfinite(noise).all()
                    or digest(series / 'native-inputs' / f'sample-{index}' / 'noise.bin') != ref['saved_noise_sha256']):
                raise ValueError('saved complete RDT noise profile changed')
            if noise.contiguous().view(torch.uint8).numpy().tobytes() != (series / 'native-inputs' / f'sample-{index}' / 'noise.bin').read_bytes():
                raise ValueError('saved RNG noise differs from the original input tensor')
            self.samples.append({'arrays': arrays, 'instruction': manifest['instruction'],
                'control_frequency': manifest['control_frequency'], 'noise': noise,
                'rng_before': saved['rng_before'], 'rng_after': saved['rng_after']})
            identities.append({'index': index, 'step': ref['step'], 'seed': ref['seed'],
                'input_manifest': observation['input_manifest'], 'noise_sha256': ref['saved_noise_sha256'],
                'image_channel_semantics': manifest['image_channel_semantics']})
        self.identities = {'source': source_identity, 'assets': asset_identity, 'environment': environment, 'samples': identities}

    def prepare(self, sample):
        import torch

        values = self.processor.prepare(sample)
        if self.loaded:
            torch.cuda.set_rng_state(sample['rng_before'], 'cuda:0')
        return values

    def infer(self, values):
        import torch

        if self.native:
            return self.native.run(values)
        language = self.loaded.text_embedder.model(input_ids=values['token_ids'],
            attention_mask=values['text_mask'])['last_hidden_state'].detach()
        vision = self.loaded.vision_tower(values['pixels']).detach()
        vision = vision.reshape(-1, self.loaded.vision_tower.hidden_size).unsqueeze(0)
        unified = self.loaded.policy.predict_action(lang_tokens=language,
            lang_attn_mask=torch.ones(language.shape[:2], dtype=torch.bool, device=language.device),
            img_tokens=vision, state_tokens=values['unified_state'], action_mask=values['action_mask'],
            ctrl_freqs=values['control_frequency'])
        native = self.wrapper._unformat_action_to_joint(unified)
        return {'action_chunk': unified.detach().cpu().contiguous(),
                'robot_action_chunk': native.detach().cpu().contiguous()}

    def qualify_inputs(self, originals):
        import torch

        if len(originals) != len(self.samples):
            raise ValueError('raw/reference sample counts differ')
        result = []
        for index, (sample, original) in enumerate(zip(self.samples, originals, strict=True)):
            values = self.prepare(sample)
            if set(values) != set(original['inputs']):
                raise ValueError('raw input adapter ports differ')
            for name, value in values.items():
                actual = value.cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                if actual != Path(original['inputs'][name]).read_bytes():
                    raise ValueError('raw preprocessing differs from official input: ' + name)
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.set_rng_state(sample['rng_before'], 'cuda:0')
                noise = torch.randn(sample['noise'].shape, dtype=torch.bfloat16, device='cuda:0')
                if not torch.equal(noise.cpu(), sample['noise']) or not torch.equal(torch.cuda.get_rng_state(0), sample['rng_after']):
                    raise ValueError('saved generator state does not reproduce the original full noise')
            result.append({'sample': index, 'input_tensors': len(values), 'all_byte_exact': True, 'rng_noise_exact': True})
        return result

    def evidence(self):
        result = {'backend': self.backend, 'identities': self.identities, 'python_host': True,
            'no_python_deployment': False, 'loops': self.loops, 'physical_robot_units_verified': False,
            'official_scope': 'official T5, SigLIP, sampler and output transform with independently qualified input adapter; Torch 2.10',
            'rng_scope': 'official original draw from saved state; Session consumes the saved noise tensor',
            'output_scope': 'complete BF16 unified/native outputs before the wrapper final lossless float32 cast'}
        if self.native:
            result.update(native=self.native.evidence, replay=[self.native.replay_info(loop['task_id'])
                for loop in self.loops if loop['policy'] == 'required'])
        return result

    def close(self):
        if self.native:
            self.native.close()


def create(config):
    return RDTHostPipeline(config)
