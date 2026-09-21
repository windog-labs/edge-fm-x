"""Observe real exported activations without changing the graph or its outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from vlaforge.analysis.precision_calibration import (
    CalibrationSample,
    CalibrationSite,
    PrecisionCalibration,
    activation_statistics,
)


def metadata_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a lowercase SHA256 digest is required")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owned_tensor_snapshot(value):
    """Copy real values to independent canonical CPU storage, even singleton views."""
    import torch

    if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided:
        raise ValueError("probe requires a real strided Tensor")
    if value.device.type not in ("cpu", "cuda"):
        raise ValueError("probe only supports actual CPU or CUDA tensors")
    return torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu").copy_(value.detach())


def tensor_identity(value):
    """Value identity, portable across device placement but not dtype/shape."""
    import torch

    owned = owned_tensor_snapshot(value)
    raw = owned.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": str(owned.dtype), "shape": list(owned.shape), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def tensor_bundle_digest(values: Mapping[str, object]):
    if not isinstance(values, Mapping) or not values:
        raise ValueError("a nonempty named tensor mapping is required")
    if any(not isinstance(name, str) or not name or name != name.strip() for name in values):
        raise ValueError("tensor names must be normalized nonempty strings")
    return metadata_digest({name: tensor_identity(value) for name, value in sorted(values.items())})


@dataclass(frozen=True)
class ProbeRegion:
    name: str
    path: str
    artifact_sha256: str
    stage: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("probe region requires a normalized name")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("probe region requires a saved ExportedProgram path")
        _sha(self.artifact_sha256)
        if self.stage not in ("context", "iteration"):
            raise ValueError("probe region stage must be explicit")


class ProbeResult:
    """A complete, output-validated trajectory with privately owned snapshots."""

    def __init__(self, *, sample, sites, step_keys, profile_sha256,
                 numerical_context_sha256, observations, report):
        self._sample = sample
        self._sites = sites
        self._step_keys = step_keys
        self._profile_sha256 = profile_sha256
        self._numerical_context_sha256 = numerical_context_sha256
        self._observations = observations
        self._report = deepcopy(report)

    def report(self):
        return deepcopy(self._report)

    def owned_snapshot(self, site, step):
        """Return an independent CPU tensor after the complete-output gate."""
        for record, tensor in self._observations:
            if record["site"] == site and record["step"] == step:
                if activation_statistics(tensor) != record["statistics"]:
                    raise ValueError("owned activation snapshot changed after verification")
                return tensor.clone()
        raise ValueError("unknown validated activation site/step")

    def publish(self, collector: PrecisionCalibration):
        """Return an updated collector; failures never mutate the original."""
        if not isinstance(collector, PrecisionCalibration):
            raise TypeError("publish requires the actual calibration collector")
        if (collector.sites != self._sites or collector.step_keys != self._step_keys
                or collector.profile_sha256 != self._profile_sha256
                or collector.numerical_context_sha256 != self._numerical_context_sha256):
            raise ValueError("probe and calibration declarations differ")
        if self._sample not in collector.calibration_samples:
            raise ValueError("held-out or undeclared sample cannot contribute to fit")
        staged = deepcopy(collector)
        for record, tensor in self._observations:
            if activation_statistics(tensor) != record["statistics"]:
                raise ValueError("owned activation snapshot changed after verification")
            staged.observe(sample_id=self._sample.sample_id, site=record["site"],
                           step=record["step"], tensor=tensor)
        return staged


class PrecisionProbe:
    """Execute saved regions through their original FX operations and full I/O.

    The caller supplies the actual complete invocation through ``invoke(regions)``.
    No selected node is substituted and no output is removed or added. Selection
    inside nested higher-order GraphModules is deliberately unsupported here.
    This synchronizing diagnostic must never be used as timing evidence.
    """

    def __init__(self, *, regions, sites, step_keys, profile, numerical_context):
        import torch

        from vlaforge.frontend.effect_audit import audit_exported_program
        from vlaforge.numerical_context import NumericalContext

        if not isinstance(profile, Mapping) or not profile:
            raise ValueError("an explicit nonempty profile document is required")
        if not isinstance(numerical_context, NumericalContext) or numerical_context.partial:
            raise ValueError("probe requires a current versioned numerical context")
        numerical_context.require_current()
        self._profile = deepcopy(dict(profile))
        self._context = numerical_context
        self._regions = tuple(regions)
        self._sites = tuple(sites)
        self._step_keys = tuple(step_keys)
        if (not self._regions or not all(isinstance(item, ProbeRegion) for item in self._regions)
                or len({item.name for item in self._regions}) != len(self._regions)):
            raise ValueError("unique typed probe regions are required")
        if (not self._sites or not all(isinstance(item, CalibrationSite) for item in self._sites)
                or len({item.name for item in self._sites}) != len(self._sites)
                or len({(item.region, item.node) for item in self._sites}) != len(self._sites)):
            raise ValueError("unique typed activation sites are required")
        if (not self._step_keys or len(set(self._step_keys)) != len(self._step_keys)
                or any(not isinstance(key, str) or not key or key != key.strip() for key in self._step_keys)):
            raise ValueError("unique actual schedule step keys are required")
        self._profile_sha256 = metadata_digest(self._profile)
        self._numerical_context_sha256 = metadata_digest(self._context.to_dict())
        self._modules, self._audits, self._versions, self._input_profiles = {}, {}, {}, {}
        self._active = None
        declarations = {item.name: item for item in self._regions}
        for site in self._sites:
            region = declarations.get(site.region)
            if region is None or region.artifact_sha256 != site.artifact_sha256 or region.stage != site.stage:
                raise ValueError("site must bind the exact declared region artifact and stage")
        for region in self._regions:
            if _file_sha(region.path) != region.artifact_sha256:
                raise ValueError("saved ExportedProgram hash mismatch")
            exported = torch.export.load(region.path)
            audit = audit_exported_program(exported)
            if not audit.passed:
                raise ValueError("probe rejects hidden effects: " + str(audit.to_dict()))
            module = exported.module()
            nodes = {node.name: node for node in module.graph.nodes}
            for site in self._sites:
                if site.region == region.name and (site.node not in nodes or nodes[site.node].op != "call_function"):
                    raise ValueError("selection requires an existing top-level call_function Tensor node")
            self._modules[region.name] = module
            self._audits[region.name] = audit.to_dict()
            self._versions[region.name] = self._state_versions(module)
            profiles = []
            for node in module.graph.nodes:
                if node.op == "placeholder":
                    value = node.meta.get("val")
                    if not isinstance(value, torch.Tensor) or any(type(size) is not int for size in value.shape):
                        raise ValueError("probe supports static Tensor input profiles only")
                    profiles.append((str(value.dtype), tuple(value.shape), str(value.device)))
            self._input_profiles[region.name] = tuple(profiles)
        self._persistent_identity = self._state_identity()

    @property
    def profile_sha256(self):
        return self._profile_sha256

    @property
    def numerical_context_sha256(self):
        return self._numerical_context_sha256

    @staticmethod
    def _state_versions(module):
        return tuple((name, id(value), value._version) for name, value in
                     (*module.named_parameters(), *module.named_buffers()))

    @staticmethod
    def _input_metadata(values):
        return tuple((name, tuple(value.shape), tuple(value.stride()), value.storage_offset(),
                      str(value.dtype), str(value.device)) for name, value in sorted(values.items()))

    def _state_identity(self):
        import torch
        from torch.utils._pytree import tree_flatten

        result = []
        for region, module in self._modules.items():
            values = dict((*module.named_parameters(), *module.named_buffers()))
            constants = {}
            for graph_name, graph in module.named_modules():
                if not isinstance(graph, torch.fx.GraphModule):
                    continue
                for node in graph.graph.nodes:
                    if node.op != "get_attr":
                        continue
                    value = graph
                    for part in node.target.split("."):
                        value = getattr(value, part)
                    if isinstance(value, torch.fx.GraphModule):
                        continue
                    leaves, structure = tree_flatten(value)
                    for index, leaf in enumerate(leaves):
                        name = f"{graph_name}:{node.target}:{index}"
                        if isinstance(leaf, torch.Tensor):
                            values[name] = leaf
                        elif leaf is None or type(leaf) in (bool, int, float, str) or isinstance(leaf, (torch.dtype, torch.device)):
                            constants[name] = (str(structure), type(leaf).__name__, str(leaf))
                        else:
                            raise ValueError("unsupported persistent graph attribute for content audit")
            seen = {}
            for name, value in sorted(values.items()):
                # Tied parameters/get_attrs share the same content hash, without
                # hiding which names or storage views participate in the graph.
                if id(value) not in seen:
                    seen[id(value)] = tensor_identity(value)
                result.append((region, name, id(value), value.data_ptr(), tuple(value.stride()),
                               value.storage_offset(), seen[id(value)]))
            result.append((region, "non_tensor_constants", constants))
        return result

    def _call_region(self, declaration, *args):
        import torch

        if self._active is None:
            raise ValueError("region call is outside an active probe sample")
        active = self._active
        count = active["counts"][declaration.name]
        expected = len(self._step_keys) if declaration.stage == "iteration" else 1
        if count >= expected:
            raise ValueError("too many actual region calls for declared schedule")
        module = self._modules[declaration.name]
        observed_profile = tuple((str(value.dtype), tuple(value.shape), str(value.device)) for value in args)
        if observed_profile != self._input_profiles[declaration.name]:
            raise ValueError("actual region inputs differ from the saved static profile")
        if self._state_versions(module) != self._versions[declaration.name]:
            raise ValueError("persistent region state changed before observation")
        sites = {site.node: site for site in self._sites if site.region == declaration.name}
        step = count if declaration.stage == "iteration" else None
        before = [tensor_identity(value) for value in args]

        class Observer(torch.fx.Interpreter):
            def run_node(self, node):
                value = super().run_node(node)
                if node.name in sites:
                    owned = owned_tensor_snapshot(value)
                    statistics = activation_statistics(owned)
                    record = {"site": sites[node.name].name, "region": declaration.name,
                              "node": node.name, "step": step,
                              "step_key": self_step_keys[step] if step is not None else None,
                              "statistics": statistics}
                    active["observations"].append((record, owned))
                return value

        self_step_keys = self._step_keys
        result = Observer(module).run(*args)
        if before != [tensor_identity(value) for value in args]:
            raise ValueError("external region inputs changed during observation")
        if self._state_versions(module) != self._versions[declaration.name]:
            raise ValueError("persistent region state changed during observation")
        active["counts"][declaration.name] += 1
        active["calls"].append({"region": declaration.name, "step": step,
                                "step_key": self._step_keys[step] if step is not None else None,
                                "inputs": before})
        return result

    def observe_sample(self, *, sample, inputs, noise_names, expected_outputs, invoke):
        import torch
        from torch.utils._pytree import tree_flatten

        if self._active is not None:
            raise ValueError("probe samples cannot overlap or reenter")
        self._context.require_current()
        if not isinstance(sample, CalibrationSample) or not isinstance(inputs, Mapping):
            raise TypeError("typed sample and actual named input tensors are required")
        noise_names = tuple(noise_names)
        if not noise_names or len(set(noise_names)) != len(noise_names) or any(name not in inputs for name in noise_names):
            raise ValueError("explicit unique noise inputs are required")
        observation = {key: value for key, value in inputs.items() if key not in noise_names}
        noise = {key: inputs[key] for key in noise_names}
        if (tensor_bundle_digest(observation) != sample.input_sha256
                or tensor_bundle_digest(noise) != sample.noise_sha256):
            raise ValueError("actual observation/noise does not match locked sample")
        external_before = tensor_bundle_digest(inputs)
        metadata_before = self._input_metadata(inputs)
        expected_leaves, expected_tree = tree_flatten(expected_outputs)
        if not expected_leaves:
            raise ValueError("complete official output references are required")
        expected_identities = [tensor_identity(value) for value in expected_leaves]
        if self._state_identity() != self._persistent_identity:
            raise ValueError("persistent state/constant contents changed before invocation")
        devices = sorted({value.device.index for value in inputs.values() if value.device.type == "cuda"})
        rng_before = [tensor_identity(torch.get_rng_state())]
        rng_before += [tensor_identity(torch.cuda.get_rng_state(index)) for index in devices]
        self._active = {"counts": {item.name: 0 for item in self._regions}, "calls": [], "observations": []}
        try:
            callables = MappingProxyType({item.name: (lambda *args, declaration=item: self._call_region(declaration, *args))
                                          for item in self._regions})
            actual_outputs = invoke(callables)
            for region in self._regions:
                expected = len(self._step_keys) if region.stage == "iteration" else 1
                if self._active["counts"][region.name] != expected:
                    raise ValueError("incomplete actual region trajectory")
            actual, actual_tree = tree_flatten(actual_outputs)
            if not actual or actual_tree != expected_tree:
                raise ValueError("complete original output structure differs from reference")
            if [tensor_identity(value) for value in expected_leaves] != expected_identities:
                raise ValueError("official output references changed during invocation")
            output_records = []
            for value, expected_id in zip(actual, expected_identities, strict=True):
                actual_id = tensor_identity(value)
                if actual_id != expected_id:
                    raise ValueError("complete output is not bitwise equal to official reference")
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite complete output cannot validate calibration")
                output_records.append(actual_id)
            if tensor_bundle_digest(inputs) != external_before or self._input_metadata(inputs) != metadata_before:
                raise ValueError("external invocation inputs changed")
            if any(self._state_versions(module) != self._versions[name] for name, module in self._modules.items()):
                raise ValueError("persistent region state changed after invocation")
            if self._state_identity() != self._persistent_identity:
                raise ValueError("persistent state/constant contents changed after invocation")
            rng_after = [tensor_identity(torch.get_rng_state())]
            rng_after += [tensor_identity(torch.cuda.get_rng_state(index)) for index in devices]
            if rng_before != rng_after:
                raise ValueError("observer invocation changed implicit RNG state")
            self._context.require_current()
            expected_count = sum(len(self._step_keys) if site.stage == "iteration" else 1 for site in self._sites)
            if len(self._active["observations"]) != expected_count:
                raise ValueError("incomplete selected activation coverage")
            report = {"schema": "vlaforge.precision_probe/1", "status": "full_output_validated",
                      "sample": asdict(sample), "regions": [asdict(item) for item in self._regions],
                      "sites": [asdict(item) for item in self._sites], "step_keys": list(self._step_keys),
                      "profile_sha256": self.profile_sha256, "profile": self._profile,
                      "numerical_context_sha256": self.numerical_context_sha256,
                      "numerical_context": self._context.to_dict(),
                      "calls": self._active["calls"], "outputs": output_records,
                      "observations": [item for item, _ in self._active["observations"]],
                      "effects": self._audits, "implicit_rng_unchanged": True,
                      "external_inputs_unchanged": True, "complete_output_bitwise_equal": True,
                      "reference_identity_frozen_before_invocation": True,
                      "persistent_state_and_constants_content_unchanged": True,
                      "invocation_boundary": "caller-supplied complete invocation; tool must bind actual source IR",
                      "timing_evidence": False, "real_low_precision_kernel_verified": False,
                      "nested_node_selection_supported": False}
            return ProbeResult(sample=sample, sites=self._sites, step_keys=self._step_keys,
                               profile_sha256=self.profile_sha256,
                               numerical_context_sha256=self.numerical_context_sha256,
                               observations=self._active["observations"], report=report)
        finally:
            self._active = None
