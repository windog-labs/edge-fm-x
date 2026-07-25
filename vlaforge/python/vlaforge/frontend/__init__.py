"""Restricted frontend and torch.export capture API for IR v0.2."""

from vlaforge.frontend.annotations import tensor_region
from vlaforge.frontend.artifact_manifest import (
    RegionCompileRequest,
    finalize_region_artifact,
    make_compile_request,
)
from vlaforge.frontend.audit_report import (
    ModelFrontendAudit,
    RegionAuditRecord,
)
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.frontend.export import (
    load_exported_region,
    save_exported_region,
)
from vlaforge.frontend.region_capture import (
    CaptureEvidence,
    CaptureOutcome,
    capture_annotated_region,
    capture_region,
)
from vlaforge.frontend.shape_profile import DynamicDimension, ShapeProfile
from vlaforge.frontend.state_lifting import (
    PersistentStateEvidence,
    lift_persistent_states,
)
from vlaforge.frontend.unsupported import FrontendUnsupportedError

__all__ = [
    "CaptureEvidence",
    "CaptureOutcome",
    "DynamicDimension",
    "FrontendUnsupportedError",
    "ModelFrontendAudit",
    "ModuleBuilder",
    "PersistentStateEvidence",
    "RegionAuditRecord",
    "RegionCompileRequest",
    "ShapeProfile",
    "capture_annotated_region",
    "capture_region",
    "finalize_region_artifact",
    "lift_persistent_states",
    "load_exported_region",
    "make_compile_request",
    "save_exported_region",
    "tensor_region",
]
