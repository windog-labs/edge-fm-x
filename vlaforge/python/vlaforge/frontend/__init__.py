"""Restricted Python construction API for VLAForge IR."""

from vlaforge.frontend.annotations import RegionSpec, tensor_region
from vlaforge.frontend.builder import ModuleBuilder

__all__ = ["ModuleBuilder", "RegionSpec", "tensor_region"]

