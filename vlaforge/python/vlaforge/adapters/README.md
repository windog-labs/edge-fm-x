# Adapter Layout

Adapters are organized by model family. A model package owns checkpoint loading,
frontend processing, reference execution, and deployment assembly for that model.

```text
adapters/
  shared/          model-independent contracts, fixtures, and output helpers
  openpi/          OpenPI assets, capture, processing, and native deployment
  smolvla/         SmolVLA fixture, frontend, processing, and deployment
  rdt/             RDT assets, reference, input, and scheduler code
  cogact/          CogACT observation and partition adapters
  openvla/         OpenVLA fixture, frontend, partition, and deployment
  diffusiondrive/  DiffusionDrive artifact and runtime adapters
  autovla/         AutoVLA adapter
  minddrive/       MindDrive adapter
  pi0/             PI0 fixture adapter
```

The root package exposes only the aggregate public API. Model-specific code should
import from its model package, for example
`vlaforge.adapters.openpi.openpi_frontend` or
`vlaforge.adapters.smolvla.smolvla_real`; shared utilities live under
`vlaforge.adapters.shared`.

Model packages may depend on `shared`; `shared` must not depend on a model
package. Cross-model deployment and benchmark utilities belong outside this
directory, in the runtime or validation packages.
