# Deployment Numerical and Latency Reports

`vlaforge.validation.deployment_metrics` is a model-independent reporting
module. It consumes already paired action chunks or already measured raw
latencies. It does not run a model, replace its preprocessing/scheduler, or
certify that an input stream is a valid continuous-inference experiment.
It adds no required NumPy or PyTorch dependency.

## Full Action Chunks

```python
from vlaforge.validation.contracts import NumericContract
from vlaforge.validation.deployment_metrics import action_fidelity_report

report = action_fidelity_report(
    [{"sample_id": "frame-0001", "reference": reference, "candidate": candidate}],
    space="normalized",
    contract=NumericContract(absolute_tolerance=1e-5, relative_tolerance=1e-5),
    near_zero_norm=1e-12,
    discrete_dimensions=(6,),
)
```

The tolerance and dimension index above are examples, not a model-independent
acceptance threshold. Freeze these values before a formal experiment. The
module reuses `NumericContract` and its symmetric `math.isclose` rule:
`abs(a-b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))`. Its
`compare_tensor_hashes` field is a trace-comparison setting; action comparison
always examines every numeric value. Declared discrete dimensions ignore
floating tolerances and require exact values.

Each call handles the complete logical chunk with at least two axes. The last
axis is the action dimension; leading axes are retained, including a supplied
batch axis. No broadcasting, truncation, reshaping or normalization is implicit.
Reports from multiple samples require the same shape, coordinate space and
contract. NumPy arrays, PyTorch tensors and nested numeric lists are accepted;
Torch tensors are detached and copied to CPU outside the timed model path.

The report retains all reference/candidate values, their available dtype names,
the full-chunk cosine/MSE/RMSE/max-abs/mean-abs metrics, corresponding per-dimension
metrics, every failed sample ID, the worst element coordinates, and the worst
sample IDs for cosine and error metrics. `exact_values` means numeric equality,
not byte/dtype equivalence: signed zero and differing dtypes are not bitwise
checked. Discrete integers retain exact comparisons before float64 statistics.

Cosine is `null` if either full-vector L2 norm is at most `near_zero_norm`.
`cosine_status` distinguishes both/reference/candidate near zero. Such samples
are retained and excluded only from the minimum-defined-cosine selection.
Acceptance still uses every element's error; two zero vectors do not receive
an invented cosine of one. A cosine of one never overrides a scale error.
Ragged or mismatched shapes, empty chunks, booleans, NaN/Inf, invalid contracts,
or float64 statistic overflow fail with `ValueError`, rather than passing or
being silently omitted.

Run separate reports in normalized and physical units. The caller must retain
and pair the actual observation and noise tensors, checkpoint revision, N,
CFG, scheduler, preprocessing and postprocessing. For gripper classification or
token output, the caller supplies the actual discrete representation; this
module does not invent a threshold. Numerical acceptance alone is not proof of
robot task behavior or mathematically lossless deployment.

## Raw Latencies and CDF

`latency_report(records, deadline_ns=...)` accepts the existing benchmark CSV
record layout: `index,latency_ns`, with optional `repeat_id` and arbitrary extra
fields. Positive int64 nanoseconds and unique `(repeat_id,index)` pairs are
required. Missing repeat IDs denote one run. Original sample order and extra
fields are retained. Warmups must already be excluded by the producer.

The output contains pooled and per-repeat mean/min/max, nearest-rank
p50/p90/p95/p99, population standard deviation and the worst raw sample.
Nearest-rank matches `benchmark_real_model_paths.py`, rather than silently
mixing its convention with another tool's interpolated quantiles. An exact
empirical CDF groups equal latencies and records cumulative counts and
`P(latency > x)`. Deadline misses use strict `latency_ns > deadline_ns`.

`sequential_calls_per_second` is `1e9 / mean_ns`, not action Hz or automatically
fresh chunks/s. Only a synchronous fresh-chunk workload with the declared
measurement boundary justifies the latter label. Pooled percentiles are not
confidence intervals across independent processes. Repeat IDs alone do not
prove independent processes, changing observations, continuous state or the
formal minimum of 1,000 samples times five processes. Retain that provenance
in the experiment manifest and original raw records.

## File Reporting

```sh
PYTHONPATH=vlaforge/python python vlaforge/tools/report_deployment_metrics.py \
  --actions /absolute/run/actions.json \
  --latencies /absolute/run/process-0.csv \
  --latencies /absolute/run/process-1.csv \
  --deadline-ns 50000000 --output /absolute/run/report
```

The action JSON has `space`, `contract`, `samples`, and optional
`near_zero_norm`/`discrete_dimensions`, matching the Python example. `contract`
is mandatory in the CLI. Samples contain `sample_id`, `reference`, `candidate`.
Either actions or latencies can be omitted. Each CSV without a `repeat_id`
column gets the input file's ordinal as its group ID.

Outputs are `report.json`, `latency_raw.csv`, and `latency_cdf.csv`. The report
includes absolute source paths and SHA-256 hashes. Plot the CDF using
`latency_ns / 1e6` versus `cdf` with a post-step curve; use
`exceedance_probability` for the tail. The files contain measured data only;
the reporting tool does not fabricate a curve from summary percentiles.
An out-of-tolerance action report is still written, but the CLI exits with
code 1 so automation cannot treat it as a successful numerical gate.

Tests: `python -m pytest vlaforge/tests/unit/test_deployment_metrics.py -q`.
