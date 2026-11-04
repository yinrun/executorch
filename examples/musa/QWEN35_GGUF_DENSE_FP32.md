# Dense FP32 weight-cache investigation

## Decision

Do not enable a dense FP32 weight cache for this workload. Neither muBLAS
nor the six tested native GEMV layouts improves on the compressed native
path. No production kernel, loader, export, runtime or default changes.
These are microbenchmarks, not a new whole-model accuracy or throughput run.
The latest qualified caller-owned Split-K series remains **7.5716 median
decode tokens/s**, versus its paired original-deployment control of 7.3557;
see [the source integration results](QWEN35_GGUF_SPLIT_K.md).
The **10 tokens/s target remains unmet**.

The model remains Qwen3.5-4B Q4_K_M with FP32 activations, state and
accumulation. Decode original packed weights once into FP32 before timing,
then reuse that matrix without requantization. The existing native embedding
decoder produces the cache; this probe does not independently verify every
cached weight against a second decoder. There is no BF16-weight conversion,
MTP, vocabulary pruning or tolerance relaxation.

## Measurement contract

Read real matrix bytes from the original GGUF. The native reference uses
the qualified caller-owned Split-K runtime, with lossless Q6 preprocessing
and the grouped output head. Its long-K microbenchmark entry point uses
legacy scratch partials plus a separate reduction; the qualified whole model
instead fuses that reduction into its consumer. Consequently these timings
are not a measurement of the fused whole-model operation boundary.

Each variant compares three complete outputs for changing FP32 inputs:
all values must be finite and satisfy `0.001 + 0.001 * abs(reference)`.
Argmax differences are recorded separately. Timings alternate native control
and candidate for 102 rounds, discarding the first two, using native GPU
events and synchronization after each event pair. Preparation and output
comparisons are outside timing. The shared desktop GPU has unlocked clocks;
per-variant controls vary and these are exploratory single sweeps, not
exclusive-GPU or whole-model ABBA measurements.

All allocations are bounded and checked against available memory with a
1 GiB reserve. Processes run serially, releasing their buffers on exit.
Original accepted artifacts and the qualified Split-K library are unchanged.

## muBLAS probe

Both SGEMV and SGEMM explicitly request `MUBLAS_PEDANTIC_MATH`; querying the
handle returns mode 2. This confirms the API setting, not generated hardware
instructions. The host harness disables implicit FP contraction. Ten real
matrices pass both variants, giving 60 complete output checks. Neither
variant improves performance: their native/candidate median ratios range
from 0.0943 to 0.9942, the latter being only the tiny 32-row projection.

The 248320-by-2560 output head passes three additional SGEMV comparisons,
but runs at only 0.1903 times native speed in the original sweep. Its SGEMM
fails the unchanged micro tolerance. A separate diagnostic reproduces the
failure: maximum absolute error 6.14 and maximum tolerance-scaled error
859.944, despite matching argmax on the three inputs. There is no accepted
SGEMM timing for that head. This does not establish the cause or prove use
of TF32; no mode waiver or looser precision gate is introduced.

## Direct native FP32 probe

The native candidates read FP32 weights directly, use explicit FP32 FMA,
and vary component width, rows per block and K partitioning:
`(width, tile, parts)` is `(4,1,1)`, `(4,2,1)`, `(16,1,1)`, `(16,2,1)`,
`(4,2,4)` or `(16,1,4)`. All use 128 threads and explicit subgroups of
8 or 32 lanes. Compilation targets MP22 with ILP scheduling and no implicit
FP contraction. A four-part candidate includes its final reduction in timing.

All six variants on six real matrices pass **108 complete-output checks**,
with zero argmax mismatches. Some combinations are byte-identical; others
change reduction order and satisfy tolerance. None is model-qualified.

| Matrix | Shape (rows x K) | Best native/candidate speed ratio |
| --- | --- | ---: |
| Q5 QKV | 8192 x 2560 | 0.5593 |
| Q6 FFN down | 2560 x 9216 | 0.6492 |
| Q4 FFN down | 2560 x 9216 | 0.4407 |
| Q4 SSM output | 2560 x 4096 | 0.5401 |
| Q4 merged FFN gate/up | 18432 x 2560 | 0.4250 |
| Q6 output head | 248320 x 2560 | 0.3563 |

Even selecting each matrix's best variant gives a slowdown of about
1.54 to 2.81 times. The actual merged FFN reads both gate and up matrices,
not a single half-sized gate proxy. The best head sample is 27.0326 ms
against its paired compressed control of 9.6323 ms.

Width-16 kernels report 64 or 128 bytes of local memory, despite zero
reported spill loads/stores. Width-4 kernels report zero local memory and
also regress. Thus poor width-16 resource usage does not explain away the
entire negative result; it also prevents attributing every slowdown solely
to memory bandwidth.

A 256-weight FP32 block is 1024 bytes, versus 144 for Q4, 176 for Q5,
210 for raw Q6 or 320 for the current predecoded Q6 representation.
The head alone requires 2,542,796,800 bytes of dense storage, versus
794,624,000 bytes of predecoded Q6. Increased traffic is a plausible cost,
not a directly measured bandwidth attribution. Keep the compressed path;
future probes should target its critical path without assuming expansion
or fewer launches automatically improves throughput.

## Local evidence

ETSupport artifacts under `artifacts/gguf-20261029`:

- `dense-fp32-probe-v1`: muBLAS harness, serial sweep, hashes and raw results;
  the failed overall status retains the expected head SGEMM rejection.
- `dense-fp32-native-probe-v1`: native kernel, harness, six libraries and
  compiler resource logs, serial sweep and `vendor-head-diagnostic.json`.
- `audit_dense_fp32_v1.py`, `dense-fp32-decision-v1.json`: independently
  rehashed inputs/binaries and assertion checks for all 108 native plus
  63 original vendor passing comparisons; the diagnostic rerun is separate.

No full-model rerun is justified by these regressions. Existing full-model
acceptance applies only to the unchanged qualified deployment, not these
experimental dense kernels.
