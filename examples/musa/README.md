# MUSA AOTI model acceptance

## Qwen3.5-4B GGUF bring-up

The next target is the unchanged Qwen3.5-4B Q4_K_M GGUF with its BF16 vision
projector, not a uniform INT4 requantization of the accepted 0.8B model. The local
text artifact contains 203 Q4_K, 24 Q5_K, 22 Q6_K and 177 F32 tensors; the vision
artifact contains 98 BF16 and 200 F32 tensors. Qwen3-VL-8B adaptation is deferred.

`examples.musa.aoti_gguf_inventory` records per-file/per-tensor hashes, encodings,
dimensions and byte sizes without changing weights. It rejects unsupported
encodings, malformed rows and files changing during inspection. It requires the
`gguf` Python package (or the checked llama.cpp `gguf-py` on `PYTHONPATH`).

`backends.musa.gguf` adds GPU-only Q4_K/Q5_K/Q6_K block decoding. A dedicated
`executorch_musa.gguf_decode` op preserves the Edge graph boundary; only the MUSA
delegate decomposes it. FP16 scale bytes are decoded arithmetically, including
subnormals, rather than depending on an unavailable native dtype-view shim.
This decoder path materializes decoded matrices; it remains useful as a GPU
reference, not as the native packed-weight execution implementation.

Fifty-seven decoder/Qwen/reference/packaging regressions pass, including native
PTE execution of the three decoders, all 65536 FP16 scale encodings in GPU eager,
45 dtype/width decode combinations and changed-bit negative controls. Three
16-row samples from the actual 4B GGUF decode byte-exactly in native execution.
The accompanying FP32 matrix outputs are not byte-identical to PyTorch/MUSA
(maximum absolute errors up to 5.484e-6); the original strict probe failure is
preserved, not relabeled as full acceptance. The 4B text state/mapping, BF16
vision, context-4096 multimodal inference and deployment benchmarks remain
pending. See workspace `QWEN35_4B_Q4KM_PLAN.md` for the complete
scope and the report/log discrepancies motivating fresh GPU reference runs.

`backends.musa.gguf_packed` now provides opaque `GgufLinear` and `GgufEmbedding`
operations. Native MUSA kernels read unchanged Q4_K/Q5_K/Q6_K bytes directly;
linear accumulates in FP32 without a dense weight allocation. Embedding decodes
only selected rows and synchronously rejects invalid indices. Both require
contiguous raw uint8 rows; linear inputs/outputs are FP32, indices are int64.
The eager linear implementation decodes temporary row chunks for reference testing;
only the exported native path is the packed fused implementation. This first
kernel is not an optimized GEMM/prefill implementation or graph-capture promise.

Standard MUSA CMake builds compile `musa_gguf_kernels.mu` with `mcc`; the SDK
compiler is now required. `EXECUTORCH_MUSA_ARCHITECTURES` defaults to `22` and
accepts a semicolon-separated list of target architecture numbers. The current
device validation is MP22 only. Non-MUSA builds do not compile these kernels.

Native PTE tests retain each raw weight exactly once in PTD, shared by embedding
and linear, with no dense matrix constant. Five real 32-row 4B tensor probes
cover Q4_K/Q5_K/Q6_K and K=2560/4096/9216. Embeddings are byte-exact; linear is
checked against an FP64 MUSA oracle with an accumulation error bound and a
perturbed-output negative control, not a byte-equality claim.

The full Q6_K shared embedding/output head, **248320 x 2560**, also executes as
one native delegate. PTD is **521472256 bytes**, containing the original
521472000-byte packed weight once, not a Q4 rewrite or FP32 expansion. Two test
activation rows yield 496640 checked outputs: maximum error is 1.073e-6 versus
FP32 MUSA and 4.341e-7 versus the FP64 MUSA oracle. Both argmax indices agree;
7680 selected embedding values are byte-exact. Three executions were made but
only the last execution's outputs were saved. These are operator inputs, not
generated model answers, and do not establish transformer or vision accuracy.

The final packed build passes 62 Python decoder/packed/Qwen/reference/packaging
tests with no skips, five native GGUF tests and eleven existing native reduction
tests. One existing two-device reduction test is explicitly skipped on this
single-device setup. The accepted 0.8B runtime and deployment artifacts are not
overwritten by these versioned builds.

`examples.musa.aoti_qwen35_gguf.load_gguf_text(path, context_length=4096)` loads
the complete 32-layer text model without converting its packed weights to HF
matrices. It validates architecture, tensor names, shapes and formats; preserves
the GGUF tiled GDN head order, negative decay coefficients and already-offset
norm weights; and shares the original Q6_K embedding/output storage. Auxiliary
weights, activations and state use FP32. This does not turn the quantized model
into the earlier BF16 0.8B deployment. Eager linear references decode at most
4096 weight rows per call to bound temporary memory; the native kernel remains
packed throughout execution.

The first full-model export has one delegate, 4697 graph nodes and 64 mutable
GDN/KV buffers, with a capacity of 4096 positions. All 426 original GGUF tensors
match their PTD segments byte-for-byte; the shared head is serialized once.
Three input tokens replayed twice pass the predeclared FP32 comparison gate
(`atol=rtol=0.001`) against GPU eager, with maximum error 1.669e-5 and matching
GPU argmax. Reset outputs are byte-identical, not native/eager outputs.

The report's heterogeneous-computing text question also runs natively twice
with thinking disabled: each run generates 89 tokens including EOS, reaches
position 115, and the two runs' 116 full-logit outputs are byte-identical.
All 232 steps retain logits and GPU token outputs. This is a generation/reset
test, not 232 independent-reference accuracy comparisons or a performance test.
Sixty-six decoder/packed/mapping/Qwen/reference/packaging tests pass with no skips.

The original report text prompt now passes an independent Transformers-derived
same-GGUF GPU reference for all **116 positions / 28805120 logits**. Its 89
generated token IDs (including EOS) match native greedy feedback exactly. Maximum
absolute error is **3.6478e-5**, mean absolute error **1.5475e-6**, and no value
exceeds the predeclared `atol=rtol=0.001` gate. This compares one native sequence
against one fresh reference; the second native sequence is a separately rehashed
reset check. It is not native/reference byte equality or arbitrary-prompt proof.

`aoti_qwen35_gguf_reference.py` uses Transformers attention, MLP, RoPE and dynamic
cache with explicit GGUF norm, negative-A and tiled/grouped-head adapters, not
the ExecuTorch transformer. It independently maps all 426 tensors but shares the
separately tested GPU K-quant decoder. An optional bounded FP32 decoded cache is
reference-only; deployed PTD bytes remain unchanged. The 8 GiB reference cache
has byte-identical outputs to the uncached short probe on its three positions.
Four adapter/cache tests pass. Independent artifact rechecking validates saved
logits, greedy inputs, hashes, the perturbed-output negative control and ten
protected 0.8B files. Evidence: `hf-trajectory-v1/result.json` and
`hf-trajectory-audit-v1.json` under the workspace GGUF artifact directory.
No original floating-point 4B checkpoint was compared, so this does not measure
quantization loss. Module/cache device checks are not a full kernel trace.

Fresh diagnostic llama.cpp
runs report all layers offloaded but expose CPU Q6_K embedding GET_ROWS; forcing
the embedding weight onto MUSA does not fix the missing operator support. Their
three sampled argmax values agree, but logits differ by up to about 0.466 and
the GPU-only reference gate fails. These results are not accepted as a healthy
GPU baseline. Complete single/multi-image answers, actual context-4096 boundary
coverage, deployable model packaging and fair repeated benchmarks remain pending.

### Original BF16/F32 vision component

`aoti_qwen35_gguf_vision.py` loads all 298 original projector tensors, preserving
98 BF16 matrices and 200 F32 tensors. Activations, attention and normalization
are FP32; matrix weights are cast on device for computation. This is the original
mixed-storage projector, not a claim of all-BF16 arithmetic. Two static-grid
exports (30x40 and 20x40 patches) each retain all 298 original tensor byte strings
in PTD, including the two original F32 temporal patch slices.

The three report PNGs keep their original dimensions and use the local HF image
processor. Each image's 24 block outputs and final merged features match an
independently mapped HF GPU vision model exactly in eager mode: 72 block checks
and 2048000 projected FP32 values. Reconstructing the original Conv3D operation
from the two stored temporal slices is important: separate linear products plus
addition had small initial errors that grew in outlier hidden activations. That
failed `vision-probe-v1` is preserved; `vision-probe-v2` changes the computation,
not its comparison threshold.

Native execution passes the same `atol=rtol=0.001` gate on all three projected
feature arrays. Maximum errors for shapes/chart/text are respectively
**2.0313e-4 / 1.6344e-4 / 2.0647e-4**. Two fresh-process executions per image
produce byte-identical outputs (six calls total). The native library audit
excludes Torch/Python dependencies. Evidence is in `vision-native-v2` and
`vision-export-800-v1`; `vision-export-v1` retains its original native failure.
These are vision features, not complete image-question answers or a benchmark.

The original native failure exposed interleaved-head and padded-row BMM layouts
which the shim rejected. `musa_addmm.cpp` now packs only such supported layouts
using stream-aware device-to-device 2D copies, retaining existing dense paths,
span/alias checks and rejection of overlapping batches. All 31 native matrix
tests pass, including FP32/FP16/BF16, offsets and input-preservation checks for
the new layouts. No CPU tensor execution is added.

`aoti_qwen35_gguf_multimodal.py` prepares host image-placeholder metadata and
provides a single-token wrapper with separate cache positions, three-axis
interleaved MRoPE and a runtime image-embedding selector. Host position metadata
and GPU MRoPE match HF, including multiple images and position 4095; small-model
text/image selection and exported cache mutation pass unit tests. These unit
checks alone do not prove full 4B execution or a context-4096 boundary sequence.
The final regression runs 80 Python tests without skips. The independent
`vision-stage-audit-v1.json` rehashes the artifacts, rereads both PTDs and all
native feature outputs, checks a perturbed-output negative control and verifies
the ten protected 0.8B files. Its status is `passed_vision_component_only`.

### Original report multimodal generation

The complete 4B five-input step now exports as 4710 nodes, 64 mutable states
and one MUSA delegate. Inputs are token ID, physical cache position, three-axis
MRoPE position, FP32 image embedding and an image-selection flag. All 426
original text tensor byte strings remain unchanged in PTD. The six-step
text/image interface probe passes twice: 12 native/eager full-logit comparisons,
maximum error **2.5273e-5**, matching GPU argmax and six byte-exact reset pairs.
Evidence: `multimodal-step-export-v1` and `multimodal-step-native-v1`.

`multimodal-generation-v1` then runs the original text, single-image and
two-image questions, using freshly executed native vision features. It retains
context capacity 4096, greedy feedback and a 160-generated-token limit, with
thinking explicitly disabled. Prompt lengths are 28, 334 and 538; generation
lengths are 89, 95 and 68 including EOS. Each case finishes at EOS twice in the
same runner process. All 2298 full-vocabulary outputs and GPU token outputs are
saved; the 1149 cross-session reset pairs are byte-identical. The CPU controller
handles tokenization, input/output transport and metadata, not model tensors.

`multimodal-generation-audit-v1.json` independently rereads those outputs,
checks their hashes, GPU greedy feedback, image rows, physical/MRoPE positions,
EOS, native library isolation and ten protected 0.8B artifacts. The text case's
232 checks also pass against the frozen independent HF reference, with maximum
error **3.1233e-5**. The three fresh visual encodings pass the existing feature
gate. The audit status is deliberately `passed_generation_and_reset_only`:
image-question full-logit accuracy against independent HF is not yet accepted.

Answers correctly identify the three shapes/colors and chart values 12/7/19,
but OCR remains imperfect: `LLAMA 42` becomes `ILLAMA 42`, and the second image's
text is incomplete. Readable output and deterministic reset do not establish
numerical correctness, nor identify the cause of OCR errors. Full HF image/text
trajectory comparison is the next gate; first validate batched reference
prefill against the frozen stepwise text reference before using it for images.
These runs are not fair throughput measurements, standalone native-controller
deployment acceptance, original-BF16 quantization-loss measurements or an
actual position-4095 cache boundary test.

The first complete independent image/text comparison is now preserved as a
failure (`hf-multimodal-batched-v1`). Text passes 116 positions, but the single
image's 428-position trajectory has 96649 out-of-gate logits, maximum error
**0.0232811**, despite identical generated IDs. All violations are in the prompt;
the multi-image reference was not reached. Feeding identical native visual
features to HF batch prefill reduces violations to 40 and the maximum to
**0.0028620**. First-16-position stepwise HF probes pass with either feature
source, reaching **6.6102e-5** maximum error with identical native features.
These diagnostics separate feature propagation from batch/recurrent arithmetic;
they do not establish correctness for the untested remainder.

`vision-stages-v1` exports 30 intermediate outputs and reproduces the original
native final features byte-for-byte. Patch embedding and positional addition
are exact; the first block differs by at most **5.0068e-6**. Differences grow to
outliers in block 23 (maximum **1.201416**, four out-of-gate values), although
the merged features still pass the earlier component gate. Further intra-block
diagnosis is needed; do not relax the complete-model tolerance to pass this case.

`vision-block0-v1` further exposes 17 stages and reproduces that block's native
output byte-for-byte. Cosine/sine are exact; the first same-input discrepancy is
the first FP32 LayerNorm, maximum **2.3842e-6**. Installed TorchMUSA implements
FP32 LayerNorm using muDNN C++ with `VarMode::WELFORD`; the exported native path
currently uses a fused Triton normalization kernel. Matching that implementation
is a candidate fix to validate, not proof that all downstream discrepancies
are explained or that complete image/text accuracy has been repaired.

The opt-in `VisionLayerNorm` now routes contiguous FP32 `[800,1024]` and
`[1200,1024]` affine inputs through muDNN Welford on the current stream.
It rejects unsupported shapes, nonpositive/nonfinite epsilon and overlapping
outputs; ordinary models are not automatically rewritten. Four native tests
and four Python tests pass, including byte-identical TorchMUSA/native outputs
on both grids and nearly constant rows. No original model package is replaced.

`vision-block0-norm-v1` makes norm1, QKV, rotary Q/K and scaled attention
scores byte-identical. The first remaining same-input discrepancy is softmax,
maximum **2.3842e-7**; block output maximum is **3.8147e-6**. Applying all 49
norm replacements in `vision-stages-norm-v1` reduces the first image's final
feature maximum from **2.0313e-4** to **1.3423e-4**, but block 23 still has
one out-of-gate value (maximum **0.745850**). This is a diagnostic improvement,
not full-vision exactness or repaired end-to-end image/text accuracy. The next
controlled candidate is TorchMUSA-compatible muDNN softmax with identical inputs.
The 84-test packed/GGUF/Qwen/vision regression passes without skips on the new
runtime. `vision-norm-audit-v1.json` independently rehashes the diagnostics,
checks all 298 original vision tensors in the candidate PTD and preserves
17 protected files. Its status is `audited_candidate_diagnostic_only`.

The next opt-in `VISION_SOFTMAX` candidate uses muDNN `ACCURATE` on contiguous
FP32 `[16,N,N]`, where N is 800 or 1200, without globally rewriting softmax.
Four native and four Python tests pass, covering both grids, extreme finite
scores, byte-identical TorchMUSA/native results, stream ordering, disjoint
offset storage and overlap rejection. The standalone runner does not load
Torch or Python libraries. `vision-block0-softmax-v1`, combined with Welford
norms, matches all first 14 checkpoints byte-for-byte through the first MLP
linear. The next same-input discrepancy is tanh-approximate GELU, maximum
**2.3842e-7**; block output maximum falls to **9.5367e-7**. This remains an
opt-in diagnostic, not full-model accuracy acceptance.

The first full-vision softmax candidate deliberately remains recorded as a
failure: Inductor padded a producer row from 1200 to 1216 elements, violating
the contiguous-only shim contract. Lowering now freezes an exact contiguous
input layout with `ExternKernel.require_contiguous`. A scaled-input producer
regression reproduces the old native API failure and passes with the fix,
while standalone direct-input cases remain byte-identical. The expanded
four-test softmax suite passes; the old full-vision export is not overwritten.

`vision-stages-softmax-v2` now executes all 24 blocks and merger. All 30
checkpoints pass the unchanged `atol=rtol=0.001` component gate. Block 23's
maximum falls to **0.266235** with no out-of-gate values; its large absolute
maximum is permitted only by the relative term. Final feature maximum is
**1.4639e-4**, mean **4.7978e-7**: the maximum is not monotonically better than
the norm-only candidate. `vision-softmax-audit-v2.json` verifies all 298 raw
vision tensors and 20 protected files. A full-logit single-image comparison
must still be rerun; this component gate is not end-to-end acceptance.
On the softmax runtime, all 88 selected Python regressions and 39 native
matrix/norm/softmax tests pass without skips. The first standalone single-image
candidate run uses the original prompt and candidate native features, with
expected IDs disabled so native greedy feedback remains independent.

That single-image run is complete (`native-single-image-softmax-v2`): all 428
input positions remain aligned with the frozen independent HF trajectory and
all 95 generated IDs match through EOS. Full-logit maximum decreases from
**0.0232811** to **0.0102887**, and out-of-gate elements from **96649** to
**21112**. Every violation remains in the 334-token prompt; decode has none.
Thus the original image/text gate still fails despite improved errors and
identical IDs. This does not validate multi-image accuracy, fair performance,
4096-position execution or a relocated deployment. Next isolate the already
observed same-input GELU discrepancy, then rerun the complete-model gate.
`single-image-softmax-audit-v2.json` independently checks all 106280960 logits,
input-prefix alignment, greedy feedback, hashes, 20 protected files and the
88 Python/39 native regressions. Its explicit status is
`audited_accuracy_failure_with_improvement`.

The next opt-in `VISION_GELU` uses muDNN C++ `GELU_TANH` and `GELU` for the
original FP32 vision MLP/merger shapes `[N,4096]`, N=200/300/800/1200. It fixes
input layout at lowering and checks mode, shape, dtype, device, storage bounds
and output overlap. Four native and four Python tests pass, including both
approximations, all original row counts, small/large signed inputs and a
producer-before-GELU path. Original BF16/F32 weights are not changed.

Combined with Welford and accurate softmax, `vision-block0-gelu-v1` matches
all 17 checkpoints byte-for-byte. `vision-stages-gelu-v1` matches all 30 full
vision checkpoints, including every block and the final 768000 features, for
the original shapes image. The independent audit verifies 298 original
tensor payloads and 23 protected files. This exactness does not yet cover
the other two original images or a non-instrumented production export.

`native-single-image-gelu-v1` completes 428 aligned positions with independent
native greedy feedback and the same 95 generated IDs. Against the frozen HF
batched-prefill reference, maximum logit error is **0.00436163**, with **754**
out-of-gate elements, all in the prompt (previously 0.0102887 and 21112).
Thus batched-reference accuracy still fails. A complete stepwise HF reference
using its own original visual features is needed to separate native execution
error from batch/chunk-versus-recurrent arithmetic; do not relax the gate or
discard the preserved batched failure.
The remaining batched violations occur only at positions 123 (750 elements)
and 124 (four). All 92 selected Python regressions and 43 native
matrix/norm/softmax/GELU tests pass without skips on the isolated GELU runtime.
The complete independent stepwise reference is launched separately as
`hf-single-image-gelu-step-v1`; it compares both native-versus-step and
HF-step-versus-HF-batch, retaining full logits and independent greedy feedback.
`single-image-gelu-audit-v1.json` independently rechecks all 106280960 logits,
input alignment, greedy tokens, protected hashes and regression counts; its
status remains `audited_accuracy_failure_with_improvement` for the batch gate.

### Native prepared-fixture controller

The MUSA `executor_runner` has an opt-in C++ greedy loop for this five-input
4B step. Tokenization and image preprocessing happen before the loop; the
fixture carries native vision features, not raw images. Six little-endian files
share a prefix: `.tokens.bin` (int64 prompt), `.positions.bin` (three axis-major
int64 prompt coordinates), `.image_rows.bin` (int64, -1 for text),
`.embeddings.bin` (FP32 rows of 2560), `.eos.bin` (int64 stop IDs), and
`.expected.bin` (int64 generated IDs, empty when no expected sequence is given).
Image rows must be sequential and correspond exactly to image placeholders.

```bash
executor_runner --musa_qwen35_fixture=/path/single_image \
  --musa_qwen35_validate_only
executor_runner --model_path=/path/qwen35_4b_multimodal_step.pte \
  --data_path=/path/aoti_musa_blob.ptd \
  --musa_qwen35_fixture=/path/single_image \
  --musa_qwen35_output=/path/native.json \
  --musa_qwen35_max_new_tokens=160 --musa_qwen35_warmup_runs=1 \
  --musa_qwen35_runs=3 --musa_cache_bytes=536870912
```

Each generation resets through physical position zero and uses the GPU argmax
as feedback. A host full-logit scan checks finiteness and argmax; it does not
replace GPU token selection. Expected IDs, when supplied, are checked online
but are never fed back. EOS, token limit and physical context limit stop the
loop. Input dtype/shape, token ranges, finite image features and prompt/decode
MRoPE bounds are checked. The six files are bounded before allocation.

JSON separates serial-token prompt processing from subsequent decode steps;
the first generated token belongs to prompt processing. Timings synchronize
the device and include host input preparation and output validation. They
exclude model loading, tokenization and vision encoding. Optional
`--musa_qwen35_dump_prefix=/path/logits` saves every full-logit output for
validation, outside per-step timing but inside total loop wall time; disable
it for benchmarking. Warmup runs remain explicitly labeled in the report.
This is not batched prefill or an integrated image/tokenizer application.

The isolated `build-native-generation-v1` compiles the new runner and passes
nine host-only C++ fixture tests. Five prepared fixtures (including a 4096-token
prompt) pass native input validation; five malformed-file/NaN/range cases are
rejected. Fixture validation does not execute the model, so that 4096-token
case is not yet a runtime boundary acceptance.

`native-generation-audit-v2` verifies 244 byte-exact outputs against the frozen
runner: two six-step text/image interface runs and two complete 116-step text
generations. Online expected-token checking rejects a deliberately changed
expected ID, and the subsequent unchanged positive control succeeds. The first
validation harness had already completed all 244 positive checks, then failed
by requiring an assertion message from a build with logging disabled. That
failure remains intact; v2 reaudits the saved positives and runs the paired
negative/positive controls without depending on disabled logging. Full original
image-question controller runs, real context-boundary execution, relocated
deployment and repeated performance measurements remain separate gates.

```bash
python -m examples.musa.aoti_gguf_inventory \
  --models /path/Qwen3.5-4B-Q4_K_M.gguf /path/mmproj-Qwen3.5-4B-BF16.gguf \
  --output /tmp/qwen35-4b-inventory.json
python -m unittest backends.musa.tests.test_musa_gguf -v
EXECUTORCH_MUSA_GGUF_RUNNER=/path/to/new/build/executor_runner \
  python -m unittest backends.musa.tests.test_musa_gguf_packed -v
```

Native tests require `EXECUTORCH_MUSA_RUNNER` and its compatible library path;
GPU/runner skips are not native acceptance. These tests do not use CPU model
inference, but do use the independent GGUF block decoder as a file-format oracle.
Packed tests require `EXECUTORCH_MUSA_GGUF_RUNNER` and the newly built shim in
`LD_LIBRARY_PATH`; an older 0.8B deployment package does not contain these symbols.

## Qwen3.5-0.8B text bring-up

**Runtime-only BF16 deployment (2026-10-29):** the standard source build and
context-256 export now pass independent packaging and relocation to a path with
Chinese characters and spaces. Two new qualitative prompts produce 512 fresh
HF/MUSA BF16 steps, all byte-identical to fused valid-length ET eager. The moved
package passes 1024 native calls on these inputs plus 648 frozen regression
calls: all 1672 complete vocabulary logits match byte-for-byte, with 836 reset
checks and two EOS stops. Both new prompts hit the context limit, not EOS.
One perturbed BF16 value still passes the old tolerance but fails the exact gate.
This does not claim arbitrary-input bitwise equivalence or answer quality.

The package includes hash-checked export metadata and an explicit 512 MiB cache
default. Actual load logs show package-local ExecuTorch libraries and no
Torch/Python dependencies; native `.text` matches the standard build. Twenty-one
packaging/reference tests and ten CI-planning tests pass. The independent audit
checks 17093 files, including rehashed evidence for the earlier 5872 source-only
comparisons; those earlier GPU comparisons were not rerun in this round.

On the final context-256 graph, six alternating runs against the correct
packed-valid v4 shim retain all 1512 inline logit/token checks and improve decode
median from 16.156 to 16.616 tok/s (+2.85%). Serial single-token prompt median is
16.521 tok/s, not batched prefill. The separate same-graph context-64 comparison
of optimized isolated v5 versus source integration is 16.696 to 16.624 tok/s
(-0.43%); source integration alone is not a speedup. Both are shared-GPU,
unlocked-clock measurements. Aggregate method-host time accounts for 95.61% of
final decode wall time but includes GPU waits, not just kernel execution.
Shared profile job windows do not support exclusive per-kernel percentages.
See workspace `QWEN35_BF16_DEPLOYMENT.md`, `deployed-bf16-audit-v1.json` and
`bf16-default-promotion-v1/result.json` for acceptance and default-entrypoint
verification. Historical milestones below retain their original scope.

**Source-integrated reductions (`a5720d982c`, 2026-10-29):** the scoped
BF16 query-normalization sum and FP32 sum/mean paths now live in MusaBackend
and the standard native shim library. BF16 covers `[1,1,16,128]` last-axis sum;
FP32 covers last-axis widths 128/256/1024 and `[1,16,128,128]` axis-2 sum.
Inputs must be nonempty, static, rank 1–8; unsupported shapes, promoted dtypes,
multi-axis reductions and CPU graphs retain their previous lowering. Native
validation checks storage bounds, physical aliases, device/dtype and exact
output layout, while allowing disjoint shared-pool views.

Direct `examples.musa.aoti_qwen35_export` exports with `--dtype bf16`,
`--valid-length-attention`, `--fuse-projections` and `--emit-greedy-token`
now need no isolated export passes or hand-linked probe objects. Fresh standard
CMake binaries and context-64/128/256 v2 packages pass 5872/5872 full-vocabulary
native/HF byte comparisons, 2936 reset checks and 20 EOS stops across seven
suites. This repeats 2936 frozen reference positions twice, not 5872 new
independent prompts; only the legacy suite uses teacher forcing. The former
Chinese counterexample passes. HF/fused-eager references are rehashed, not
regenerated. All measured logit errors are zero without relaxed tolerances.

Fifty-six Python tests and 168 same-input native/TorchMUSA reduction checks
pass. Each of three C++ rounds passes 61 tests with three explicit dual-device
skips. The single-BF16-value negative control still fails the strict gate despite
passing the old tolerance. Independent auditing verifies 14287 files, full
saved logits, token feedback, resets/EOS, native-library isolation and zero
exit cache/live counters. See workspace `QWEN35_REDUCTION_INTEGRATION.md` and
`precision-source-reduction-audit-v2.json` for evidence.

Use an ASCII, space-free `TORCHINDUCTOR_CACHE_DIR` with this installed compiler.
The first source-only export used a Chinese cache path and failed at load time:
TorchInductor's Unicode symbol-name regex did not match GNU ld's byte-wise
binary-path mangling. That failed v1 package ran zero inference steps and remains
preserved. The accepted v2 packages use fresh ASCII caches; their model/results
paths remain Chinese. No system PyTorch files were modified. Arbitrary cache
paths were not accepted at this milestone; independent BF16 packaging/relocation
and default-entrypoint promotion were still pending then. This reduction-only
round makes no new performance claim. Historical
milestones below retain their original validation scope.

**Valid-length attention implementation (2026-10-29, experimental):**
`--valid-length-attention` opts into an exportable opaque MUSA operator for
BF16, batch-one, single-token Qwen attention with context 64/128/256. Runtime
position and the causal mask select the valid KV prefix before QK/PV;
the BF16 QK, separate scaling, FP32 accurate softmax and BF16 PV boundaries
remain explicit. Native execution validates storage bounds, output aliasing,
device, geometry and causal-prefix controls. It reads only position/mask on
the host; model tensor arithmetic stays on MUSA. Each attention call currently
synchronizes the stream for those controls, so this is not a graph-capture
or performance acceptance. The option defaults off. At this earlier milestone,
full-model integration still needed isolated BF16-sum and FP32-reduction passes; standalone
operator support alone does not make the full Qwen export source-only.
Fresh source-path eager replay now matches all 836 frozen HF rows. Three newly
exported context-64/128/256 packages pass 5872/5872 full-vocabulary BF16 native/HF
byte comparisons, 2936 reset checks and 20 EOS stops across seven suites. Except
for the legacy fixed trajectory, these use native greedy feedback. The fresh
four prompts alone contribute 1672 native calls, including the former Chinese
counterexample. The 2100 older fused-eager files are rehashed, not rerun.
Fifty Python regressions, three native attention rounds (8 pass and one explicit
dual-device skip each), and 62 same-input operator comparisons pass. A real
runner negative control with one perturbed BF16 reference value is rejected
despite satisfying the older tolerance. Independent auditing checks 12795
files, full logits, tokens, runtime isolation and zero exit cache/live counters.
These are finite-workload results, not arbitrary-input bitwise guarantees;
no default deployment is promoted. See workspace `QWEN35_VALID_ATTENTION.md`
and `precision-valid-attention-audit-v1.json` for the frozen evidence.

The subsequent same-runtime context-64 paired benchmark preserves all 1512
full logits/token checks, but decode median falls from 17.976 to 16.236 tok/s
(-9.68%); all three pairs regress. This is a shared-GPU, nonblocking file-server
measurement, not batched prefill, exclusive GPU time or a speedup acceptance.
Control synchronization, KV packing and staged execution still need profiling;
the method-host timing increase alone cannot identify their individual costs.
The padded baseline is correct only for that benchmark fixture, not the Chinese
counterexample. See `valid-attention-performance-v1.json` for both paths.

**Attention overhead optimization (`c776a6b068`, `de304968fa`):** native KV
descriptors now retain cache strides rather than explicitly packing the prefix.
One pinned control buffer lets position/mask copies share a single stream wait.
Profiling showed the previous pageable-copy wrapper added two hidden waits per
layer: total method stream-sync calls fall from 20 to 8, including two output
readbacks; explicit KV `Memcpy2DAsync` calls fall from 12 to zero. Kernel launch
counts do not fall, and overlapping job labels cannot establish kernel-time
percentages. This is a correctness-preserving control/layout change, not a
return to padded attention or a relaxation of the comparison gate.

The rebuilt v5 runtime reuses the same PTE/PTD and again passes all 5872 complete
BF16 logit comparisons, 2936 reset checks, 20 EOS stops, 50 Python regressions,
62 direct operator comparisons and the exact-gate negative control. The expanded
native attention suite passes nine tests with one explicit dual-device skip in
each of three rounds, including 36 offset-KV configurations per round. Frozen
HF/eager files are rechecked, not regenerated. A same-graph/runner paired A/B
against the correct packed-valid v4 runtime improves decode median from 16.005
to 16.670 tok/s (+4.16%); all three pairs improve, with all 1512 inline logit/token
checks preserved. Shared desktop GPU conditions, single fixture and serial
prompt inputs still limit this result. Remaining reductions/default deployment
are not promoted. See workspace `QWEN35_ATTENTION_OPTIMIZATION.md` and
`attention-optimization-validation-manifest-v1.json` for final evidence.

**Original padded-attention counterexample (2026-10-29, before this fix):**
Four new qualitative prompts produce 836 independent HF BF16 reference steps;
fused ET eager matches only 669 byte-for-byte. The Chinese causal-evidence prompt
first differs at zero-based position 89, with a worst full-logit error of 0.40625
over its trajectory. Native execution matches ET eager for all 196 completed
calls, but its greedy token differs from HF at position 195; the comparator
correctly stops before consuming a different input at position 196. The remaining
native fixtures and reset/EOS checks are not completed by that interrupted run.
Unfused eager reproduces the same discrepancy. At layer 11, identical Q/K/V
produce one different BF16 QK score (134 versus 133) when the key length changes
from its valid prefix of 90 to a padded 256. Same-input length probes reproduce
the difference, while using the valid prefix restores that attention checkpoint.
An isolated eager-only valid-prefix experiment restores all 256 full-logit rows
of the Chinese case to byte equality. It reads `input_pos.item()` on the host,
so this is not an exportable or native deployment fix, nor a rerun of the other
three prompts under that experimental path.
The older 4200-call results below remain valid for their frozen inputs, not an
unrestricted input guarantee. The opt-in valid-length path above now provides
an exported/native fix for these fixtures; the default padded path has not
been promoted as accurate for them.

Generate fresh, independent HF/MUSA BF16 greedy references instead of replaying
an existing trajectory:

```bash
MUSA_LAUNCH_BLOCKING=1 "$PYTHON" -m examples.musa.aoti_qwen35_generate_reference \
  --checkpoint-dir /path/to/Qwen3.5-0.8B \
  --prompts examples/musa/qwen35_precision_prompts.json \
  --context-length 256 --fuse-projections --valid-length-attention \
  --output-dir /tmp/qwen35-fresh-reference
```

The JSON input is a nonempty list of unique `name` and nonempty `prompt` pairs.
HF generates until EOS or the context limit; ET eager replays those exact inputs
and checks every BF16 vocabulary value, optionally with source projection fusion.
`--valid-length-attention` selects the matching causal-prefix eager path and
records it in the manifest; omitting it retains the historical padded default.
The manifest records checkpoint/prompt hashes, FP32 RoPE buffer dtypes, stopping
reasons, token IDs, and eager byte-equality counts. Generation refuses to overwrite
an output directory and exits unsuccessfully if eager differs, while retaining
the reference and diagnostics. Use the existing comparison tool with
`--free-running --gpu-greedy --require-byte-equal`, matching export metadata,
and two or more repeats to validate native feedback and reset behavior. These
are single-sequence, token-by-token inputs, not batched prefill or quality scores.

**Native unary integration (2026-10-29, `11e0b6ab59`):** FP32 MUSA exp and
default softplus now use source-registered custom ops, decompositions, lowering,
and muDNN out shims. Other dtypes/softplus parameters retain refs decomposition;
promoted FP32 exp subexpressions can use the native path while final dtype is
preserved. Unary execution now checks physical storage spans and aliases and
guards the calling device before allocation/copies/handle creation. Outputs
must be contiguous and disjoint from inputs; transpose, offset, broadcast,
scalar, and empty inputs are covered. This does not change BF16 model weights
or logits into an FP32 model.

All 100 Python regressions pass, including dynamic standalone runner and
relocated-package checks. Each of three fresh native rounds passes 11 unary
and 30 matrix tests; the additional two-device test is explicitly skipped on
the single-device host. A same-input FP32 shim probe matches all 48 outputs
byte-for-byte, including special values and threshold/underflow boundaries.

Three fresh source-unary context-64/128/256 packages retain 4200/4200 complete
BF16 native/HF byte matches, 2100 reset checks, and 18 EOS stops. The independent
`precision-unary-integration-audit-v1.json` rehashes saved outputs, source and
package metadata, GPU tokens, runtime libraries, and drained caches. The old
unary registration/decomposition wrapper is no longer used. SDPA rounding,
reductions, accurate softmax, and QK-view candidates still require integration;
no default package or new performance result is promoted. See the workspace
report `QWEN35_UNARY_INTEGRATION.md`.

**Matrix integration (2026-10-29, `1dbca8613f`):** the BF16 contiguous
single-row MM/addmm muDNN path is now in source, with storage-span and output
alias checks and explicit empty-product handling. Output/multiplicand overlap
is rejected, including independently wrapped storage; disjoint views and exact
matrix-bias updates remain supported. Bounds checks trust declared Storage
capacity and conservatively include padding, not arbitrary pointer validity.
Thirty native matrix tests pass in three fresh processes. Same-input TorchMUSA
BF16 probes pass 72/72 MM and 240/240 addmm cases byte-for-byte; the old MM
dispatch fails 27/72, with maximum error 0.5. This old operator error is not
the current model error.

Replacing only the matrix object in the accepted isolated runtime retains
4200/4200 full-logit native/HF byte matches across contexts 64/128/256, all
2100 reset checks, and 18 explicit EOS stops. The independent
`precision-matrix-integration-audit-v1.json` rechecks saved outputs, GPU tokens,
library isolation, hashes, and zero cache/live counters at teardown. These
are fresh calls over frozen trajectories, not new independent questions.
The 2100 prior fused-eager outputs were rehashed, not rerun; 37 supported
Python tests were rerun successfully. That matrix-only milestone preceded the
unary integration above; SDPA, reductions, softmax, and QK views remain pending.
No default deployment package
is promoted and no new performance measurement is claimed. See the workspace
report `QWEN35_MATRIX_INTEGRATION.md` for the separate standard-CMake and
combined-runtime artifacts.

The exporter now has a default-off `--fuse-projections` option for BF16,
batch-one single-token inference at context 64/128/256. It concatenates FFN
gate/up, GDN qkv/z, and GDN b/a weights; full-attention projections are unchanged.
The model-local implementation reuses GDN's shared post-projection computation,
preserves tied embedding/output weights and mutable states, and validates all
layers before replacement. Training, unsupported input shapes/layouts, biased
or non-BF16 projections, and unsupported context capacities are rejected.
Fusion is applied after checkpoint loading, not to the on-disk checkpoint.

Fresh packages from source commit `87348ccb8e` pass all 4200 strict native/HF
full-vocabulary comparisons across contexts 64/128/256; separately executed
source-fused eager matches all 2100 positions, with 2100 reset checks and 18 EOS
stops. The independent `precision-formal-fusion-audit-v1.json` rechecks saved
tensors against HF and the prototype. A new six-process nonblocking paired A/B
retains all 1512 full-logit checks: decode median is 17.620 versus 18.140 tok/s
(+2.95%), with all three pairs positive. This is a shared-GPU, context-64,
file-protocol workload with serial prompt input, not batched prefill or a
universal performance guarantee. Details are in `QWEN35_FORMAL_FUSION.md`.

All 32 MUSA Qwen tests and five existing attention regressions pass. A broader
38-test run cannot execute one portable fused-op test because this environment
lacks `executorch.extension.pybindings._C`; its import failure is retained,
not reported as a passing or intentionally skipped numerical test.

This projection integration does not complete all native math/layout paths.
Other newly compiled packages still require
their own validation; the prototype measurements below are historical evidence,
not a substitute for source-package acceptance. The seven new fusion tests cover
state/reset and parameter identity, exported buffer mutation, nonzero offsets, input/layout and
weight rejection, preflight non-mutation, and the default-off CLI.

**Projection-fusion candidate validation (2026-10-29):** an isolated prototype
combines FFN gate/up and the separate GDN qkv/z and b/a pairs, retaining the
original full-attention projections. Three separately exported BF16 packages
pass 1780/1780 (context 64), 936/936 (128), and 1484/1484 (256) full-vocabulary
native/HF byte comparisons. All 2100 reset checks and 18 explicit EOS stops pass;
separately executed fused eager matches all 2100 reference positions. These are
repeated calls over frozen reference trajectories, not 4200 independent prompts.
The native comparisons retain all 248320 logits, GPU greedy feedback except for
the legacy fixed trajectory, and native-only library/cache teardown checks.

The projection probe rejects GDN-all and full-attention QKV/KV fusion: they
introduce BF16 differences up to 0.001953125 and 0.0078125 respectively. Matching
selected projection outputs alone is not a whole-model acceptance. The accepted
grouping is still an artifact prototype, not a default backend feature or a
batched-prefill/arbitrary-context guarantee. No deployment package is promoted.
The local `precision-fused-projections-audit-v1.json` rechecks saved native,
fused-eager, HF, and prior unfused-native tensors; details and reproduction are
in the workspace report `QWEN35_PROJECTION_FUSION.md`.

Use `aoti_qwen35_compare.py --require-byte-equal` for strict native/HF
full-vocabulary acceptance. It checks every session, including prompt and reset
calls, and exits unsuccessfully on any differing byte while preserving the
per-step diagnostics in `result.json`. Without this flag, comparison completion
is diagnostic only: matching tokens or passing the legacy tolerance is not
byte-identical acceptance. Neither mode marks the entire runtime validated.
All 25 Qwen Python tests pass. A real-runner negative control changes one BF16
reference logit bit in a separate synthetic fixture: token IDs and the old
tolerance still pass, but strict byte acceptance correctly fails and saves the
first differing session/fixture/position. Official references are not modified.

**Latest context validation (2026-10-29):** independently exported context-128
and context-256 candidates retain one MUSA delegate, all 48 mutable states, full
BF16 logits, and GPU greedy output. HF/converted eager match 468/468 and 742/742
positions. Two native rounds match 936/936 and 1484/1484 full-vocabulary outputs
byte-for-byte, with 1210 reset checks and six explicit EOS stops in total. The
256 package includes 548 calls at positions >=128. Two long prompts contain
112 and 68 tokens; some answers still end at the configured context limit.
`precision-context128-256-audit-v1.json` independently rechecks saved tensors and
hashes. The metadata guard has 24 passing Qwen Python tests; its CLI rejects a
missing or mismatched context declaration before launching the runner. Three
native softmax tests pass in three fresh processes, including offsets, streams,
aliases, strides, capacity bounds, and rejected unsupported geometry.
These remain isolated, single-token-input candidates, not batched-prefill,
arbitrary-context, or default-backend acceptance; no default package is promoted.

**Previous expanded isolated BF16 accuracy result (2026-10-28):** eight additional
Chinese/English prompts cover evidence, conflicts, honesty, causality, negation,
conditional reasoning, ambiguity, and mixed-language instructions. Independently
generated HF trajectories contain 477 input steps; converted eager matches all
477 full-vocabulary outputs. Two native GPU-greedy rounds match all 954 outputs
byte-for-byte, all GPU tokens agree, and all 477 reset checks pass. Three fixtures
reach EOS (46/62/49 steps); five stop at context=64 and are not complete answers.
Together with the previous 566 comparisons and 260 EOS comparisons, the same
dual-output package passes 1780 full-vocabulary comparisons and 890 reset checks,
with 12 explicit EOS stops. Saved native/reference files and runtime/input hashes
are independently rechecked in `precision-gpu-greedy-expanded-audit-v1.json`;
the new run is `compare-gpu-greedy-expanded-v1/result.json`. Maximum/mean logit
error and distribution differences are zero. All 22 Qwen Python tests pass.
These are batch-one, single-token-input, context-64 checks of isolated candidate
math/layout paths, not arbitrary-input, long-context, batched-prefill, or answer
quality acceptance. The default deployment package remains unchanged.

**Previous isolated BF16 accuracy result:** all 91 legacy
teacher-forced steps and 192 new native-greedy-feedback steps, each repeated
twice, have byte-identical full-vocabulary logits against correctly loaded
Transformers 5.3.0 on MUSA. All 566 comparisons have zero maximum/mean error,
all generated token IDs match, and all 283 reset checks are byte-identical.
The native library audit passes without Torch/Python runtime dependencies.
The three new generation fixtures stop at the 64-position context limit,
not EOS. This uses combined experimental matrix, attention, math, reduction,
and layout paths; it is not default-backend, arbitrary-prompt, long-context,
or performance acceptance. The default model package is unchanged. Reports:
`compare-bf16-sum-{generation,legacy}-v1/result.json` and the hash-tracked
`precision-bf16-sum-v1.json` under the local Qwen artifact directory.

**Earlier expanded validation failure:** three new prompts run HF greedy
generation to the 64-position context limit. Converted eager/HF logits match
152/192 steps; two fixtures develop tolerance failures after positions 43/45.
Native greedy feedback also diverges on the English fixture: prediction at
position 50 is 1785 rather than 1852, so comparison stops before position 51
would use different inputs. The earlier 91-step result is not broad acceptance.
Opt-in host timing and faster BF16 logit decoding are available independently
of this numerical investigation; no default model package is promoted.

The converted eager discrepancy is now fixed for this suite. At trust position
43, layer 7, identical Q/K/V produce eight different BF16 context elements with
MUSA SDPA versus HF's explicit attention; cropping the cache does not fix it.
Qwen BF16 loading now installs a model-local `Qwen35MusaSDPA` that preserves
BF16 QK/scale/mask and probability rounding with FP32 softmax. It does not
monkey-patch global SDPA or change FP32 loading. All 283 old and new reference
steps then match unchanged official HF logits byte-for-byte; 20 Python tests
pass. The earlier native package diverges at trust position 39 in layer 11,
so the eager fix alone does not establish native acceptance.

The native discrepancy was isolated to FP32 softmax: identical QK/scale/mask
scores produced a difference of at most 5.96e-8 that crossed a BF16 probability
rounding boundary. An isolated muDNN accurate-softmax replacement makes all
64 trust steps exact. Fairness independently diverged at position 42, layer 2:
one of 16 BF16 Query L2 sums was 2.703125 instead of 2.71875. An identical-input
five-stage probe reproduces the whole-layer discrepancy. Matching the muDNN
BF16 sum path makes all 1032 layer checkpoints exact, then the full-model
566-comparison suite above passes. This aligns the specified BF16 reference;
it is not evidence that every reference reduction is closer to an exact sum.
The new sum replacement is restricted to BF16 [1,1,16,128] last-axis sums
(36 nodes), combined with six softmax, six QK-view, and 115 FP32 Reduce nodes.
These remain isolated candidates pending shape/stride/alias safety coverage.

Export a local Hugging Face checkpoint with the existing ExecuTorch Qwen3.5
weight converter and a single MUSA delegate:

```bash
python -m examples.musa.aoti_qwen35_export \
  --checkpoint-dir /path/to/Qwen3.5-0.8B \
  --output-dir /tmp/qwen35-musa-fp32 \
  --dtype fp32 --context-length 64 --token-ids 9707 11 2082

python examples/musa/aoti_qwen35_validate.py \
  --artifacts /tmp/qwen35-musa-fp32 \
  --runner /path/to/executor_runner --output-dir /tmp/qwen35-validation
```

Each output directory must not exist. The exporter writes `qwen35.pte`,
`aoti_musa_blob.ptd`, three MUSA eager reference logits, the exported graph,
and a hash-bearing `result.json`. It checks all 48 mutable buffers (18 GDN
convolution/recurrent pairs and six attention K/V pairs) and rejects host
tensor operators outside the delegate. GDN uses an ATen recurrence on MUSA,
not the portable fused CPU operator. Recurrent state remains FP32 with BF16
weights. Only text, batch one, static single-token decoding is covered;
vision, MTP, quantization, and a production generation API are not included.

On the local M1000, **FP32 native decoding passes the three-token fixture**
twice in one process, including a position-zero reset with byte-identical
logits. Maximum error against MUSA eager is `1.25e-4`; all token IDs match.
Eager FP32 differs from Transformers 5.3.0 MUSA by at most `1.41e-5` on that
fixture. The native loader audit observes no Torch/Python runtime libraries.
This is a correctness PoC, not a performance or broad quality claim.

The chat fixture `What is 2 + 3? Answer briefly.` also passes: 23 prompt
tokens followed by native greedy output `[20, 248046]` (`5`, then EOS),
24 calls repeated after reset for 48 calls total. Every logit passes and
every argmax matches; maximum native/eager error is `8.70e-5`.
Transformers MUSA FP32 matches the entire reference token trajectory with
maximum logit error `2.25e-5`. This single short answer is not a general
language-quality evaluation.

**The original BF16 package exports but fails native numerical acceptance.** The corrected
unit-offset RMSNorm gives byte-identical eager logits to Transformers for the
three reference tokens. The native first-token argmax matches, but its logits
fail `atol=0.04, rtol=0.04`. Do not use argmax alone to accept this path.
The exporter always records `status=exported`, `runtime_validated=false`;
only a separate successful validator report establishes runtime acceptance.

Two independent rounding differences have been isolated on the local MUSA
stack. Eager low-precision SiLU rounds the sigmoid before multiplication,
whereas the default decomposition performs both in FP32. The MUSA
decomposition now preserves the intermediate rounding. Separately, Edge
dim-order copies can lose an explicit FP32-to-BF16-to-FP32 rounding boundary
inside AOTInductor, even with `emulate_precision_casts=True`. The MUSA cast
decomposition uses a compiler-only precision operator with an explicit Triton
storage-dtype conversion; it does not add a Torch runtime dependency.

Native runner regressions cover SiLU in FP32/FP16/BF16 and explicit cast
roundtrips in FP16/BF16. The latter failed on 1022/1024 values before the fix
and now requires exact equality. Layout preservation is tested separately.
These local fixes are not sufficient evidence for whole-model acceptance;
keep the unchanged full-vocabulary and reset checks as the deployment gate.

An isolated follow-up combines these fixes with muDNN for contiguous
single-row BF16 matrix inputs and BF16 score/probability rounding for the
256-dimensional attention heads. It passes the original three-token fixture
twice: positions zero and one are byte-exact against eager; position two has
maximum error `0.0625`, with no values outside the unchanged combined
absolute/relative tolerance. Argmax IDs, byte-identical reset, and the native
library audit pass. The matrix and attention experiments are not backend
defaults: broader shape/stride/bias coverage and longer prompt/decode
validation are still required. The original BF16 package remains a negative
fixture, not a newly accepted package.

A longer isolated BF16 comparison against Transformers 5.3.0 on MUSA covers
two Chinese prompts, 27 and 31 prompt tokens, each followed by 16 greedy
predictions. Including the original fixture, 91 forward calls repeat twice
with byte-identical reset outputs and a clean native-library audit. Precision
still fails: native/HF maximum logit errors are `0.53125` and `0.640625` for the
Chinese fixtures. One prompt-position argmax differs; all 32 generated-token
predictions match under teacher forcing. The converted PyTorch eager model
is byte-identical to correctly loaded official HF across all 91 steps. An
earlier reference script incorrectly cast HF's RoPE inverse-frequency buffers
to BF16; those comparisons must not be used to diagnose a frontend mismatch.
Use `from_pretrained(dtype=torch.bfloat16).to(device="musa")`, preserving FP32
RoPE buffers, rather than casting the entire constructed model to BF16. This
is not a successful long-context or free-running generation acceptance.

A further isolated muDNN FP32 exp/default-softplus experiment makes all three
original-token logits byte-identical to the corrected HF reference. It does
not resolve the Chinese fixtures: maximum errors become `0.8125` and
`0.609375`, and prompt-position argmax mismatches increase from one to two.
Although mean errors decrease slightly, this is not whole-model acceptance.
The experimental C++ changes are not backend defaults; remaining recurrent
state/reduction differences need independent isolation and validation.

The next isolated FP32 reduction candidate identifies a separate first-token
failure in layer 18 (zero-based): identical residual input produces a one-ULP
mean-square difference in FFN RMSNorm, crossing a BF16 rounding boundary.
An explicit-axis muDNN C++ Reduce shim aligns all 24 inspected stages of that
block. It also aligns all inspected stages and FP32 recurrent states of layer
22 over the original three-token sequence. Neither experiment changes the
eager model or reference tensors.

On the same 91-step trajectories, repeated twice, this candidate increases
byte-identical steps from 3 to 31. Chinese-fixture maximum errors decrease
from `0.8125`/`0.609375` to `0.375`/`0.25`, and mean errors from
`0.03249442`/`0.03436223` to `0.01690326`/`0.01652340`. However, 60 of 91 steps
still violate the original tolerance. One fridge prompt-position argmax now
differs at position 21; the evidence fixture has no argmax mismatches, and
all 32 generated-token predictions still match under teacher forcing. Reset
outputs are byte-identical and the native-library audit passes. This is an
isolated accuracy experiment, not a default-backend promotion, free-running
generation acceptance, or performance result.

The reduction candidate's first evidence-fixture divergence is at position 7,
layer 19 (zero-based). An eight-step trace matches uninstrumented logits.
At that position, Q/K/V, gates, and KV caches agree exactly; SDPA context has
one differing BF16 element out of 2048, with maximum error `0.0009765625`.
The earlier seven steps match at every inspected block stage. Subsequent
same-input isolation identifies a PV layout issue, not a cache or softmax bug:
the `[8, 1, 64]` left operand has an unused zero row stride, which selected
muBLAS instead of the positive-stride muDNN path. Normalizing the singleton
descriptor stride and preserving muDNN dispatch fixes the reproduced BF16
difference. A regression fails on the old library; all 20 matrix tests pass
with the narrow fix. The same full-model PTE/PTD then gives 90/91 byte-exact
steps and all 91 steps within the unchanged tolerance, with matching argmax.

The last differing step, evidence position 11, starts at QK in layer 23.
Edge materializes a transposed key with `transpose_copy`, selecting muDNN NN
instead of the eager NT view path. Same-input native probes reproduce each
path's exact scores. An isolated pass preserves six BF16 QK transpose views,
restricted to `[1, 8, 64, 256]` inputs and matrix-RHS-only consumers through
view/reshape chains. All 180 inspected layer checkpoints then match eager.
With this pass and the combined runtime, the original, fridge, and evidence
fixtures give 3/3, 42/42, and 46/46 byte-exact steps in each of two rounds.
The original tolerance is unchanged; full-vocabulary maximum/mean errors,
total variation, and argmax mismatches are all zero. Reports are named
`compare-bf16-stride-v1/result.json` and `compare-bf16-qk-view-v1/result.json`.
The comparison remains `runtime_validated=false`: broader shapes, aliasing,
mutation safety, unseen prompts, and independent free generation are not
validated. The QK pass and other experimental paths are not backend defaults.

For longer BF16 diagnostics against official Transformers and converted
ExecuTorch eager references, use the separate teacher-forced comparator:

```bash
python -m examples.musa.aoti_qwen35_compare \
  --references /path/to/references/manifest.json \
  --pte /path/to/qwen35.pte --ptd /path/to/aoti_musa_blob.ptd \
  --runner /path/to/executor_runner --output-dir /tmp/qwen35-comparison
```

Add `--free-running` when the references contain an independently generated HF
greedy trajectory. After the prompt, the comparator feeds back native argmax,
not the next reference token. If trajectories diverge, it stops before comparing
logits for different inputs and records a failed report. Numerical tolerance
and generated-token agreement are separate report fields; neither is inferred
from the other. `--runtime-preload` explicitly opts into a hash-tracked isolated
runtime library experiment; it is not inherited from the controller environment.
References that end at EOS must declare the tokenizer's `eos_token_id` in the
manifest and `stop_reason: "eos"` in the fixture. In free-running mode the
comparator stops explicitly on the native EOS prediction, ignoring prompt-only
predictions. Early EOS, missing final EOS, or EOS at a context-limited reference
endpoint fails validation. The report counts actual `native_eos_stops` separately
from context-limited trajectories. Three additional context-64 fixtures reached
EOS at 55, 53, and 22 input steps: both native rounds match all 248320 BF16
logits byte-for-byte, GPU tokens agree, all 130 reset checks pass, and all six
sequences explicitly stop on EOS. One fixture is a short Chinese stop-control
case, not a complex reasoning evaluation.
Use repeatable `--fixture NAME` arguments to isolate a failing trajectory or
measure later fixtures after an earlier free-running divergence. Selection
preserves manifest order and reference hashes, rejects unknown or duplicate
names, and records `selected_fixtures` in the report. An isolated result does
not establish that the complete reference suite passes.

For a context other than the legacy 64 positions, pass
`--export-metadata /path/to/export/result.json`. The comparator requires the
reference context to match the exporter context and verifies the actual PTE/PTD
hashes, checkpoint hashes, BF16 dtype, single delegate, and output indices before
starting the runner. It tracks the metadata hash during comparison and bounds
every fixture by that verified context. Merely increasing a reference manifest's
context does not make a 64-position package safe for longer input. This option
also supports the 64-position package and does not by itself qualify new backend
shapes or claim long-context accuracy.

For single-row Qwen greedy output, export with `--emit-greedy-token`. Output 0
remains the full logits tensor; output 1 is a one-element int64 tensor computed
by GPU argmax inside the same MUSA delegate. It selects the first maximum on
ties and does not use MTP, quantization, or CPU tensor operators in the graph.
The exporter records both output indices and still requires all 48 mutable
states and exactly one delegate. Omitting the option preserves the logits-only
interface.

Use the matching comparator option `--gpu-greedy` to read output 1, require an
exactly eight-byte in-vocabulary token, check it against the full-logit argmax,
and feed it back in `--free-running` mode. Full-vocabulary numerical comparison,
reference hashes, finite-value checks, and reset checks remain enabled; matching
only the token does not pass precision validation. The optimized benchmark can
read the GPU token without Python float expansion while still checking every
logit byte against the preverified HF reference outside the timed sampling path.

For file-server timing, `executor_runner --server_mode --server_timing` emits
`SERVER TIMING` JSON before each completion marker. It separates input loading
and preparation, host `method.execute`, and output retrieval/writing. Method
time includes backend transfers and synchronization and is not pure GPU time;
an asynchronous backend may also finish after that host interval. No extra
device synchronization is inserted. Timings are opt-in and do not alter the
default completion protocol. GPU timelines must be measured separately.

### Opt-in native MUSA allocation cache

`executor_runner --musa_cache_bytes=536870912` enables a shared, bounded
event-based cache in `extension_cuda`; the default is zero (disabled).
It retains synchronous allocation rather than switching back to the unstable
MUSA async-allocation path. A freed block can be reused only on the same device,
at exactly the same size, after its release-stream event completes. Individual
blocks above 16 MiB and over-aligned allocations bypass retention. The byte
limit bounds retained blocks, not live model weights or total GPU memory.

Embedding applications configure it through
`CudaAllocator::configure_musa_allocation_cache` before allocation begins;
configuration changes require no tracked live or cached blocks and externally
quiescent callers. Release on the stream that owns the final use, or establish
the appropriate cross-stream dependencies first. After device work completes,
`release_cached_memory` frees ready retained blocks without freeing live ones.
OOM retries reclaim ready cached blocks once; unfinished blocks are not reused
or forcibly synchronized. The runner trims at teardown and prints a cache
summary. No GPU cleanup is performed from a static destructor.

The BF16 comparator accepts `--musa-cache-bytes 536870912`, passes it explicitly
to the runner, audits the loaded extension hash, and requires zero cached blocks
at normal teardown. `--server_timing` additionally emits per-step
`MUSA ALLOCATION CACHE SAMPLE` counters when caching is enabled. No LD_PRELOAD
interposer is needed. Graph capture, dynamic library unloading, and untested
multi-device hardware are outside this opt-in validation scope.

On the local shared M1000, the byte-exact BF16 sum candidate was measured in
six fresh processes ordered off/on/on/off/off/on. Each process used one warmup
and five measured fridge sequences (27 serial prompt calls, 15 decode calls).
All 1512 full-vocabulary outputs match HF. Across 15 measured sequences per
mode, decode median increases from 5.848 to 8.456 tokens/s (1.446x); mean host
method time falls from 125.565 to 74.871 ms. Python readback/argmax still costs
42.273 ms, approximately 35.4% of cache-on decode wall time. These are file-server
measurements, not batched prefill or pure GPU throughput, and do not replace
the independent numerical acceptance gate.

A separate 21-sequence fairness process checks 1344 outputs exactly, resetting
at position zero every 64 steps. Tracked allocations plateau at 206 after five
sequences, RSS stays constant, cached memory peaks at 6060632 bytes, and all
cache/live counters are zero at exit. This is finite-duration fixed-shape
evidence, not a claim of arbitrary long-context or leak-free service operation.

The reference manifest must have `status="complete"`, `dtype="bf16"`, and a
nonempty `fixtures` list. Each fixture contains `name`, `token_ids`,
`prompt_token_count`, and `hashes` mapping tensor filenames to SHA256 values.
Store full-vocabulary BF16 logits in `NAME/hf-POSITION.bin` and
`NAME/et-POSITION.bin`, relative to the manifest directory. Both references
must use the same checkpoint, positions, cache reset, and input token
trajectory; record their provenance and generation settings in the manifest.
Use official Transformers BF16 on MUSA for `hf`, and the unpatched converted
PyTorch model for `et`. This command does not generate references itself.

To regenerate both references on an existing fixed trajectory:

```bash
MUSA_LAUNCH_BLOCKING=1 python -m examples.musa.aoti_qwen35_reference \
  --checkpoint-dir /path/to/Qwen3.5-0.8B \
  --trajectories /path/to/previous/manifest.json \
  --output-dir /tmp/qwen35-bf16-references
```

The source manifest must additionally contain `checkpoint_hashes`; the
generator checks these against the local checkpoint and requires a new output
directory. It uses the official HF loader with strict loading diagnostics,
checks BF16 weights, tied embeddings, FP32 RoPE buffers, and MUSA placement,
and records buffer dtypes in the resulting manifest. It keeps the input token
trajectory unchanged; it does not regenerate or silently replace old evidence.

Unlike acceptance validation, this diagnostic continues past numerical or
argmax mismatches, always feeding the reference trajectory rather than native
predictions. It reports every logit's error, top-k overlap, softmax total
variation, and KL(reference || actual), separating native/HF, eager/HF, and
native/eager differences. Its `summary` aggregates first-session fixture
metrics, separates prompt/generation ranking failures, and counts reset
checks without double-counting repeated steps. A single run reports reset
agreement as null, not passed. Two repetitions check byte-identical cache reset;
invalid tensors, changed hashes, loader violations, and process failures
still fail closed. `status="comparison_complete"` means the measurement
completed, **not** that precision passed; `runtime_validated` stays false.
Teacher-forced token agreement does not establish free-running generation
agreement. The deployment validator and its thresholds remain unchanged.

The validator uses Python's standard library for control and readback only;
all model tensor computation runs in the native MUSA delegate. It checks
every logit, exact argmax IDs, two sequences in the same process, reset
repeatability, loader libraries, and artifact hashes. FP32 gates are
`atol=0.001, rtol=0.001`. It fails closed and retains diagnostics on errors.
Set `LD_LIBRARY_PATH` to the selected runtime's shared-library directories
and MUSA SDK; do not preload Torch libraries.

For a local chat template and greedy reference trajectory, replace
`--token-ids ...` with `--prompt 'What is 2 + 3? Answer briefly.'
--generate-tokens 16`. Tokenization additionally requires Transformers and
the checkpoint's tokenizer files, with network loading disabled. Native
validation feeds back the native argmax after the prompt and checks it
against the eager trajectory. Prompt ingestion is token-by-token, not
optimized batched prefill. Longer context, multiple prompts, and throughput
need separate acceptance before deployment beyond this PoC.

On the local TorchMUSA 2.9.0/MUSA 5.1.2 environment, eager reference generation
for the chat prompt encountered an asynchronous device error without launch
blocking. Prefix the **export command** with `MUSA_LAUNCH_BLOCKING=1` for
that diagnostic workflow; its root cause is not yet established. This is
not enabled in the native validator and is not a performance configuration.

Use `package_aoti.py` to package the selected runner, shared libraries and
model. To validate a relocated package against the same reference artifacts,
pass `--runner PACKAGE/bin/executor_runner --pte PACKAGE/model/model.pte
--ptd PACKAGE/model/model.ptd` to the validator and set `LD_LIBRARY_PATH` to
`PACKAGE/lib:/usr/local/musa/lib`. The PTE/PTD hashes must still match the
export report. FP32 weights occupy approximately 2.9 GiB in the PTD;
the checkpoint's original BF16 storage does not make this an accepted BF16
native inference path.
The local FP32 chat package passes the same 48-call acceptance after moving
to a directory containing spaces, loading its two ExecuTorch shared libraries
from the relocated package rather than the development build.

Mutable MUSA delegates retain tensor constants rather than inlining small
buffers, which would discard cross-call updates. The server runner emits
`SERVER MODE DONE` on stderr even when logging is disabled. A regression
checks three consecutive accumulator calls in the same process; rebuilding
the runner is required for the protocol fix.

## Native device metadata and compatibility

New `MusaPartitioner` exports default to `target_device=musa:0`. PTE tensor
metadata, memory-planning arenas, weight descriptors, and SlimTensor use a
distinct `MUSA=2` device type; CPU remains 0 and CUDA remains 1. MUSA weight
storage keys use `musa_fqn_weight:musa:`. The shared weight container magic
`ETCUDAFQN0` and the PTD flat-tensor layout are unchanged.

These serialized values are **not** PyTorch AOTI device IDs: TorchMUSA uses
`PrivateUse1=20`. The shim boundary translates MUSA metadata to/from 20;
passing serialized value 2 as an AOTI device ID is rejected. Native MUSA
SlimTensors report `is_musa()` and `is_gpu()`, but not `is_cuda()`.

A MUSA runtime also accepts existing `MusaBackend` artifacts whose tensor,
arena, and weight descriptors use the legacy CUDA tag 1. It registers both
native and legacy allocators and normalizes legacy SlimTensor device tags
to MUSA. This is compatibility with earlier **MUSA** exports, not execution
of CUDA binaries: a MUSA build registers `MusaBackend` only and rejects a
`CudaBackend` delegate. CUDA/ROCm builds retain their existing CUDA-tagged
path and do not allocate MUSA tensors. Older runtimes without the new enum
cannot run native-MUSA artifacts; rebuild the runtime together with the
exporter.

The implementation still shares the `backends/cuda` runtime, allocator,
and SlimTensor GPU storage code through MUSA API adapters. Native device
metadata does not imply that standalone packaging, full ATen-mode runtime
integration, or CUDA/ROCm hardware regression testing is complete.

## Runtime-only Linux package

New MUSA exports omit Torch/Python libraries from the AOTI link command.
The runtime supplies the MUSA device/stream-guard ABI and PrivateUse1 device
constant itself. Stream guards accept the default stream and restore nested
stream selections. Compiler changes are scoped to MUSA AOTI, not eager/JIT,
CPU, CUDA, or compile-only options.

The native MUSA matrix shims retain one muBLAS handle per host thread for
repeated operations on the same device and stream. Switching either destroys
and recreates the handle; threads never share its workspace. The handle is
also destroyed at thread exit, so workspace can outlive an individual model
call. This does not change model math, the muDNN paths, or PTE/PTD metadata.
The readback staging workaround remains necessary and is independent of
handle reuse. Correctness and performance evidence is tracked in
`backends/musa/tests/README.md`.

Export on the development machine with the normal TorchMUSA toolchain, then:

```bash
python examples/musa/package_aoti.py \
  --build-dir /tmp/executorch-musa-ci \
  --pte /tmp/model/mlp.pte --ptd /tmp/model/aoti_musa_blob.ptd \
  --output-dir /tmp/musa-package --patchelf /path/to/patchelf

mv /tmp/musa-package '/tmp/moved musa package'
MUSA_HOME=/usr/local/musa '/tmp/moved musa package/run.sh' \
  --inputs=/tmp/model/input-0.bin --input_shapes=2x16 \
  --output_file=/tmp/model/result --print_output=summary
```

The development-side packager requires the export Python environment,
`flatc`, GNU `readelf`, and `patchelf`. It copies the runner, two ExecuTorch
shared libraries, and PTE/PTD; replaces build-tree RPATHs with `$ORIGIN`
paths; checks every referenced AOTI library in PTE/PTD; and records SHA-256
hashes in `manifest.json`. It rejects Torch/Python dependencies, unexpected
external libraries, and non-relocatable search paths. Failed packaging does
not publish a partial output directory, and existing packages are not
overwritten. Do not use this tool on untrusted binaries.

Optionally pass `--export-metadata /tmp/model/result.json` to check the exported
PTE/PTD hashes and include `model/export.json`. Its artifact names/hashes match
the packaged files, and `source_export_metadata_sha256` identifies the original
metadata without modifying it. This permits the strict Qwen comparison tool to
validate a relocated package. Export-stage `runtime_validated` fields are not
rewritten into acceptance claims; retain a separate runtime validation report.
Undefined embedded `__triton_*` binary symbols are rejected before publication,
including the observed Unicode compiler-cache mangling failure.

`--musa-cache-bytes 536870912` records an optional 512 MiB allocator-cache
default in `manifest.json` and `run.sh`. The default is zero when omitted;
explicit runner arguments override the launcher default, including
`--musa_cache_bytes=0`. This is a cache limit, not a model-memory limit. Check
accuracy and exit live/cache counters with the selected runtime configuration.

The target needs a compatible Linux architecture/glibc/libstdc++, MUSA SDK,
GPU driver, GPU ISA matching the embedded kernels, GPU access, and writable
temporary storage. Cross-GPU-architecture deployment is not validated.
It does **not** need
Python, PyTorch, TorchMUSA, the source checkout, or the build directory for
the tested models. The SDK/driver are not copied or redistributed. The
shell launcher uses only the package and `$MUSA_HOME/lib` for its library
search path and clears Python/preload environment overrides. It deliberately
does not preserve a caller's custom `LD_LIBRARY_PATH`.

This packager currently targets the default static-core MUSA build with
`libaoti_cuda_shims.so` and `libextension_cuda.so`, not shared-core/wheel
layouts or arbitrary third-party operator libraries. Historical models that
still link `libtorch_cpu`/`libmusa_python` remain usable through the earlier
development environment but must be re-exported for this runtime-only path.
An audit of declared dependencies is not an accuracy check: run model-specific
acceptance after relocation, especially after changing the SDK/driver.

## Synchronized stateless-model benchmarks

Rebuild and repackage the runner before using the MUSA-only benchmark flags:

```bash
MUSA_HOME=/usr/local/musa /tmp/musa-package/run.sh \
  --inputs=/tmp/model/input-0.bin --input_shapes=2x16 \
  --musa_benchmark_output=/tmp/timing.json \
  --musa_warmup_executions=10 --num_executions=50 --print_output=none
```

This runs ten unmeasured warmups followed by fifty measured executions in
one process. `execute_ms` measures host wall-clock time around
`Method::execute()` and a final `musaDeviceSynchronize`; input preparation
is synchronized before that timer starts. `prepare_execute_ms` additionally
includes per-iteration input preparation, resizing, and its synchronization.
Neither metric includes process startup, program/method loading, initial
input-file reads, final output-file writes, or warmup samples. Transfers
inside `Method::execute()` and synchronization overhead are included.
These are not device-event kernel timings or service request latency.

The opt-in mode requires exactly one planned MUSA device, a `MusaBackend`
delegate, non-server mode, and a positive measured count. Legacy CUDA-tagged
MUSA packages are accepted; native CUDA delegates are not. The default
runner path is unchanged when `--musa_benchmark_output` is absent. Run on
an otherwise idle GPU: device-wide synchronization can wait for other work.
Use stateless models only; warmups advance mutable state and this interface
does not reset a KV cache or define an LLM prefill/decode workload.

For fresh-process repeats with frozen inputs, package integrity checks,
raw timing arrays, per-process statistics, and pooled statistics:

```bash
python3 -S examples/musa/benchmark_aoti.py \
  --package /tmp/musa-package --inputs /tmp/model/input-0.bin \
  --input-shapes 2x16 --reference-output /tmp/model/accepted-output-0.bin \
  --warmup 10 --iterations 50 --repeats 5 \
  --output-dir /tmp/musa-benchmark
```

The driver needs only Python 3.11+ standard libraries on the benchmark host;
the packaged inference process still has no Python/Torch dependency. Its
clean environment disables loader tracing during measurement. It refuses
to reuse an output directory, checks SHA-256 hashes before and after the
run, and compares each process's final tensor outputs byte-for-byte against
previously accepted runner outputs for these exact inputs. Establish MUSA
eager accuracy separately before supplying those references: repeatability
is not an independent accuracy check, and intermediate iterations are not
checked. Reports include minimum, median, mean, nearest-rank p90, and maximum
milliseconds; startup and individual process medians are not mixed into
the pooled sample distribution. No speedup is implied without a comparable
baseline using the same weights, shapes, dtypes, and measurement boundaries.

### MUSA API/kernel trace

On the validated MUSA System Profiler 1.9.0 installation, profile the packaged
binary directly, not `run.sh`: the deployment launcher intentionally clears
`LD_PRELOAD`, which can remove the profiler's injection library. For example:

```bash
PACKAGE=/tmp/musa-package
OUTPUT=/tmp/musa-profile
mkdir -p "$OUTPUT"
env -i PATH=/usr/bin:/bin HOME="$OUTPUT" LD_BIND_NOW=1 \
  LD_LIBRARY_PATH="$PACKAGE/lib:/usr/local/musa/lib" \
  /usr/bin/msys profile --trace=musa,osrt --output="$OUTPUT/model" \
  "$PACKAGE/bin/executor_runner" \
  --model_path="$PACKAGE/model/model.pte" \
  --data_path="$PACKAGE/model/model.ptd" \
  --inputs=/tmp/model/input-0.bin --input_shapes=2x16 \
  --musa_benchmark_output="$OUTPUT/timing.json" \
  --musa_warmup_executions=3 --num_executions=5 \
  --output_file="$OUTPUT/output" --print_output=none
```

Keep the profiler log, `.msys-rep`, timing JSON, and output tensors together.
Check output accuracy after capture, and do not report instrumented latency
as the unprofiled performance baseline. The inspected `.msys-rep` is SQLite:
`all_events.name` references `event_names.id`; event names are not columns
in `all_events`. Separate the measured iterations from loading, warmup and
teardown. Host API, driver, OS wait and GPU events overlap, so their durations
must not be added together as total inference time. The October 26 diagnostic
and the rejected handle-cache experiment are recorded in
`backends/musa/tests/README.md`.

## Model coverage

`aoti_model_acceptance.py` exercises the complete MUSA path for these models:

- `mlp`: Linear + ReLU + Linear + ReLU;
- `simple_cnn`: Conv2d + BatchNorm2d + ReLU + MaxPool2d + Conv2d + Flatten + Linear;
- `simple_audio_cnn`: Conv1d + BatchNorm1d + ReLU + strided Conv1d + AvgPool1d + Linear;
- `simple_video_cnn`: Conv3d + BatchNorm3d + ReLU + MaxPool3d + Conv3d + pooling + Linear;
- `simple_decoder`: ConvTranspose2d + ReLU + ConvTranspose2d with output padding;
- `simple_audio_decoder`: grouped ConvTranspose1d + ReLU + ConvTranspose1d with output padding;
- `simple_video_decoder`: ConvTranspose3d + ReLU + ConvTranspose3d with output padding;
- `dilated_audio_decoder`, `dilated_image_decoder`, `dilated_video_decoder`: ConvTranspose1d/2d/3d with output padding smaller than dilation but not stride;
- `tiny_unet`: Conv2d, GroupNorm, SiLU, down/up sampling, concatenated skip connection, and ConvTranspose2d;
- `bilinear_unet`: Conv2d, GroupNorm, SiLU, bilinear upsampling to the skip size, and concatenation;
- `transformer_block`: torchvision-independent TransformerEncoderLayer;
- `causal_attention`: causal grouped-query attention with Q/K/V projections;
- `embedding_classifier`: int64 token IDs, Embedding, pooling, and Linear;
- `token_encoder`: int64 token IDs, Embedding, TransformerEncoderLayer, and Linear;
- `token_decoder`: int64 token IDs, Embedding, causal GQA, LayerNorm, and vocabulary projection;
- `narrow_resnet18`: ResNet-18 block topology with narrow channels and a 32x32 input;
- `resnet18`: unmodified torchvision ResNet18 with random weights and a 64x64 input;
- `mobilenet_v2`: torchvision MobileNetV2 with depthwise convolutions and a 64x64 input.
- `efficientnet_b0`: torchvision EfficientNet-B0 with squeeze-and-excitation, SiLU, and depthwise convolutions;
- `densenet121`: torchvision DenseNet121 with dense concatenation blocks and transition pooling;
- `convnext_tiny`: torchvision ConvNeXt-Tiny with depthwise convolutions, LayerNorm, GELU, and residuals;
- `vit_b_16`: torchvision ViT-B/16 with twelve Transformer blocks, multihead attention, and LayerNorm;
- `deeplabv3_mobilenet_v3_large`: torchvision DeepLabV3 segmentation with a MobileNetV3 backbone and an output of 21-class pixel logits.

The script writes the PTE, external AOTI data, raw input, expected output,
runner output, and successful `result.json` report to the selected directory.
Without `--runner`, the report records an export only (`runner` is null).
It compares the `executor_runner`
output with the MUSA eager result; no CPU execution is used as a correctness
baseline. The GPU random input generator is synchronized before taking an
immutable host snapshot; eager execution and the runner read the same input.
Without that synchronization, an odd-sized batched input produced a raw file
with incomplete tail values even though the eager output was finite.

On the observed M1000 SDK/driver stack, synchronization alone is insufficient:
an SDK-only probe also reproduces incomplete pageable-host readbacks of
immutable GPU data. Input/reference snapshots therefore use
`backends/aoti/host_tensor.py`: copy into pinned CPU storage, synchronize, then
clone into owned ordinary CPU storage. MUSA weight serialization uses the same
helper in eight-MiB storage chunks. KV tensors reused by later eager steps
remain on the GPU. Allocation/copy errors fail acceptance instead of falling
back to an unsafe read or retrying a failed comparison. The C++ runtime has a
separate pageable-D2H staging workaround; see `backends/musa/tests/README.md`
for reproduction, costs, and the distinction from unchanged pinned transfers.

Floating inputs and weights default to FP32. `--dtype fp16` or `--dtype bf16` exercises
the same export and runner path in lower precision. The default absolute and
relative comparison tolerances are 1e-4 (FP32), 1e-2 (FP16), and 4e-2 (BF16);
token models always accept raw int64 input IDs and `--dtype` selects the
weights and output dtype. For `token_decoder`, the script also checks exact
argmax token IDs against MUSA eager. For ResNet18, MobileNetV2, EfficientNet-B0,
DenseNet121, ConvNeXt-Tiny and ViT-B/16 it checks predicted class IDs; for
DeepLabV3 it checks exact per-pixel predicted labels. These semantic checks
apply to every shape and every independent runner replay, not only the first
execution. ViT intermediate-feature diagnostics are not class predictions.
Passing a logit tolerance alone does not excuse a changed token, class or
pixel label. Predictions are compared against eager at the same dtype, not
against another precision or dataset labels.
Override the tolerances with `--atol` and
`--rtol` if needed. `--input-size 224` exports a supported torchvision
image model with a 224x224 input instead of its default size. Except
for the bounded dynamic-spatial `bilinear_unet` and DeepLabV3 cases below,
fixed spatial shapes and dtypes require separate exports.
`--batch-size 2` exports and checks a separate fixed-shape, two-input batch
(the default batch size is one).

`--batch-size 2 --dynamic-batch-max 4` instead exports one PTE with a bounded
dynamic batch dimension and checks batch sizes 2, 1, and 4 against MUSA eager.
The script pads raw input files to the bound and passes the actual shape via
`executor_runner --input_shapes`; that runner otherwise interprets the raw
buffer as a bound-sized tensor. For `bilinear_unet` only,
`--dynamic-spatial-max 48` exports one PTE with independent height and width
between 24 and 48. It checks 32x40, 24x30, 48x48, and 47x39 using that PTE.
For `resnet18`, `mobilenet_v2`, `deeplabv3_mobilenet_v3_large`, and `efficientnet_b0`,
`--dynamic-spatial-max 80` checks
64x64, 48x54, 80x80, and 79x71 using one PTE with independent height and
width between 48 and 80. Both include odd sizes. Dynamic batch and spatial
dimensions can also be combined for these supported models using
`--batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80` (use 48 for
the bilinear U-Net). This accepts a batch of 1–3 with independent bounded
height and width in one PTE.
ResNet18 and MobileNetV2 use `export_with_musa_pooling_shapes` for this path.
The installed PyTorch pooling meta helper tests the remainder of a symbolic
integer division even though Python `//` already rounds toward negative
infinity. This adds an unnecessary parity constraint to the ResNet stem's
pooling input. The helper temporarily uses `operator.floordiv` during export,
restores the original function on success or failure, and leaves the actual
pooling operators and model architecture unchanged. It preserves independent
integer height/width bounds rather than requiring multiples of four. As with
the SDPA export helper, export calls using this compatibility patch must be
serialized.
For `densenet121`, the tested export range uses `8 * Dim` for independent
height and width between 32 and 80, checking 64x64, 32x40, 80x80, and 72x64.
This constraint comes from the current exporter's pooling guards; it is not a
claim that eager DenseNet cannot process other sizes. The example size and
maximum must be multiples of 8, and the maximum must exceed 32.
`convnext_tiny` supports fixed spatial sizes and dynamic batch acceptance.
Other dynamic dimensions are not covered.
For torchvision models, `--weights /path/to/state_dict.pth` loads a
local PyTorch state dictionary (for example, a cached pretrained checkpoint)
before exporting. Without it, the model uses random weights; the script never
downloads weights automatically.

The MUSA backend disables TorchInductor's layout optimization so Conv1d/2d/3d
fallback outputs and generated kernels agree on contiguous NCHW layout. The
eager reference disables muDNN TF32 to match the fallback's FP32 FMA math.
The native fallback supports non-transposed and transposed Conv1d/2d/3d.
The MUSA BMM fallback supports contiguous output and contiguous or transposed
left- and right-hand matrices with safe batch padding. This covers the
non-contiguous batch view of ViT attention without copying the matrices;
arbitrary per-row strides are not yet supported.
Single-row BMM operands may have a zero stride for that singleton row axis,
as in the fixed-token upstream Llama KV-cache export; the leading batch
stride must not overlap distinct matrices unless the operand is a read-only
broadcast with zero batch stride. Broadcast batches use individual GEMM calls
instead of a strided-batched GEMM.
For packed BMM batches, a zero stride on the size-one row axis is normalized
in the muDNN descriptor and uses the same muDNN path as its positive-stride
equivalent. Selecting muBLAS only because of that unused stride changed BF16
PV rounding in Qwen3.5. An exact layout-equivalence regression covers
FP32/FP16/BF16 and fails on the old dispatch; all 20 native matrix tests pass
with the fix. Real eight-step attention operands also match eager after the
change. Broadcast batch strides and padded-batch fallback selection are unchanged.
The `transposed_bmm` acceptance model exercises a transposed left-hand matrix
through PTE/PTD export and `executor_runner` output comparison, including
bounded dynamic batch sizes.
The `broadcast_bmm` model broadcasts a read-only right-hand weight across
batches using a zero inter-batch stride; output matrices remain disjoint.
Earlier standalone addmm tests exposed intermittent padded-row FP16/BF16
errors. The subsequent packing and GEMM-path fixes described below now pass
repeated native regressions, including the re-enabled no-bias padded case;
this does not establish support for arbitrary matrix strides.
For transposed convolution, output padding must be smaller than either its
stride or its dilation, as required by PyTorch. Transposed Conv1d uses muDNN's
GEMM backward-data algorithm; its implicit-GEMM algorithm rejects the 1D
descriptor.
MUSA AOTI decomposes inference-only SDPA into matmul, softmax, and masking to
avoid an unavailable TorchMUSA AOTI fallback. Pattern matching is disabled
only while compiling graphs containing SDPA or attention-like softmax/matmul
patterns, as it otherwise fuses those ops back into the unavailable fallback.
Dropout is unsupported. TorchMUSA's native attention export specializes batch
dimensions. For `transformer_block`, `causal_attention`, and `vit_b_16` with
bounded dynamic batch, the acceptance script instead expands calls through
`torch.nn.functional.scaled_dot_product_attention` into the backend's math
implementation before export. Other callers can explicitly use
`export_with_musa_sdpa_decomposition` from `executorch.backends.musa`. This
temporarily replaces a process-global Python function during export, so do not
run concurrent exports, and calls bound to a separate SDPA alias are not covered.
Boolean/additive masks, causal attention, and GQA are covered separately;
combining a causal flag with a custom mask is not supported.
All-masked rows are not covered: on M1000, FP32 SDPA and the backend math path
return NaNs, while low-precision TorchMUSA flash attention returned nonzero
values for both boolean and additive masks in a 3-token probe. Silently mapping
these cases to zeros or flash output would change masking semantics.

## Prerequisites

- MUSA and `torch_musa` are available to the selected Python environment;
- the MUSA-compatible Triton package required by TorchInductor is installed;
- `torchvision` is available for the standard ResNet18, MobileNetV2, EfficientNet-B0, ViT-B/16, and DeepLabV3 models;
- `flatc` is available on `PATH` for PTE/PTD serialization;
- an ExecuTorch `executor_runner` built with `EXECUTORCH_BUILD_MUSA=ON`.

For a reproducible build plus native/compiler/runner regression entry point,
see [MUSA regression instructions](../../backends/musa/tests/README.md).
It builds the runner and native test targets directly from this checkout,
without hand-linking test binaries or installing the C++ libraries first.

For example, from the repository root:

```bash
export PYTHON=/usr/bin/python3.12
export PYTHONPATH=/tmp/executorch-pydeps:$PWD/src:$PWD
export PATH=/tmp/executorch-flatc-build:$PATH
export RUNNER=/tmp/executorch-musa-cmake-1792075449/executor_runner

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model mlp \
  --output-dir /tmp/executorch-musa-acceptance/mlp \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model transposed_bmm --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/transposed_bmm \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model broadcast_bmm --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/broadcast_bmm \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model transformer_block --dtype fp16 \
  --batch-size 2 --dynamic-batch-max 3 --runner-repeats 4 \
  --output-dir /tmp/executorch-musa-acceptance/transformer_block_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_cnn \
  --output-dir /tmp/executorch-musa-acceptance/simple_cnn \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_audio_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/simple_audio_decoder_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_video_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/simple_video_decoder_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model tiny_unet --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/tiny_unet_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model bilinear_unet --dtype bf16 --dynamic-spatial-max 48 \
  --output-dir /tmp/executorch-musa-acceptance/bilinear_unet_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model bilinear_unet \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 48 \
  --output-dir /tmp/executorch-musa-acceptance/bilinear_unet_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --input-size 224 --dtype bf16 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --batch-size 2 --dynamic-batch-max 3 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model efficientnet_b0 --batch-size 2 --dynamic-batch-max 3 \
  --dynamic-spatial-max 80 \
  --weights /path/to/efficientnet_b0_rwightman-7f5810bc.pth \
  --output-dir /tmp/executorch-musa-acceptance/efficientnet_b0_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --batch-size 2 --dynamic-batch-max 3 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 --batch-size 2 --dynamic-batch-max 3 \
  --vit-blocks 6 --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_prefix6 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model vit_b_16 --dtype fp16 --batch-size 2 --dynamic-batch-max 3 \
  --vit-layer-tokens --vit-attention-stages --runner-repeats 5 \
  --weights /path/to/vit_b_16-c867db91.pth \
  --output-dir /tmp/executorch-musa-acceptance/vit_b_16_layer_tokens \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_pretrained \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --dynamic-spatial-max 80 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_dynamic_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80 \
  --output-dir /tmp/executorch-musa-acceptance/deeplabv3_batch_spatial \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model dilated_video_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/dilated_video_decoder_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model resnet18 --input-size 224 \
  --output-dir /tmp/executorch-musa-acceptance/resnet18_224 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model mobilenet_v2 --batch-size 2 --input-size 64 \
  --output-dir /tmp/executorch-musa-acceptance/mobilenet_v2_batch2 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_cnn --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/simple_cnn_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model simple_video_decoder --dtype bf16 \
  --batch-size 2 --dynamic-batch-max 3 \
  --output-dir /tmp/executorch-musa-acceptance/video_decoder_dynamic \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model causal_attention --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/causal_attention_bf16 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model token_decoder --dtype bf16 \
  --output-dir /tmp/executorch-musa-acceptance/token_decoder_bf16 \
  --runner "$RUNNER"
```

To validate official pretrained ResNet18 weights, download
`https://download.pytorch.org/models/resnet18-f37072fd.pth` separately and pass
the local checkpoint with `--weights`. Its SHA-256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
This checks inference against MUSA eager using a random input, not ImageNet
accuracy or performance.
MobileNetV2 IMAGENET1K_V2 weights are available at
`https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth` (SHA-256
`7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`).
EfficientNet-B0 IMAGENET1K_V1 weights are available at
`https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`
(SHA-256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`).
ViT-B/16 IMAGENET1K_V1 weights are available at
`https://download.pytorch.org/models/vit_b_16-c867db91.pth`
(SHA-256 `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`).
ViT-B/16 requires 224x224 inputs. Its untrained classifier produces all-zero
logits, so only the pretrained checkpoint provides a meaningful output check.
DeepLabV3 MobileNetV3 COCO-with-VOC-labels weights are available at
`https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth`
(SHA-256 `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70`).
DeepLabV3 loads the full checkpoint, including its auxiliary classifier, but
exports only the primary `out` logits. The acceptance input is random, not
normalized dataset images; no segmentation accuracy is measured.

On M1000 with torch-musa 2.9.0.post1+musa5.1.2mp22, the following exports
passed `executor_runner` versus MUSA eager (maximum absolute output error):

| Model | Input | Dtype | Weights | Max error |
| --- | --- | --- | --- | ---: |
| MLP | 1x16 | FP16 / BF16 | Random | 4.88e-4 / 3.91e-3 |
| Simple CNN | 1x3x32x32 | FP16 / BF16 | Random | 2.44e-4 / 1.95e-3 |
| Simple 3D CNN | 1x1x4x8x8 | FP32 | Random | 7.45e-9 |
| Simple decoder | 1x3x8x8 | FP32 | Random | 5.96e-8 |
| Simple audio decoder | 1x2x5 | FP32 / FP16 / BF16 | Random | 2.98e-8 / 2.44e-4 / 1.95e-3 |
| Simple video decoder | 1x1x2x3x4 | FP32 / FP16 / BF16 | Random | 0 / 0 / 0 |
| Dilated audio decoder | 1x2x5 | FP32 / BF16 | Random | 5.97e-8 / 3.91e-3 |
| Dilated image decoder | 1x2x5x4 | FP32 / BF16 | Random | 0 / 0 |
| Dilated video decoder | 1x2x5x4x3 | FP32 / BF16 | Random | 0 / 0 |
| Tiny U-Net | 1x3x32x32 | FP32 / FP16 / BF16 | Random | 6.56e-7 / 1.95e-3 / 1.56e-2 |
| Bilinear U-Net | 1x3x32x40 | FP32 / BF16 | Random | 2.38e-7 / 7.81e-3 |
| TransformerEncoderLayer | 1x4x16 | FP32 / FP16 / BF16 | Random | 4.77e-7 / 1.95e-3 / 7.81e-3 |
| Causal GQA | 1x4x16 | FP32 / FP16 / BF16 | Random | 1.19e-7 / 1.22e-4 / 3.91e-3 |
| Token encoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 5.96e-8 / 4.89e-4 / 1.96e-3 |
| Token decoder | 1x4 int64 | FP32 / FP16 / BF16 | Random | 1.79e-7 / 4.89e-4 / 3.91e-3 |
| ResNet18 | 1x3x224x224 | FP32 / FP16 | Official pretrained | 6.05e-6 / 3.91e-3 |
| ResNet18 | 1x3x64x64 | BF16 | Official pretrained | 0 |
| ResNet18 | 1x3x224x224 | BF16 | Official pretrained | 0 |
| MobileNetV2 | 1x3x224x224 | FP32 | Official pretrained | 2.03e-6 |
| MobileNetV2 | 1x3x64x64 | FP16 | Official pretrained | 1.37e-2 (`--atol 0.02`) |
| EfficientNet-B0 | 1x3x64x64 | FP32 | Official pretrained | 7.39e-6 (class 644) |
| EfficientNet-B0 | 1x3x224x224 | FP32 / FP16 / BF16 | Official pretrained | 1.79e-6 / 4.09e-3 / 2.83e-2 (class 314) |
| ViT-B/16 | 1x3x224x224 | FP32 | Official pretrained | 3.81e-6 (class 21) |
| ViT-B/16 | 1x3x224x224 | FP16 / BF16 | Official pretrained, initial single run only | 1.95e-3 / 3.13e-2; repeated execution fails |
| ViT-B/16 | batch 2/1/3 x3x224x224 | FP32 | Official pretrained, one dynamic PTE | <=7.15e-6 (class IDs match) |
| DeepLabV3 MobileNetV3 | 1x3x64x64 | FP32 | Official pretrained | 9.16e-5 (0/4096 differing pixel labels) |
| DeepLabV3 MobileNetV3 | 2x3x64x64 | FP32 | Official pretrained | 1.18e-4 (0/8192 differing pixel labels) |
| ResNet18 | 2x3x64x64 | FP32 | Official pretrained | 3.34e-5 |
| MobileNetV2 | 2x3x64x64 | FP32 | Official pretrained | 1.76e-5 |

With a single dynamic-batch PTE for each model, batches 2/1/4 of FP32 MLP
passed (maximum error 4.47e-8). Batches 2/1/3 also passed for FP32 simple CNN
(3.58e-7), pretrained ResNet18 (3.53e-5), and pretrained MobileNetV2
(1.76e-5); both pretrained models used 64x64 inputs.
Official pretrained EfficientNet-B0 also passed batches 2/1/3 using one FP32
PTE at 64x64 (maximum logit error 8.35e-6); predicted class IDs matched eager
for all batches. One dynamic-batch PTE for the FP32 TransformerEncoderLayer
also passed batches 2/1/3 (maximum error 4.77e-7). The FP16 model passed
An official pretrained ResNet18 BF16 PTE at 224x224 also passed dynamic batches
2/1/3 with maximum errors 0, 0, and 0.00390625, respectively.

The FP16 model passed
12 fresh-runner replays for each of batches 2/1/3 (maximum error 1.95e-3);
BF16 passed four executions for each batch (maximum error 1.56e-2).
`--runner-repeats` checks the same exported PTE/PTD and input against eager
on every execution, rather than relying on a single run. Official pretrained
FP32 ViT-B/16 passed the same batches at 224x224 using one PTE (maximum
error 7.15e-6); predicted class IDs matched MUSA eager for each input.
Dynamic ViT-B/16 FP16 and BF16 do **not** pass this check: the original
full-depth PTE produced finite batch-2 logits but 1,000 NaNs for batch 1,
while MUSA eager stayed finite. The `--vit-blocks` option truncates the
encoder after loading weights to help reproduce the issue. The
`--vit-layer-tokens` option outputs a `[batch, blocks + 1, 768]` tensor:
stage 0 is the class token after the positional embedding and stage `j + 1`
is the class token after encoder block `j`. On failure, the script reports
per-batch NaN counts at each stage without classifying them as image labels.
`--vit-attention-stages` adds a stage after each attention residual:
stage `2*j + 1` follows attention in block `j`, while stage `2*j + 2`
follows its MLP. In eight runs of a 12-block FP16 diagnostic PTE, the first
class-token NaNs in both batches appeared at odd-numbered stages, though the
block varied across runs. This implicates the attention half of the block
without proving which internal tensor first became invalid.

The acceptance exporter now uses `export_with_musa_sdpa_decomposition` for
both static and dynamic ViT-B/16 exports. This avoids the MUSA export-time
shape-specializing SDPA path; the backend already provides the same
decomposition for lowering. A newly exported static pretrained BF16 PTE passed
8/8 fresh runner replays with maximum error 0.03125, and a newly exported
static pretrained FP16 PTE passed 5/5 with maximum error 0.001953125. This
fixes the static acceptance path, but does not yet establish dynamic low
precision ViT support; the dynamic checks above remain failing.
The shim regression tests reproduce the ViT QKV projection layout with
`[197 or 394, 768] @ [768, 2304]`, a transposed weight and a broadcast bias.
Under repeated fresh processes, the standalone projection occasionally
produced a zero-valued prefix (4-16 elements) instead of the expected 2.5,
including FP32 and FP16 cases. One failed run read the input tensor's first
element back as zero too; this does not distinguish an input-transfer or
storage problem from corruption during GEMM. A device-wide synchronization
before GEMM and synchronous host-to-device copies did not eliminate the
intermittent failure. The projection stress test was initially disabled to
avoid flaky CI, and is enabled after the broadcast-copy improvement below.
The existing small addmm and BMM tests also intermittently returned zeros
in separate full-suite runs; the issue is not isolated to this new shape.
The separate ViT-sized attention-score BMM regression passed 30 fresh test
processes. These standalone tests do not yet prove the cause of the pretrained
low-precision model's NaNs: an extra QKV output inserted into a diagnostic
model compiled into a *second* projection GEMM, not an observation of the
original projection output.
Replacing the one-device-copy-per-row bias broadcast in `musa_addmm.cpp`
with non-overlapping doubling copies reduced a 197-row projection from 197
copies to 9. The standalone projection passed 120 fresh processes after
this change, versus intermittent failures before it; a five-row non-power-of-two
broadcast regression also passes. This does **not** fix the pretrained
low-precision ViT: replaying the same static PTE/PTD five times with the new
shim still yielded 1,000 NaN logits in 4/5 FP16 and 5/5 BF16 runs. The FP32
control passed 5/5 runs (maximum error 3.81e-6). More attention-path diagnosis
is required before counting pretrained low precision as supported. Across 12
full shim-test processes after this change, 7 passed and 5 still had
intermittent zero outputs, including the pre-existing padded addmm case and
an attention-score BMM case. The improved standalone projection is not a
general reliability fix.
A focused padded-mat1 check now compares transfers with and without GEMM:
three independent input allocations, including all seven physical elements
of a padded row layout, survived 100 fresh transfer-only processes. The
same layout with `beta=0` (no bias broadcast) returned second-row NaNs by
the third fresh process. A failing biased run returned the same NaNs on
repeated readback even though the original mat1, mat2, and bias read back
correctly afterward from distinct allocations. Thus neither bias broadcast
nor pageable device-to-host copies alone explains the remaining failure.
Packing padded mat1 into a contiguous scratch buffer still failed in an
isolated replay, as did synchronization before destroying the mublas handle;
both experiments were reverted. The no-bias stress check was disabled during
this investigation. After the later packing fix and repeated successful
validation, it is enabled in normal CI again; select it explicitly with
`--gtest_filter=MusaAddmmTest.PaddedMat1WithoutBias`.
These observations implicate the GEMM execution or its ordering but do not
identify a validated runtime or SDK fix.
For comparison, PyTorch MUSA eager `torch.addmm` with the same padded
`[2, 3]` input, `[3, 2]` weight, 1-D bias, three dtypes, and `beta=0/1`
matched exact hand-calculated results in 20 fresh processes, with ten
allocations per dtype per process (1,200 addmm calls total). Eager may use a
different algorithm, handle lifetime, or input layout; this contrast does
not establish which shim or SDK step fails. The next useful comparison is
the two paths' actual GEMM arguments, stream ordering, and input packing.
For a full-depth FP16 PTE in layer-token mode, five replays of the same
batch-2 input first produced NaNs at stages [none, 5], [4, 4], [6, 6],
[none, none], and [4, 2] for its two batches. Stage 0 and stage 1 stayed
finite in these runs, but the first failing block was not fixed. For a
six-block FP16 PTE, one acceptance run passed batches 2/1/3; repeating its
identical batch-2 input in fresh runner processes produced NaN counts of [0, 1000],
[1000, 1000], and [0, 1000]. A seven-block PTE first returned 2,000 NaNs
for batch 2, then three identical replays returned finite logits. These
results do not establish a specific failing layer or a reliable low-precision
dynamic path. Synchronizing after the full AOTI run also left all 2,000
six-block FP16 logits as NaNs in two separate replays. Device-wide
synchronization during some diagnostic matrix
operations removed FP16 NaNs, but synchronizing before mublas handle
destruction did not fix the full ViT; changing streams, retaining handles,
or disabling Inductor buffer reuse did not validate a fix. No synchronization
workaround is enabled for production. `MUSA_LAUNCH_BLOCKING=1` yielded five
finite six-block replays but only one of two finite full-depth FP16 replays
and no finite full-depth BF16 replays; it is not a validated workaround.
The static-batch pretrained FP16/BF16 entries above are not stable either:
their original PTE/PTD and input produced 1,000 NaN logits in each of eight
fresh runner processes per dtype, although the recorded first runs and MUSA
eager outputs were finite. A newly exported static FP16 PTE also returned
1,000 NaN logits on its first run. The issue is not limited to dynamic batch;
the original pretrained FP32 static PTE passed five identical replays with
maximum logit error 3.81e-6.

A subsequent MUSA-only shim change uses the same muDNN `MatMul` and
`BatchMatMul` classes as PyTorch MUSA eager for addmm/mm and packed,
non-broadcast BMM, including fused one-dimensional bias. Padded addmm input
rows are packed first; broadcast or padded BMM batches retain the existing
muBLAS path. SlimTensor's MUSA allocations use synchronous `cudaMalloc`
instead of stream-ordered allocation because the latter still gave intermittent
zero outputs in the focused matrix tests. This is not a measured performance
improvement and its allocation overhead and graph-capture compatibility need
evaluation. All 13 C++ shim tests, including the explicitly enabled padded
no-bias regression, passed in one fresh process; the focused addmm/BMM tests
passed in five more fresh processes. Replaying the *unchanged* static pretrained
ViT PTE/PTD and input eight times per dtype gave finite FP16 logits in 8/8
runs (maximum absolute error 0.001953125 versus recorded MUSA eager output)
and finite BF16 logits in 6/8 runs (maximum finite-run error 0.03125); the
other two BF16 runs each contained 1,000 NaN logits. The earlier muBLAS
results above describe the old shim, not this one. BF16 pretrained ViT is
still not a reliable supported configuration, and eight successful FP16
replays do not establish production reliability or throughput. The original
FP32 static PTE also produced finite logits in 3/3 fresh runs with maximum
absolute error 1.1921e-6 after the shim change. The 13 C++ shim tests and all
31 MUSA AOTI Python export tests passed with this implementation.

To narrow down the remaining BF16 NaNs, a new classifier-shaped shim regression
covers `[1, 768] @ [768, 1000]` with a zero first-row stride, transposed
weights, and one-dimensional bias. All three dtypes passed four calls each in
30 fresh processes, and the full shim suite (including the explicitly enabled
padded regression) passed 13/13. A diagnostic BF16 pretrained ViT PTE that
returns class tokens before and after each attention residual passed 12/12
fresh replays with no NaNs; its input matched the original PTE's input
byte-for-byte. Separately, a BF16 PTE containing the *actual pretrained*
final LayerNorm and classifier, fed the eager last-stage class token, matched
MUSA eager exactly in 16/16 fresh runs. Neither experiment preserves the
original full model's allocation and execution schedule, so these results do
not prove that its classifier or attention is sound. Switching the MUSA
SlimTensor free path from `cudaFreeAsync` to synchronous `cudaFree` still
produced 1,000 NaN logits in one of five original BF16 full-model replays;
the change was reverted. The intermittent full-model BF16 failure remains.
These checks use random inputs, not ImageNet accuracy or
throughput measurements.
A separate pretrained FP32 PTE passed independent spatial sizes 64x64, 48x54,
80x80, and 79x71 (maximum logit error 5.96e-6). Another FP32 PTE with jointly
dynamic batch and spatial dimensions passed 2x64x64, 1x48x54, 3x80x80, and
2x79x71 (maximum logit error 5.96e-6); predicted class IDs matched eager in
each case. This does not validate arbitrary combinations within the bounds.
For the jointly dynamic PTE at 2x64x64, pretrained FP16 and BF16 do **not**
meet their default tolerances: 5/2,000 and 7/2,000 logits exceed tolerance,
respectively (maximum differences 0.021484375 and 0.0703125). Both predicted
class IDs remain correct for this input, but neither dtype counts as passed
for combined dynamic shapes. Static 224x224 FP16/BF16 remain separately verified.
For the simple 3D CNN, batches 2/1/3 passed with a single PTE in both FP16
(maximum error 6.11e-5) and BF16 (4.89e-4).
For the simple decoder, batches 2/1/3 passed with a single PTE in both FP16
and BF16 (maximum error 0 for this test input).
With a separate bounded dynamic-batch PTE for each BF16 model, batches 2/1/3
passed for the grouped audio decoder (maximum error 1.95e-3), video decoder
(0), dilated audio decoder (3.91e-3), dilated image decoder (0), and dilated
video decoder (0). This checks dynamic batch only, not dynamic spatial sizes.
The tiny U-Net also passed BF16 batches 2/1/3 with one PTE (maximum error
1.56e-2), covering GroupNorm and the concatenated skip connection.
One dynamic-spatial bilinear U-Net PTE per dtype passed the 32x40, 24x30,
48x48, and 47x39 inputs in FP32 and BF16 (maximum errors 3.58e-7 and
7.81e-3, respectively). These are independent bounded spatial dimensions,
not a validation of arbitrary sizes or a pretrained segmentation model.
One jointly dynamic FP32 bilinear U-Net PTE also passed 2x32x40, 1x24x30,
3x48x48, and 2x47x39 (maximum error 4.17e-7). The last case previously
failed with an unsynchronized GPU-generated input snapshot; replaying clean
input through the same PTE showed no operator mismatch.
The official pretrained DeepLabV3 output varies spatially; all 86,016 FP32
logits met the default 1e-4 tolerances and all 4,096 predicted pixel labels
matched MUSA eager. Pretrained FP16 at 64x64 does **not** meet the default
tolerance: 2,412/86,016 logits differ beyond 1e-2 absolute and relative
tolerances (maximum error 0.03125), although the predicted labels match for
this input. Do not count it as a passed FP16 configuration; BF16 has not been
validated for this model.
The same pretrained FP32 DeepLabV3 also passed batches 2/1/3 using one
bounded-dynamic-batch PTE (maximum logit error 1.18e-4); all predicted pixel
labels matched for each batch. A separate bounded-dynamic-spatial PTE also
passed 64x64, 48x54, 80x80, and 79x71 (maximum logit error 1.63e-4), with
all predicted pixel labels matching eager. One jointly dynamic PTE also
passed inputs 2x64x64, 1x48x54, 3x80x80, and 2x79x71 (maximum logit error
1.63e-4); all pixel labels matched eager. This validates only the bounded
shapes exercised here, not arbitrary resolutions or every combination.
The embedding classifier also passed batches 2/1/3 in FP32 with one PTE
(maximum error 1.20e-7), confirming dynamic int64 inputs without attention.
The token decoder matched eager token IDs `[4, 13, 56, 44]` in all three
precisions. It is a fixed-sequence toy model, not a pretrained LLM or a
KV-cache/incremental decoding validation.

`aoti_kv_cache_acceptance.py` separately exercises two consecutive single-token
decode steps with a rolling four-position KV cache. The same fixed-shape PTE
accepts int64 token IDs plus key/value tensors and returns logits plus updated
key/value tensors; the second `executor_runner` invocation reads the first
invocation's raw cache outputs. Every output is compared with MUSA eager and
the predicted token IDs must match exactly:

```bash
python examples/musa/aoti_kv_cache_acceptance.py \
  --output-dir /tmp/musa-kv-cache --runner /path/to/executor_runner \
  --dtype fp32
```

The default rolling mode tests raw multi-input/multi-output cache roundtrips,
not dynamic cache growth, batched decoding, a pretrained LLM, or throughput.
The oldest cache position is discarded at each step to keep all shapes static.
On M1000, FP32/FP16/BF16 all passed both steps with eager-matching predicted
token IDs `[3, 32]`; maximum logits differences were 2.38e-7, 9.77e-4, and
3.91e-3 respectively. Both returned caches also matched MUSA eager at each step.

Pass `--cache-mode growing` to test an append-only cache instead. One PTE with
bounded dynamic key/value sequence dimensions runs with past lengths 3 then 4
and returns updated cache lengths 4 then 5. The first input is padded to the
exported upper bound and resized with `--input_shapes`; the second runner call
reads the preceding runner output cache files directly. Pass
`--initial-cache-length 2` to exercise past lengths 2 then 3; its second input
is padded to the upper bound before the runner reads it. On M1000, FP32 passed
both starting lengths with maximum logits differences 2.38e-7 (3/4) and
1.91e-7 (2/3); FP16/BF16 passed the 3/4 case with zero output difference for
this input. All predicted token IDs matched MUSA eager. This does not establish
arbitrary sequence lengths beyond the exported input bound (2–4), variable
batches, or pretrained-model support.

`aoti_prefill_decode_acceptance.py` exports a separate three-token causal
prefill and bounded-dynamic decode PTE from the *same* toy decoder weights.
The prefill runner produces logits and key/value caches; its output files are
padded to the decode bound for the first call, whose cache outputs feed a
second decode call without padding. Runner outputs are checked against MUSA
eager at every step, including exact
argmax token IDs:

```bash
python examples/musa/aoti_prefill_decode_acceptance.py \
  --output-dir /tmp/musa-prefill-decode --runner /path/to/executor_runner \
  --dtype fp32
```

The prefill returns transposed key/value views. The shared CUDA-compatible
runtime now copies noncontiguous SlimTensor outputs with nonzero storage offsets
to planned ET tensors; the decode export uses contiguous cache inputs to match
the runner's raw file layout. On M1000, FP32 prefill and two decode calls passed
with maximum logits errors of 2.99e-7, 1.20e-7, and 1.50e-7 respectively;
FP16 and BF16 also passed with exact matching token IDs `[16, 40, 32]`.
This is a randomly initialized toy model, not a pretrained LLM or general
variable-length prefill acceptance.
Use `--dynamic-prefill` to export one bounded-dynamic prefill PTE and run
prompts of lengths 2, 3, and 4 against MUSA eager before decoding the
three-token prompt. Input buffers are padded to the export bound and their
actual shape is supplied through `--input_shapes`. On M1000, FP32, FP16, and
BF16 passed all three prompt lengths and both subsequent decode steps with
their respective pairs of PTEs.

`aoti_llama_acceptance.py` exercises the repository's actual five-layer Llama
architecture (RoPE, RMSNorm, SwiGLU, and attention) using its bundled
`demo_rand_params.pth` checkpoint and `demo_config.json`. It exports a PTE/PTD,
runs `executor_runner` on a three-token prompt, and compares all 512 logits
and the exact predicted token ID with MUSA eager. It also rejects unlowered
operators outside the single MUSA delegate:

```bash
python examples/musa/aoti_llama_acceptance.py \
  --output-dir /tmp/musa-llama-demo --runner /path/to/executor_runner
```

This checkpoint contains **random weights**, not a pretrained Llama. The
validation covers one fixed prompt with no KV cache; it establishes neither
useful language generation nor performance for a production model.
On M1000, FP32/FP16/BF16 all predicted token ID `364`; maximum absolute logit
errors against MUSA eager were 4.77e-7, 1.46e-3, and 1.17e-2, respectively.

`aoti_llama_kv_acceptance.py` exports the same five-layer Llama demo with its
upstream in-module KV cache enabled and runs five single-token steps through
one `executor_runner --server_mode` process and one PTE/PTD. Positions 0–2
prefill `[11, 7, 22]` token-by-token, and positions 3–4 decode `[5, 17]`.
Every 512-logit output and predicted token ID is checked against a separately
advanced MUSA eager model:

```bash
python examples/musa/aoti_llama_kv_acceptance.py \
  --output-dir /tmp/musa-llama-kv --runner /path/to/executor_runner
```

For this export the KV buffers live inside the AOTI delegate; restarting the
runner resets the cache, so all steps must execute in one process. The export
uses a fixed one-token input and an eight-position context bound, rather than
variable-length prompt prefill or externally serialized caches. On M1000,
FP32/FP16/BF16 passed all five steps with maximum logit errors 6.56e-7,
1.50e-3, and 1.56e-2, respectively, and predicted IDs
`[33, 351, 364, 76, 464]`. This still uses **random**, not pretrained, weights.
The upstream model passes its position tensor inside an `attn_options` dict;
the shared AOTI compiler now reconstructs that input tree. Its mutable cache
graph also requires MUSA's device assertion to use the actual indexed device
instead of the unindexed device string produced by PyTorch's device move pass.

Pass `--dynamic-prefill` to export one bounded-dynamic Llama KV PTE that accepts
one to seven tokens per call within its eight-position context. The runner
processes `[11, 7, 22]` together, then `[5]` and `[17]` separately in the
**same process**. A second, fresh runner process uses the **same PTE/PTD** to
prefill `[11, 7, 22, 5, 17, 13, 19]` in one call and decode `[3]` at the final
context position. Server requests carry
`input_shapes=1xL,L` to resize the token and position inputs on each step;
upper-bound-sized raw input files keep the storage valid. TorchMUSA currently
lacks a fake implementation for its dynamic SDPA math operator, and a fake
kernel alone still specializes sequence length. This export retains the
unaltered upstream SDPA module and uses
`export_with_musa_sdpa_decomposition` to expand functional SDPA before
TorchMUSA selects its native attention operator. Each runner step is checked
against upstream MUSA eager. On M1000, FP32/FP16/BF16 passed all three calls
with predicted IDs `[364, 76, 464]` and maximum logit differences 5.96e-7,
1.50e-3, and 9.77e-3, respectively. The long prefill and final decode also
passed all three dtypes with predicted IDs `[103, 344]` and maximum logit
differences 4.77e-7, 1.34e-3, and 7.81e-3. The checkpoint still has random
weights; this does not validate a production Llama model.

These are random-input correctness checks, not arbitrary shape-generalization
or throughput measurements. The previously recorded pretrained MobileNetV2
BF16 discrepancy was isolated to BatchNorm rounding. The backend now routes
inference BatchNorm1d/2d/3d through a native muDNN AOTI shim. The official
pretrained MobileNetV2 checkpoint now passes at 64x64 with `--atol 0.02
--rtol 0.04`, maximum difference `7.62939453125e-06`, and two independent
runner replays.

Without `--runner`, the script only exports the PTE/PTD and writes the eager
reference files. The runner command automatically adds the MUSA, Torch, and
Python shared-library directories used by the current environment.

## Latest MUSA AOTI validation (2026-10-18)

The current branch `codex/musa-aoti-poc` includes commits
`4c2151d066` (`stabilize Musa BF16 addmm paths`) and `b4d21a2493`
(`musa: avoid unstable strided batched bmm`). The BF16 small-batch and
padded-row addmm paths use muBLAS GEMM with explicit bias initialization;
larger BF16 addmm keeps the muDNN path. A zero-row-stride `[1, 768]` input,
transposed `[768, 1000]` weight, and padded `[2, 768]` input are covered by
repeated C++ regressions. The 13-test shim suite passed five complete
repetitions, and all 31 MUSA AOTI Python export tests passed.

End-to-end FP32 acceptance also passed with generated PTE/PTD and
`executor_runner`: `simple_cnn` with dynamic batches 2/1/3 (maximum error
3.58e-7), torchvision `resnet18` at 64x64 (maximum error 3.22e-6), and
`mobilenet_v2` at 64x64 (maximum error 1.28e-15); each model passed an
independent runner replay. These are random-weight functional checks, not
ImageNet accuracy or performance measurements.

Additional BF16 functional checks now pass with generated PTE/PTD and the
runner: `mlp` with dynamic batches 2/1/3 (maximum error 0.00390625),
`simple_cnn` with dynamic batches 2/1/3 (maximum error 0.001953125), and
random-weight `resnet18` and `mobilenet_v2` at 64x64 (maximum error 0.0 in
both cases). These checks cover the same SlimTensor/AOTI path but do not
establish pretrained-model accuracy.

The new `simple_audio_cnn` model exercises Conv1d, strided Conv1d, AvgPool1d,
and Linear end to end. Its dynamic batch 2/1/3 acceptance passes in FP32 with
maximum error 1.49e-8 and in BF16 with maximum error 0.00048828125, including
an independent runner replay for each batch size.

Commit `b13bf26773` (`musa: add muDNN batch norm AOTI shim`) adds an inference
BatchNorm fallback for contiguous tensors with or without affine parameters in
FP32/FP16/BF16. It uses a
custom decomposition plus an AOTInductor `ExternKernelAlloc` call, so generated
wrappers call `aoti_torch_musa_batch_norm` directly instead of ProxyExecutor.
The shim uses muDNN's inference API and supports rank-2/3/4/5 inputs, including
FP32 BatchNorm parameters with FP16/BF16 activations. BF16
runner acceptance now passes for `simple_audio_cnn` (maximum error
0.0001220703125), `simple_cnn` (0.001953125), `simple_video_cnn` (0), and
`narrow_resnet18` (0).

The rank-2 BatchNorm1d path and a mixed-dtype BF16 input/FP32 parameter path
also pass generated PTE/PTD plus `executor_runner` with maximum error `0`. The
inference path with `affine=False` is also covered and matches eager exactly.

The same commit closes the pretrained MobileNetV2 BF16 case using
`mobilenet_v2-7ebf99e0.pth`: the 64x64 export and two runner replays pass with
maximum error `7.62939453125e-06` under `--atol 0.02 --rtol 0.04`. The earlier
`0.140625` maximum error was caused by AOTI's pointwise BatchNorm arithmetic,
not by the convolution or BMM shims.

The deeper BF16 ViT failure was traced to the strided-batched BMM path used by
attention. The MUSA shim now uses one `mublasGemmEx` launch per batch for all
supported BMM dtypes in the fallback path, avoiding the unstable
`mublasGemmStridedBatchedEx` schedule. Full-depth pretrained ViT-B/16 BF16 now
passes generated PTE/PTD plus `executor_runner` for dynamic batches 2/1/3,
with three independent runner replays per batch and maximum error 0.03125.
This is a correctness-first fallback; batched-BMM performance optimization
and broader BF16 model coverage remain open.

The random-weight EfficientNet-B0 BF16 path also passes one combined dynamic
PTE/PTD with `executor_runner`: batch 2 at 64x64, batch 1 at 48x54, batch 3 at
80x80, and batch 2 at 79x71 all match MUSA eager with maximum error below
`7e-16`. This exercises the squeeze-and-excitation, depthwise convolution,
BatchNorm, SiLU, and dynamic-shape paths together.

Before the native activation and fused-bias precision fixes described below,
the official pretrained DeepLabV3-MobileNetV3 BF16 path passed the same four
dynamic batch/spatial shapes and two runner replays when invoked with
`--atol 0.5 --rtol 0.04`. Maximum raw-logit errors are `0.25`, `0.125`,
`0.25`, and `0.5`; all 86,016 predicted segmentation labels match MUSA eager
exactly across the four shapes. The wider absolute tolerance was required for
the accumulated BF16 logits at odd spatial sizes, while the exact label check
remains strict.

Random-weight DenseNet121 passes FP32 PTE/PTD export and two fresh runner
executions at 1x3x64x64 (maximum error `2.15e-6`). One combined dynamic PTE
also passes shapes 2x3x64x64, 1x3x32x40, 3x3x80x80, and 2x3x72x64, with
two runner executions per shape and maximum error `4.06e-6` at
`atol=rtol=1e-4`. This exposed a non-contiguous BatchNorm input produced by
dense-block buffer layouts; the BN shim now materializes contiguous inputs
and parameters only when necessary. Four native BN tests pass, including
non-contiguous input and padded-batch/strided-parameter cases.

The same four DenseNet shapes also pass BF16, with two runner executions per
shape and maximum errors `0`, `1.19e-7`, `0`, and `0.00390625` respectively
at `atol=rtol=0.04`. Predicted classes match eager for every execution.
Random-weight ConvNeXt-Tiny passes FP32 and BF16 at 64x64 with dynamic batches
2/1/3 and two runner executions per batch. After the fused-convolution-bias
change, regenerated PTEs pass again with maximum errors `3.58e-6`
and `0.0078125` respectively; predicted classes match on the initial runs.
These random-weight checks establish execution correctness for the tested
inputs, not pretrained classification accuracy or performance.

Reproduce the DenseNet and ConvNeXt runner checks using the environment above:

```bash
$PYTHON examples/musa/aoti_model_acceptance.py \
  --model densenet121 --dtype fp32 --batch-size 2 \
  --dynamic-batch-max 3 --dynamic-spatial-max 80 --runner-repeats 2 \
  --output-dir /tmp/executorch-musa-acceptance/densenet121_fp32 \
  --runner "$RUNNER"

$PYTHON examples/musa/aoti_model_acceptance.py \
  --model convnext_tiny --dtype fp32 --batch-size 2 \
  --dynamic-batch-max 3 --runner-repeats 2 \
  --output-dir /tmp/executorch-musa-acceptance/convnext_tiny_fp32 \
  --runner "$RUNNER"
```

`aoti_backbone_diagnostics.py` measures per-stage DeepLab-MobileNetV3 errors
against MUSA eager. It exports all backbone stages as outputs and records
max/mean/p99 absolute errors and NaN/Inf counts in `result.json`.
Use `--stage 4`, for example, to export just stage 4 with its eager predecessor
as input; this separates local error from accumulation through the backbone.
Run the same command with `--dtype fp32` and `--dtype bf16` into separate
directories. These are diagnostic measurements, not an accuracy acceptance
threshold. Retaining intermediate outputs can change fusion and allocation,
so also run the ordinary full-model acceptance path.

With the official pretrained checkpoint, seed 0 and a 2x3x64x64 input,
the FP32 diagnostic has maximum error at most `4.58e-5` across 17 stages.
The BF16 diagnostic has stage-0 maximum error `0.015625`, stage-6 error
`0.3125`, and final-stage error `0.171875`; no NaN/Inf values appear.
Isolated stage 4 has maximum error `0.0625` (versus `0.1201171875` in the
cumulative run). This confirms a local BF16 discrepancy as well as propagated
error; it does not yet identify a particular convolution algorithm as the
cause. These stage outputs are backbone features, not segmentation logits.

The subsequent operator-level probe (`--stem-operators`) identifies the
stem's BF16 discrepancy in Hardswish: Conv2d and BatchNorm2d match exactly,
while decomposed Hardswish differs by `0.015625`, including when isolated
with `--stem-operator 2`. The MUSA backend now calls a native muDNN Hardswish
shim instead. With that shim all three stem outputs and backbone stages 0–3
match exactly; the remaining backbone discrepancy starts at stage 4 and the
last-stage maximum error is `0.140625`. This only resolves the Hardswish
component of the BF16 difference; full-model acceptance remains separate.

Hardswish's C++ tests cover FP32/FP16/BF16, contiguous and strided offset views,
scalar and empty inputs, and preservation of input storage. All three tests
pass. The fallback registration and dynamic-batch AOTI compile tests pass for
all three dtypes. The muDNN C unary API binds data addresses into its tensor
descriptors, so the shim uses separate input and output descriptors.

`--block-operators 4` emits the intermediate operators in the main branch of
inverted-residual stage 4, fed by its eager predecessor. For residual stages,
this mode intentionally excludes the final residual addition. Diagnostic
outputs are cloned before the next operator runs so in-place activations do
not overwrite an earlier observation.

Further SE isolation finds another source of BF16 error: the eager convolution
fuses bias into the accumulator, whereas Inductor's generic GPU lowering
removes bias from the convolution and adds it after rounding the output.
The MUSA compile context now keeps bias in the native forward-convolution
call. The shim uses `mudnnConvolutionBiasActivationForward` with identity
activation and a one-dimensional bias descriptor. Transposed convolution keeps
its separate bias addition, matching the eager implementation.

The convolution ABI also now matches AOTI's optional tensor argument: bias is
passed as a pointer to a tensor handle, and `transposed` is an `int32_t`.
Previously the signature mismatch was hidden because lowered calls always
passed null bias. Both compiler lowering and runtime changes are needed;
existing PTEs that already separated the bias must be regenerated.

With these changes, all six observed operations in pretrained stage-4 SE
(pool, fc1, ReLU, fc2, Hardsigmoid, multiply) match BF16 eager exactly on the
diagnostic input. Eleven native convolution tests pass, including a two-output
channel BF16 cancellation test: `(1 + 0.00390625) - 1` must return `0.00390625`
with fused bias, rather than `0` after an intermediate BF16 rounding.
Six PTE/PTD runner cases also pass exactly: Conv1d/2d/3d in FP16 and BF16,
each exported with a dynamic batch bound and executed at batch 2. Run them with:

```bash
EXECUTORCH_MUSA_RUNNER="$RUNNER" $PYTHON -m unittest \
  backends.musa.tests.test_musa_runner -v
```

Without `EXECUTORCH_MUSA_RUNNER`, these runner tests are explicitly skipped.

Before the native Hardsigmoid change, the regenerated pretrained DeepLab BF16
full model still fails the default
`atol=rtol=0.04` check at 2x3x64x64: 1,005 of 172,032 values fail
`torch.testing.assert_close`, down from 16,419 with native Hardswish alone.
The maximum absolute error over all logits is `0.125`; all 8,192 pixel labels
match eager. This is not a passing default-tolerance acceptance, and the
remaining dynamic shapes have not passed that stricter check.

The diagnostic script accepts `--input /path/to/input-0.bin` to reuse an
acceptance input with the specified dtype, batch, height, and width. Trailing
zero padding from the dynamic input allocator is allowed. With
`--include-classifier`, it also emits the ASPP, classifier, and final bilinear
upsampling outputs. `--classifier-operators` instead feeds the classifier with
the eager backbone output to isolate head-local error.
On the failing acceptance input, backbone stages 0–4 match exactly, the first
discrepancy is at stage 5 (`0.0625`), and the final backbone error is `0.078125`.
The full diagnostic's final logits reproduce the same `0.125` maximum and
`0.0289871` mean absolute error as the ordinary full-model runner.

Further stage-5 SE isolation finds exact pooling, fc1, ReLU, and fc2 outputs,
followed by a `0.00390625` Hardsigmoid difference and a `0.0078125` multiply
difference. TorchMUSA's native Hardsigmoid uses muDNN with alpha `0.166667`
and beta `0.5`, rather than decomposed `clamp(x + 3, 0, 6) / 6`. At BF16
rounding boundaries, these are not interchangeable: for example, input
`0.85546875` produces eager `0.64453125` versus decomposed `0.640625`.
The backend now preserves native Hardsigmoid through a dedicated custom op
and shared unary SlimTensor shim, using the eager implementation's parameters.
All four native activation tests pass. The runner regression covers those
boundary inputs in FP32/FP16/BF16 and batches 1/2/3, with exact eager equality
for all nine runs. This operator result does not alone establish full-model
accuracy.

With native Hardswish, fused forward-convolution bias, and native Hardsigmoid
together, the regenerated pretrained DeepLab BF16 PTE/PTD passes the default
`atol=rtol=0.04` without overrides. One dynamic PTE passes two fresh runner
executions for each of these shapes:

| Input shape | Maximum absolute logit error (both runs) |
| --- | ---: |
| 2x3x64x64 | 0 |
| 1x3x48x54 | 0.015625 |
| 3x3x80x80 | 0 |
| 2x3x79x71 | 0.03125 |

An additional readback of all eight saved outputs verifies exact segmentation
labels for all 82,404 pixels, including the repeated executions.
After these precision fixes, the complete 38-test MUSA AOTI Python suite also
passes, including the existing convolution, normalization, KV-cache, SDPA,
Transformer, EfficientNet, DeepLab, and ViT compile regressions.

These checks use the official pretrained weights with generated inputs and
MUSA eager as the reference; they are not dataset-level segmentation accuracy
or performance measurements. Reproduce the stricter acceptance with:

```bash
$PYTHON examples/musa/aoti_model_acceptance.py \
  --model deeplabv3_mobilenet_v3_large --dtype bf16 \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --batch-size 2 --dynamic-batch-max 3 --dynamic-spatial-max 80 \
  --runner-repeats 2 --runner "$RUNNER" \
  --output-dir /tmp/executorch-musa-acceptance/deeplab_bf16_native
```

```bash
$PYTHON examples/musa/aoti_backbone_diagnostics.py \
  --dtype bf16 --batch-size 2 --height 64 --width 64 \
  --weights /path/to/deeplabv3_mobilenet_v3_large-fc3c493d.pth \
  --runner "$RUNNER" \
  --output-dir /tmp/executorch-musa-acceptance/deeplab_backbone_bf16
```

The BatchNorm decomposition now also covers direct `native_batch_norm`,
`_native_batch_norm_legit`, `_native_batch_norm_legit_no_training`, and
`_native_batch_norm_legit_functional` inference graphs, plus
`_batch_norm_no_update`. Their empty saved-stat/reserve-space outputs and
functional running-stat outputs are preserved; the complete 38-test MUSA AOTI
Python suite passes on the current MUSA environment. A `BatchNorm2d` module with
`track_running_stats=False` also passes a complete PTE/PTD runner check with
maximum FP32 error `2.38e-7`.

A random-weight ViT-B/16 with one encoder block also passes FP32 export and
two fresh `executor_runner` executions at 1x3x224x224 with maximum error `0`.
This provides a runner-level smoke test for the attention and LayerNorm path;
full-depth ViT coverage is tracked separately above.
