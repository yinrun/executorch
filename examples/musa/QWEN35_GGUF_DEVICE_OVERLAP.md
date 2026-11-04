# Global device-engine overlap

This extends the [wait-policy global trace](QWEN35_GGUF_WAIT_POLICY.md) beyond
the model PID. It is read-only analysis of the same completed capture, not a
new model benchmark or a change to GPU scheduling. The qualified runtime,
model, clocks and other applications are unchanged. Stable 10 decode tokens/s
remains unproved.

## Scope and method

The capture contains four complete text generations: auto/blocking warmups
followed by auto/blocking measurements, with the original 28-token prompt,
88 decode steps and EOS. It includes loading, serial prefill, decode and output
checks. Actual runtime/driver flags and native library maps were audited in
the preceding experiment. The native trace PID and GPU ID select the model
and physical device; other exported PIDs on that same GPU are kept separate.

SQLite `all_events` is joined with `event_names` for the device-engine track.
The analysis merges intervals rather than summing overlapping event labels.
Each step's window starts after its preparation synchronization and ends at
its completion synchronization. Engine intervals crossing either window edge
are clipped, not dropped. Other-process Compute, Geometry, Pixel and any other
exported engines are analyzed separately and as a combined union.

An independent checker uses an event-count sweep over raw start/end points,
not the analyzer's interval-merging implementation. It reproduces all metrics
for **464 execution windows** and recomputes the two 88-step measured-decode
summaries. The maximum raw engine interval in these windows is 18.392 ms.
Source trace/report hashes are checked before analysis and verification.

## Measured-decode interval coverage

| Median per-step interval, ms | Auto | Blocking |
| --- | ---: | ---: |
| Model host execution window | 138.061 | 138.530 |
| Model Compute union | 113.560 | 112.373 |
| Other-process Compute union | 2.249 | 2.242 |
| Other-process Geometry/Pixel union | 104.135 | 104.499 |
| All other-process engines union | 106.336 | 106.745 |
| Model Compute overlapping other engines | 89.467 | 89.315 |
| Other engines without model Compute | 16.947 | 17.756 |
| No exported engine interval | 7.382 | 8.490 |

Each row is the median of a separately computed per-step metric. Medians are
not additive, and the Compute/graphics/combined rows overlap. The last row
does not prove that the hardware is idle: it describes the absence of an
exported engine interval in this capture, not comprehensive hardware counters.
Model 2D activity is included when calculating the all-engine union.

The model's Compute interval is therefore not the only recorded device work
during its execution windows. Most non-model coverage is graphics rather
than another exported Compute track. The analysis does not identify an
application from a PID, treat an open GPU descriptor as proof of utilization,
or terminate/reconfigure any application.

## What this does and does not establish

This is direct evidence of overlapping exported engine activity, stronger
than an idle-time GPU-utilization snapshot. It does **not** measure how many
MPs, memory channels or execution slots each process uses. Simultaneous
intervals do not mean that two engines occupy the entire GPU independently.
Nor do they establish that the other engine caused any particular kernel's
latency or the earlier 8--10 tok/s variation.

In particular, the roughly 89 ms overlap cannot be subtracted from token
latency as removable overhead. The approximately 17 ms other-only coverage
cannot all be attributed to contention either: the model may be preparing
work or waiting for dependencies. No exclusive-device throughput, causal
slowdown percentage or new normal decode rate is inferred from this trace.

Two retained exporter warnings concern inverted Compute/Pixel intervals
before the first native model event. The prior independent warning audit
proves neither overlaps any generation; exported rows contain no inverted
intervals. Their PIDs are absent from the warning text. This report retains
that limitation and does not claim complete pre-model device accounting.

The optimization objective remains model throughput with the original output
checks and numerical gate. Kernel candidates still need controlled paired
timings and full-model qualification. A quiet/exclusive-device experiment,
if separately arranged by the user, would be a different environment and
must be reported as such, not silently substituted for shared-desktop results.

## Evidence

Under ETSupport `artifacts/gguf-20261029`:

- `global-wait-policy-profile-v1`: original raw trace, launch and native report.
- `wait-policy-decision-v1.json`: original trajectory, flag and warning audit.
- `analyze_global_device_overlap_v1.py`, `global-device-overlap-v1.json`:
  all-PID engine unions and complete per-step window results.
- `audit_global_device_overlap_v1.py`, `global-device-overlap-audit-v1.json`:
  independent endpoint-sweep validation of all 464 windows.
