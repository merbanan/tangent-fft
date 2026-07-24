# Intel Core i7-6500U benchmark

Date: 2026-07-24

## Host and methodology

The remote host is a two-core/four-thread Intel Core i7-6500U (Skylake,
family 6 model 78), with AVX2 and FMA, 32 KiB L1 data and instruction caches
per core, 256 KiB L2 per core, and 4 MiB shared L3. It ran Ubuntu 24.04 with
GCC 13.3 and NASM 2.16.03. The machine was on AC power, used the `powersave`
HWP governor, and reported 43--48 C during the recorded runs.

The laptop was doing unrelated interactive work. To keep that from turning
one lucky pass into the result:

- all tests were pinned to CPU 2 or CPU 3, which are on different physical
  cores;
- the precise harness serializes invariant-TSC reads, warms each size, and
  takes the median of 31 batches;
- the lane4 harness was repeated six times, three on each physical core;
- the broader tangent harness was repeated four times;
- the complete wall-clock harness was warmed and repeated three times with a
  100 ms target per algorithm and size;
- the gather microbenchmark was repeated four times;
- the FFmpeg `vinsertf128` experiment used separately linked binaries,
  alternating run order, followed by ten same-CPU pairs per size.

The complete forward/inverse correctness suite passed before measurement.
The `vinsertf128` candidate also passed the same suite.

## Lane4 versus FFmpeg

These are medians of the six independent cycle-harness results. The best AVX
entry is selected independently at each size. `gain` is relative to native
FFmpeg AVTX and includes the same public in-place API boundary.

| N | best lane4 AVX | cycles | FFmpeg cycles | AVX gain | lane4 SSE cycles | SSE gain |
|---:|---|---:|---:|---:|---:|---:|
| 16 | AVX-FMA | 46.427 | 92.231 | 49.7% | 71.059 | 23.0% |
| 32 | AVX2 | 77.331 | 140.977 | 45.1% | 144.772 | -2.7% |
| 64 | AVX2-FMA | 155.707 | 284.525 | 45.3% | 294.317 | -3.4% |
| 128 | AVX2-FMA | 305.356 | 511.212 | 40.3% | 640.389 | -25.3% |
| 256 | AVX2-FMA | 620.549 | 1,102.490 | 43.7% | 1,299.697 | -17.9% |
| 512 | AVX2-FMA | 1,431.290 | 2,346.023 | 39.0% | 2,981.628 | -27.1% |
| 1024 | AVX-FMA | 3,298.692 | 5,470.529 | 39.7% | 6,193.070 | -13.2% |
| 2048 | AVX2-FMA | 8,996.316 | 12,406.971 | 27.5% | 14,844.281 | -19.6% |
| 4096 | AVX2-FMA | 21,352.408 | 30,706.416 | 30.5% | 32,493.117 | -5.8% |
| 8192 | AVX-FMA | 49,059.215 | 71,598.027 | 31.5% | 75,485.277 | -5.4% |

The important conclusion survives the loaded host: a lane4 AVX variant beats
native FFmpeg at every size, by 27.5--49.7%. Unlike the Zen 2 result, lane4
SSE only beats native FFmpeg at N=16 on Skylake.

Full per-pass results and spreads are in
`intel-lane4-cycles-p*-cpu*.txt`, `intel-lane4-summary.txt`, and
`intel-lane4-vs-ffmpeg.txt`.

## Tangent assembly versus FFmpeg

| N | tangent x86 cycles | FFmpeg cycles | tangent gain |
|---:|---:|---:|---:|
| 16 | 67.480 | 92.203 | 26.8% |
| 32 | 133.730 | 147.701 | 9.5% |
| 64 | 258.041 | 285.353 | 9.6% |
| 128 | 560.913 | 510.784 | -9.8% |
| 256 | 1,210.676 | 1,159.936 | -4.4% |
| 512 | 2,832.756 | 2,383.605 | -18.8% |
| 1024 | 6,326.064 | 5,515.812 | -14.7% |
| 2048 | 13,822.885 | 12,507.436 | -10.5% |
| 4096 | 32,629.447 | 30,509.271 | -6.9% |
| 8192 | 75,494.246 | 71,503.219 | -5.6% |

Tangent assembly wins at 16--64 but FFmpeg wins from 128 onward on this CPU.

## AVX2 gather crossover

The exact four-complex permutation microbenchmark compares `vgatherdpd`
against scalar `vmovq`/`vpinsrq` loads. Values are medians of four runs:

| N | gather cycles | scalar cycles | gather change |
|---:|---:|---:|---:|
| 16 | 21.812 | 19.620 | 11.2% slower |
| 32 | 39.900 | 37.024 | 7.8% slower |
| 64 | 76.293 | 72.317 | 5.5% slower |
| 128 | 149.210 | 141.637 | 5.3% slower |
| 256 | 292.472 | 284.437 | 2.8% slower |
| 512 | 585.554 | 609.599 | 3.9% faster |
| 1024 | 1,157.802 | 1,178.290 | 1.7% faster |
| 2048 | 2,300.082 | 2,407.394 | 4.5% faster |
| 4096 | 5,437.767 | 5,813.851 | 6.5% faster |
| 8192 | 20,764.210 | 23,312.331 | 10.9% faster |

This is the clearest Intel-specific follow-up: a Skylake-gated gather path
from N=512 upward is plausible. It still needs an end-to-end FFT candidate
because permutation is only one part of execution.

## `vinsertf128` A/B

FFmpeg's low-half FFT8 duplication was changed from `vperm2f128` to
`vinsertf128`. Ten paired runs per size gave:

| N | paired median change | candidate wins |
|---:|---:|---:|
| 64 | +0.05% | 5/10 |
| 128 | -0.45% | 6/10 |
| 256 | +0.56% | 3/10 |
| 512 | +0.26% | 4/10 |
| 1024 | -1.26% | 7/10 |
| 2048 | -1.22% | 7/10 |
| 4096 | +0.32% | 5/10 |
| 8192 | +0.65% | 5/10 |

Negative is faster. N=1024 and 2048 show a plausible roughly 1.2% gain, but
the remaining sizes are mixed and background interference widened some
ranges substantially. This does not support an unconditional replacement.
The reproducible candidate is `analysis/ffmpeg_vinsertf128.patch`.

## Complete results

- `intel-wall-summary.csv`: all algorithms and sizes, median/minimum/maximum
  across three complete wall-clock runs.
- `intel-tangent-summary.txt`: all nine primary x86 comparison entries with
  medians and spreads across four runs.
- `intel-lane4-summary.txt`: all lane4 ISA variants and FFmpeg across six
  runs.
- `intel-system.txt`: complete CPU, compiler, power, temperature, and source
  archive metadata.

The cycle summaries should be preferred over wall-clock values at small
sizes. The raw files are intentionally retained so results can be
recalculated without rerunning the loaded laptop.
