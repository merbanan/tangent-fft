# Tangent FFT assembly optimization

## Scope

This follow-up applies the useful FFmpeg and lane4 scheduling ideas to the
tangent FFT while preserving Bernstein's four scaled transform families.
All transform data, factors, and arithmetic remain binary32 `float`.

The target is the Ryzen 9 3900X used by the other repository reports, with
special attention to powers of two from 16 through 8192. Plan creation and
table construction are outside execution timing.

## Retained AVX2/FMA changes

### Fixed FFT16 through FFT256

FFT16 gathers every input before its first in-place store, making a direct
scratch-free transform safe. FFT32 and FFT64 use fixed register schedules;
FFT64 carries its final FFT16 child directly into the parent.

FFT128 and FFT256 now have complete fixed transform trees. Their leaves and
upper nodes execute in dependency order without the generic C level/family
dispatch, node-offset loads, or inner node-loop branches. Low tangent regions
use the `1+i*tan` FMA expansion while boundary vectors retain the general
complex product.

On the final warmed run this changes the important middle sizes to:

| N | Previous tangent | Final tangent | FFmpeg AVTX |
|---:|---:|---:|---:|
| 128 | 0.20–0.22 µs | 0.16 µs | 0.17 µs |
| 256 | 0.36–0.38 µs | 0.33 µs | 0.34 µs |

### Execution-ordered split factors

Normal upper nodes formerly loaded interleaved complex factors and issued
`vmovsldup` plus `vmovshdup`. The plan now also stores execution-ordered
real and imaginary factor vectors with every scalar duplicated into both
complex lanes. The kernel trades those two shuffles for one additional
linear load.

Long-batch A/B measurements were about 1.4% better at 1024 and 2048, close
to tied at 512 and 8192, and slightly better at 4096. The split table is
therefore retained.

### One AVX/SSE transition per transform

The generic scheduled path previously executed `vzeroupper` after every
homogeneous leaf or node batch. No legacy-SSE computation occurs between
those batches. The batch returns now preserve AVX state and one explicit
`vzeroupper` runs immediately before returning to C. This removed redundant
transition instructions and improved the 512-point path by roughly 1–2%.

### Result

The final forward wall-clock benchmark is in `benchmark.csv`. At its 0.01 µs
small-size resolution, tangent AVX2/FMA beats native FFmpeg at 16, 32, 1024,
2048, 4096, and 8192; it ties at 64 and remains slower at 128, 256, and 512:

| N | tangent-x86-asm | FFmpeg AVTX | Tangent relative |
|---:|---:|---:|---:|
| 16 | 0.040 µs | 0.060 µs | 33.3% faster |
| 32 | 0.060 µs | 0.070 µs | 14.3% faster |
| 64 | 0.100 µs | 0.100 µs | tie |
| 128 | 0.170 µs | 0.160 µs | 6.3% slower |
| 256 | 0.340 µs | 0.330 µs | 3.0% slower |
| 512 | 0.730 µs | 0.690 µs | 5.8% slower |
| 1024 | 1.420 µs | 1.480 µs | 4.1% faster |
| 2048 | 3.010 µs | 3.210 µs | 6.2% faster |
| 4096 | 6.940 µs | 7.650 µs | 9.3% faster |
| 8192 | 18.450 µs | 21.161 µs | 12.8% faster |

Sub-microsecond medians have coarse timer quantization. The final
`make tangent-cycles` run gives the more precise comparison:

| N | tangent cycles | FFmpeg cycles | Tangent relative |
|---:|---:|---:|---:|
| 16 | 70.4 | 130.3 | 46.0% faster |
| 32 | 149.6 | 139.4 | 7.3% slower |
| 64 | 250.7 | 316.9 | 20.9% faster |
| 128 | 570.4 | 569.8 | tied |
| 256 | 1206.6 | 1204.2 | tied |
| 512 | 2695.7 | 2551.4 | 5.7% slower |
| 1024 | 5285.7 | 5466.9 | 3.3% faster |
| 2048 | 11229.5 | 12305.5 | 8.7% faster |
| 4096 | 26159.1 | 29446.6 | 11.2% faster |
| 8192 | 70048.3 | 81154.0 | 13.7% faster |

The target of being clearly faster at every size is therefore not fully met:
32 and 512 remain the measured losses, while 128 and 256 are effectively
tied.

## SSE-family tangent implementation

`tangent_sse_stage.asm` is a legacy 128-bit assembly implementation, not an
intrinsics wrapper. It supplies:

- baseline SSE arithmetic using `shufps`, sign masks, adds, and multiplies;
- an SSE3 arithmetic family using `movsldup`, `movshdup`, and `addsubps`;
- fixed 4-, 8-, and 16-point tangent leaves for all N/S/S2/S4 families;
- homogeneous upper-node batches with two complex values per XMM;
- reduced tangent products for profitable larger S regions;
- optional leaf-local permutation gather.

The public SSE and SSE2 entries share the baseline body. SSE3, SSSE3, SSE4.1,
and SSE4.2 share the SSE3 body because the later revisions add no instruction
that improves this float dataflow. Every public entry has its own CPUID/API
boundary and is printed independently by the harness.

Whole-array permutation followed by local leaves is faster below 8192.
At 8192 and above, gathering each leaf immediately before executing it wins
by keeping its new input block hot, so dispatch switches there.

SSE3 is about 2.0–2.3 times slower than FFmpeg over the larger sizes. The
implementation demonstrates that the tangent arithmetic translates cleanly
to 128-bit SIMD, but it does not reach FFmpeg parity on this CPU. Lane4 SSE
remains the stronger 128-bit schedule because its four-real/four-imaginary
layout fills an XMM more naturally.

## Correctness and reproduction

The retained code passes:

- forward and normalized-inverse direct-DFT tests through 512;
- cross-checks against radix-2 through `2^22`;
- ASan and UBSan with the same full correctness range.

LeakSanitizer itself cannot run in the managed traced environment; the
ASan/UBSan run used `ASAN_OPTIONS=detect_leaks=0`.

Reproduce the primary measurements with:

```sh
make
make test
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark.csv
taskset -c 2 ./fft_harness --bench --inverse --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark-inverse.csv
make tangent-cycles
```
