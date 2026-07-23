# FFmpeg x86 FFT TODO investigation

Date: 2026-07-23

## Scope

This report investigates the six TODO items at the top of FFmpeg's
`libavutil/x86/tx_float.asm` and applies the useful ideas to
`tangent-x86-asm`. The target machine was:

- AMD Ryzen 9 3900X (Zen 2), AVX2 and FMA, no AVX-512;
- GCC 15.2.0;
- NASM 2.16.03;
- LLVM MCA 20.1.8 with `-mcpu=znver2`.

All execution benchmarks were pinned to CPU 2. A complete unreported pass was
run first to warm the CPU, followed by the reported median pass. The harness
excludes plan construction and includes the same public in-place API costs for
each transform. Its 10 ns display resolution limits conclusions for the
smallest transforms, so a change was retained only when it also had a clear
structural or instruction-count advantage.

The instruction-model results can be reproduced with:

```sh
llvm-mca -mcpu=znver2 -iterations=100 analysis/ffmpeg_todo_mca.s
```

The final warmed benchmark was produced with:

```sh
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 100 >/dev/null
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark.csv
```

## Results summary

| FFmpeg TODO | Finding | Decision |
|---|---|---|
| Carry registers from smaller transforms | Large win at 64 points | Implemented fixed FFT64 |
| `vinsertf128` versus `vperm2f128` duplication | Better Zen 2 model, inconclusive whole-FFT timing | Kept upstream FFmpeg form |
| Faster FFT8 | AVX2 gather version was substantially slower on Zen 2 | Rejected gather; retained scalar load/insert |
| XORs versus blends/addsubs | One instruction removed from every final rotation; two more in high tangent products | Implemented |
| Shuffles versus blends | Equivalent when lanes already align; most shuffles also permute | Retained existing blends only where valid |
| AVX-512 split-radix | Promising design, but unavailable on this host | Design recorded; no untestable kernel added |

The retained changes preserve the previous worst relative error for
`tangent-x86-asm`, `1.413e-07`, and pass direct-DFT validation through 512 plus
cross-checks through `2^22`.

## 1. Carry registers from smaller transforms

### What FFmpeg means

FFmpeg's split-radix path builds larger transforms out of fixed 8-, 16-, and
32-point kernels. A child result is normally written to `out` and loaded again
by its parent. If the parent immediately consumes that child and enough vector
registers remain, keeping it live removes a store/load pair for each vector.

The existing tangent FFT32 already follows this rule: its 16-point even child
remains in registers while only the two 8-point children use scratch.

### Implemented FFT64

The new `tangent_x86_gather_fft64_normal` is a single fixed kernel. It:

1. gathers and computes the FFT32 child;
2. computes and stores the first FFT16 child;
3. computes the final FFT16 child into `ymm0` through `ymm3`;
4. consumes those four registers directly in the FFT64 combine.

This removes four 32-byte stores and four 32-byte reloads, or 256 bytes of
scratch traffic, while also collapsing five C-to-assembly dispatches into one.
This is the concrete eight-load/store saving anticipated by the FFmpeg TODO.

At 64 points the median time changed from 0.110 µs to 0.080 µs, a 27% reduction.
FFmpeg AVTX takes 0.100 µs on the same host, so this changes the result from 10%
slower than FFmpeg to 20% faster.

Extending the same technique mechanically is not always profitable. An FFT128
child can occupy eight YMM registers, leaving too few registers for four input
streams, factors, and combine temporaries. A larger fixed kernel therefore
needs a new lifetime schedule, not just more unrolling.

## 2. `vinsertf128` versus `vperm2f128` duplication

The viable one-instruction substitution in FFmpeg's `FFT8_AVX` is:

```asm
vperm2f128 ymm3, ymm3, ymm3, 0x00
```

versus:

```asm
vinsertf128 ymm3, ymm3, xmm3, 1
```

Both duplicate the low 128-bit half. LLVM MCA's Zen 2 model reports:

| Instruction | µops | Latency | Reciprocal throughput |
|---|---:|---:|---:|
| `vperm2f128` | 1 | 3 cycles | 1.00 cycle |
| `vinsertf128` | 1 | 2 cycles | 0.33 cycle |

Three alternating whole-transform A/B passes produced no consistent timing
difference through 8192; changes at larger sizes moved in both directions and
stayed within normal run-to-run noise. The high-half duplicate cannot use the
same trick: it would need an extraction followed by insertion and would add an
instruction.

The vendored FFmpeg source therefore retains its original `vperm2f128`.
`tangent-x86-asm` already uses `vinsertf128` where it naturally builds a YMM
register from two XMM results.

## 3. Faster FFT8

FFmpeg's AVX FFT8 arithmetic is already a 20-instruction kernel. The remaining
opportunities are mostly critical-path and input-layout changes rather than
obvious instruction deletion.

One input-side variant was tested: gather four permuted complex values with
`vgatherdpd`, then extract the two XMM halves. For an eight-point leaf this
reduces the static gather sequence from sixteen scalar load/insert instructions
to eight vector gather/setup/extract instructions.

The experimental replacement for each group of four values was:

```asm
vpcmpeqd     ymm15, ymm15, ymm15
vmovdqu      xmm14, [permutation]
vgatherdpd   ymm0, [input + xmm14*8], ymm15
vextractf128 xmm1, ymm0, 1
```

It was much slower on Zen 2:

| Transform size | Scalar loads/inserts | AVX2 gathers | Regression |
|---:|---:|---:|---:|
| 32 | 0.050 µs | 0.060 µs | 20% |
| 64 | 0.080 µs | 0.110 µs | 38% |
| 128 | 0.200 µs | 0.260 µs | 30% |
| 256 | 0.380 µs | 0.470 µs | 24% |

This agrees with FFmpeg's own dispatch metadata, which excludes its AVX2
split-radix codelet on CPUs marked `AV_CPU_FLAG_SLOW_GATHER`. The experiment was
reverted. On this CPU, fewer architectural instructions do not compensate for
the microcoded gather cost.

A future arithmetic FFT8 should be tested as an independent fixed kernel. The
most plausible candidate is a slightly longer two-FFT4 decomposition that
avoids the two cross-128-bit `vperm2f128` dependencies. It may reduce latency
on a shuffle-bound core even though it cannot beat the current instruction
count. It should not replace the present kernel without per-microarchitecture
dispatch and measurements.

## 4. Replace XORs with blends and addsubs

### Final split-radix rotation

The common final rotation originally used:

```asm
vpermilps  difference, difference, 0xb1
vxorps     minus_i, difference, sign_odd
vxorps     plus_i,  difference, sign_even
vaddps     out1, u1, minus_i
vaddps     out3, u1, plus_i
```

It is now:

```asm
vpermilps  difference, difference, 0xb1
vxorps     negated, difference, sign_all
vaddsubps  out1, u1, negated
vaddsubps  out3, u1, difference
```

This removes one instruction and one sign-mask load per combine. The same
rewrite was applied to all S2 scaled butterflies.

LLVM MCA changes the modeled Zen 2 block from five instructions and 1.3-cycle
throughput to four instructions and 1.0-cycle throughput.

### High tangent product

For the `cot - i` half of a tangent product, the original pair of sign XORs can
be expressed directly by alternating FMA forms:

```asm
vfmsubadd213ps z,  factor, swapped_z
vfmaddsub213ps zp, factor, swapped_zp
```

This reduces that branch from six vector instructions to four. The `1 + i*tan`
half cannot use the analogous `231` form without changing the subtraction
ordering around the old destination. It would still need sign correction, so
that branch was left unchanged.

Together these changes improve the larger tangent transforms by roughly 2–5%
in the final benchmark, with no accuracy change.

## 5. Replace shuffles with blends

`vblendps` is useful only when the desired values already occupy corresponding
lane positions. `tangent-x86-asm` uses it in `REG_BASE` and `REG_Q1` to select
the lower half from a sum and the upper half from a difference.

Replacing those blends with equivalent `vshufps ..., 0xe4` sequences produced
identical 0.010 µs-resolution results from 16 through 256. LLVM MCA models both
as one µop, one-cycle latency, and 0.5-cycle reciprocal throughput on Zen 2.
The blend form was retained because it expresses the operation directly.

Most shuffles in FFmpeg's FFT8 are not blend candidates: they also duplicate
or reorder lanes. Replacing those requires changing the surrounding data
layout and must be evaluated as a complete kernel, not as a mnemonic swap.

## 6. AVX-512 split-radix

This Ryzen 9 3900X has no AVX-512, so an AVX-512 implementation could be
assembled but not correctness-tested or benchmarked. Adding such a kernel here
would violate the project's rule that optimized paths must be executable and
validated on the development host.

A practical FFmpeg implementation would require:

1. an `INIT_ZMM avx512` split-radix family and x86 codelet declarations;
2. eight complex values per ZMM, doubling combine width;
3. ZMM twiddle tables or a load/rearrangement scheme compatible with the
   existing tables;
4. `vshufi64x2`, `vpermt2ps`, or equivalent replacements for the current
   128-bit-half `vperm2f128` layout operations;
5. use of the 32-register file to retain larger child transforms and reduce
   scratch traffic;
6. `TX_DEF` entries gated by `AV_CPU_FLAG_AVX512` or the narrower feature set
   actually used;
7. FFmpeg checkasm validation for forward/inverse, in-place/out-of-place, and
   preshuffled variants;
8. size- and CPU-specific dispatch because AVX-512 frequency effects can make
   a wider kernel slower for the important 16–8192 range.

The most promising first kernel is fixed FFT64: eight ZMM registers can hold
the complete output, and the 32-register file leaves room for child state and
combine temporaries. Larger synthesis stages can then process eight complex
values per iteration.

## Final benchmark

Median microseconds after the retained changes:

| N | tangent-x86-asm | FFmpeg AVTX | Tangent relative to FFmpeg |
|---:|---:|---:|---:|
| 16 | 0.050 | 0.050 | tie |
| 32 | 0.050 | 0.060 | 16.7% faster |
| 64 | 0.080 | 0.100 | 20.0% faster |
| 128 | 0.200 | 0.170 | 17.6% slower |
| 256 | 0.380 | 0.340 | 11.8% slower |
| 512 | 0.720 | 0.690 | 4.3% slower |
| 1024 | 1.480 | 1.470 | 0.7% slower |
| 2048 | 3.090 | 3.250 | 4.9% faster |
| 4096 | 7.040 | 7.600 | 7.4% faster |
| 8192 | 18.500 | 21.310 | 13.2% faster |

The complete machine-readable run is in `benchmark.csv`.
