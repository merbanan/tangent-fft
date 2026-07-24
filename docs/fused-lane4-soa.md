# Fused lane4-SoA split-radix/Stockham kernel

The AArch64 NEON and x86 SSE fixed-size codelets in this project are the
leaves of a **fused lane4-SoA split-radix/Stockham kernel with dedicated
16-, 32-, and 64-point leaves**. The AVX variants use the same fused schedule
but retain their existing interleaved-complex register layout, so they are not
literally SoA implementations.

“Lane4-SoA” has a precise physical meaning. Four complex values are held in
two vectors:

```text
re = [z0.re, z1.re, z2.re, z3.re]
im = [z0.im, z1.im, z2.im, z3.im]
```

On NEON and SSE this occupies two 128-bit registers. It is not a literal port
of the AVX layout. Keeping real and imaginary components separate makes a
four-way complex product four vector arithmetic instructions and removes the
`REV64`/sign-mask sequence needed by the interleaved lane2 representation.

## Fusion and autosort

The codelets fuse decomposition, twiddle multiplication, transposition, and
the final store:

- N=16 performs two radix-4 networks, the intervening three complex products,
  and a 4x4 transpose entirely in registers.
- N=32 performs a radix-2 split into two N=16 leaves. `ZIP`/unpack operations
  interleave the even and odd spectra while writing natural-order output.
- N=64 performs a four-way DIF split into four N=16 leaves. Four 4x4
  transposes perform the Stockham autosort while writing natural-order output.

The output permutation is therefore part of the final vector store. There is
no separate bit-reversal pass in any of the three leaves.

The larger-transform design uses the same leaves below a flat split schedule
and fuses each upper twiddle/combine with its Stockham destination write.
“Split-radix/Stockham” names that hybrid schedule; it does not claim that the
register-level N=16 arithmetic network itself is the classical split-radix
recurrence.

## AArch64 execution layout

The NEON implementation stores each four-frequency real/imaginary coefficient
pair adjacently and orders the three pairs exactly as the fused finish
consumes them. Three post-index `LDP Q` operations therefore load an entire
root packet without indexed address construction.

The 4x4 transpose deliberately exposes the native destination order produced
by its eight `TRN1`/`TRN2` instructions. Radix and store consumers use those
registers directly instead of normalizing them with copies. The FFT8 gather
leaf likewise loads eight permutation indices as four pairs.

These choices came from the complete FFmpeg ARM/AArch64 audit in
[`arm-optimization-audit.md`](arm-optimization-audit.md).

## Architecture implementations

| Architecture entry | N=16 | N=32 | N=64 |
|---|---|---|---|
| AArch64 NEON | register fused | two FFT16 leaves + ZIP autosort | radix-4 split + four FFT16 leaves + 4x4 autosort |
| SSE through SSE4.2 | shared SSE1-compatible fixed leaf | fixed-size FFT8/finish pipeline | fixed-size FFT4/stage/finish pipeline |
| AVX and AVX2 | register fused | register/fixed fused | fixed-size radix-4 pipeline |
| AVX+FMA and AVX2+FMA | register fused | register/fixed fused | fixed-size FMA radix-4 pipeline |

Later SSE feature levels deliberately share the SSE body: SSE2, SSE3, SSSE3,
SSE4.1, and SSE4.2 do not add an instruction that improves the split-complex
single-precision hot path. They remain separate public benchmark entries so
dispatch and call overhead are measured consistently.

## x86 benchmark

The following values are median execution times in microseconds on an AMD
Ryzen 9 3900X. `FFmpeg native` is FFmpeg's runtime-selected AVTX path;
`FFmpeg SSE` forces its SSE implementation. Lower is better.

| N | SSE | SSE2 | SSE3 | SSSE3 | SSE4.1 | SSE4.2 | AVX | AVX-FMA | AVX2 | AVX2-FMA | FFmpeg native | FFmpeg SSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 | 0.07 |
| 32 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 | 0.12 |
| 64 | 0.08 | 0.09 | 0.08 | 0.08 | 0.08 | 0.08 | 0.06 | 0.06 | 0.06 | 0.06 | 0.10 | 0.26 |
| 128 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.16 | 0.09 | 0.10 | 0.09 | 0.09 | 0.16 | 0.59 |
| 256 | 0.31 | 0.32 | 0.31 | 0.32 | 0.32 | 0.32 | 0.17 | 0.18 | 0.18 | 0.19 | 0.33 | 1.30 |
| 512 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.69 | 0.37 | 0.37 | 0.36 | 0.37 | 0.68 | 2.78 |
| 1024 | 1.46 | 1.44 | 1.46 | 1.45 | 1.44 | 1.44 | 0.80 | 0.81 | 0.80 | 0.80 | 1.44 | 6.22 |
| 2048 | 3.32 | 3.31 | 3.31 | 3.31 | 3.24 | 3.25 | 1.79 | 1.81 | 1.79 | 1.77 | 3.28 | 14.20 |
| 4096 | 7.43 | 7.43 | 7.43 | 7.43 | 7.43 | 7.42 | 4.48 | 4.60 | 4.48 | 4.46 | 7.80 | 31.26 |
| 8192 | 16.80 | 16.82 | 16.79 | 16.17 | 16.23 | 16.27 | 10.18 | 9.86 | 10.02 | 9.89 | 20.45 | 70.33 |

The complete machine-readable results, including sample counts, minima,
checksums, and the scalar/reference implementations, are in
[`benchmark-fused-lane4.csv`](../benchmark-fused-lane4.csv).

### Pre-fusion lane4 comparison

There is no separate public x86 algorithm named `lane4-SoA`: the SSE lane4
implementation already used `re[4]` and `im[4]` vectors. The new fixed leaves
were installed behind the existing `lane4-*` entries. Therefore, the valid
comparison is the committed pre-fusion implementation against the current
fused-leaf implementation.

The following controlled runs used the same Ryzen 9 3900X core, executable
options, and 100 ms timing budget. Values are median microseconds. AVX2-FMA is
included as the representative 256-bit path, although its layout is
interleaved rather than SoA.

| N | lane4 SSE before | fused lane4-SoA SSE | change | lane4 AVX2-FMA before | fused-schedule AVX2-FMA |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.04 | 0.03 | 25% less time | 0.03 | 0.03 |
| 32 | 0.05 | 0.05 | no measured change | 0.04 | 0.04 |
| 64 | 0.09 | 0.08 | 11% less time | 0.06 | 0.06 |

Only the N=16, N=32, and N=64 public dispatch paths changed. At N >= 128 the
same generic upper path is measured in both revisions, so small differences
between separate runs are measurement noise rather than a lane4-SoA
algorithmic speedup.

## Correctness

`make aarch64-qemu-test` runs the production NEON planner and assembly in a
freestanding AArch64 binary. It checks the complete fused lane4 path at every
power of two from N=16 through N=8192 against a double-precision reference
FFT.

`make test` checks every available x86 SIMD entry against a long-double direct
DFT through N=512 and cross-checks power-of-two sizes through `2^22`.

The FFmpeg source changes, exact function-size deltas, and AArch64 instruction
traces through N=8192 are recorded in
[`ffmpeg-aarch64-optimization-results.md`](ffmpeg-aarch64-optimization-results.md).
