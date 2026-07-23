# Hardware-width top-level FFTs

## Result

Two top-level decompositions were retained:

- `lane8-avx2-fma` factors `N=8M` and evaluates eight residue transforms in
  split-complex YMM registers. Its planner is scalar C, while every
  execution load, shuffle, butterfly, FMA, transpose, and store is handwritten
  FFmpeg-style NASM in `lane8_avx.asm`.
- `hw-sse-auto` uses the two-complex interleaved assembly schedule through
  `N=64` and the four-complex split assembly schedule above that point. This
  is the measured crossover on the Ryzen 9 3900X.

The AVX experiment establishes an important negative and positive result.
The split representation reaches the direct-form SIMD instruction bound for
the dominant complex rotation, but converting the public array-of-structures
input into that representation costs enough that the existing interleaved
lane4 AVX schedule remains 8--33% faster through 8192. Lane8 is nevertheless
1.4--2.0 times faster than FFmpeg AVTX from 64 through 8192 on the recorded
run. It remains a public benchmark entry rather than silently replacing a
faster implementation.

## AVX factorization

Let `N=8M` and write

```text
n = 8m + r,       0 <= r < 8
k = q + Mp,       0 <= p < 8.
```

For `W_N=exp(-2*pi*i/N)`, define

```text
F_r[q] = sum(m=0..M-1) x[8m+r] W_M^(mq).
```

Then

```text
X[q+Mp] = sum(r=0..7) F_r[q] W_N^(rq) W_8^(rp).
```

The implementation therefore evaluates eight length-`M` FFTs, multiplies
seven residue columns by `W_N^(rq)`, and performs one final FFT8 for each
`q`. One work row contains

```text
ymm re = [F_0.re, F_1.re, ... F_7.re]
ymm im = [F_0.im, F_1.im, ... F_7.im].
```

The inner transforms use digit-reversed FFT4/FFT8 leaves and block-major
radix-4 stages. A mixed-radix permutation and all float coefficient streams
are created outside the timed region.

### Complex-rotation kernel

For eight values and eight replicated coefficients,

```text
out.re = in.re*wr - in.im*wi
out.im = in.re*wi + in.im*wr.
```

The split AVX2/FMA kernel emits:

```text
2 x vmulps
2 x FMA
```

The old interleaved layout needs two lane swaps, two multiplies, and two
`vfmaddsubps` instructions for the same eight complex values. Lane8 thus
reduces this local kernel from six vector instructions to four. Four is the
lower bound for this direct split-complex/FMA dataflow: two independent
outputs each need both an ordinary product and a cross product. This is a
local machine-model bound, not a proof of a global DFT arithmetic lower
bound.

### Full-width finish

The first finish prototype evaluated only four frequencies in XMM registers.
At `N=8192` it took about 29,049 cycles by itself, and the complete transform
took 12.48 microseconds.

The retained finalizer:

1. reads eight split work rows;
2. transposes the real and imaginary 8-by-8 matrices;
3. applies seven lane-major twiddle streams;
4. evaluates eight FFT8s in parallel in all sixteen YMM registers;
5. interleaves and stores eight contiguous frequencies per output block.

This reduced the measured finish to about 17,515 cycles and the complete
8192-point transform to 10.08--10.68 microseconds. A four-frequency fallback
handles `N=32`. The 16-point API entry reuses the existing AVX2/FMA tangent
assembly leaf.

## SSE factorization

SSE has the same number of architectural vector registers but half the vector
width. Two layouts are useful:

```text
lane2: xmm = [F0.re, F0.im, F1.re, F1.im]
lane4: xmm re + xmm im = four split-complex transforms
```

Lane2 has less transpose and table overhead for the smallest transforms.
Lane4 amortizes loop and address work across four transforms and wins after
the measured small-size crossover. `hw-sse-auto` therefore dispatches:

```text
16 <= N <= 64: lane2-sse
N >= 128:      lane4-sse
```

The retained lane2 upper stage now issues the three independent complex
rotations through three XMM scratch registers before reusing those registers
for the radix-4 outputs. This changes neither arithmetic nor table traffic,
but removes the artificial dependency created by reusing one scratch
register for all three rotations. It is a handwritten SSE1 schedule; later
SSE API variants share it because they add no primitive needed by this
kernel.

The split SSE generic rotation uses four `mulps` and two `addps/subps`
instructions for four complex values, exactly the direct-form scalar
multiplication/addition count lifted into XMM. SSE has no FMA with which to
shorten that dependency graph.

An upper radix-8 SSE stage was not retained. Eight split-complex rows consume
all sixteen XMM registers before twiddle temporaries or address-independent
outputs exist, so the ostensibly lower-pass schedule creates spills. The
register-balanced radix-4 stage is faster.

## What “lower bound” means here

No proven general lower bound is known that says an arbitrary complex DFT
must use the operation count of modified split radix. The tangent/modified
split-radix formula reported by this project is an achieved upper bound:

```text
(34/9) N lg N - (124/27) N - 2 lg N
- (2/9)(-1)^lgN lg N + (16/27)(-1)^lgN + 8.
```

Lane8 does not claim that scalar count. It reaches a lower instruction count
for the machine's dominant vector complex-product primitive, while accepting
ordinary radix-4 arithmetic elsewhere. This distinction matters: the first
vertical modified-split-radix prototype executed fewer scalar operations but
was almost twice as slow because its global passes, irregular subtransform
sizes, and data movement did not match the machine.

## Ryzen 9 3900X measurements

Median microseconds from the checked-in `benchmark.csv`; plan construction is
excluded and FFmpeg includes the copies required by the common in-place API:

| N | lane4 AVX2/FMA | lane8 AVX2/FMA | hw SSE | FFmpeg AVTX |
|---:|---:|---:|---:|---:|
| 16 | 0.040 | 0.040 | 0.040 | 0.060 |
| 32 | 0.040 | 0.050 | 0.050 | 0.060 |
| 64 | 0.060 | 0.070 | 0.080 | 0.100 |
| 128 | 0.090 | 0.120 | 0.160 | 0.160 |
| 256 | 0.180 | 0.230 | 0.320 | 0.340 |
| 512 | 0.370 | 0.450 | 0.710 | 0.690 |
| 1024 | 0.790 | 0.980 | 1.480 | 1.460 |
| 2048 | 1.820 | 1.990 | 3.350 | 3.220 |
| 4096 | 4.190 | 4.830 | 7.220 | 7.540 |
| 8192 | 9.920 | 10.680 | 16.391 | 21.371 |

Timer quantization is significant below 128. `make lane8-profile` reports
serialized-TSC base, upper-stage, and finish costs separately.

## Reproduction

```sh
make
make test
make bench
make lane8-profile
```

The implementation uses only single-precision `ps` instructions and float
tables. Wider C types are restricted to sizes, counters, the correctness
reference, and timers.
