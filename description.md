# The lane4 FFT algorithm

The repository now also contains an `N=8M` split-complex AVX2/FMA
decomposition and a size-adaptive SSE top-level path. Their mathematical
derivation, assembly dataflow, direct-form machine-operation bound, and
benchmarks are documented in
[`docs/hardware-top-level-fft.md`](docs/hardware-top-level-fft.md). Lane8 is
kept separate from lane4 because its eight-residue split layout makes a
different hardware tradeoff; it is not a rename of the algorithm described
below.

## 1. What the algorithm is

`lane4` is a single-precision complex FFT for power-of-two lengths. Its core
kernel computes the forward transform; a normalized inverse API reuses that
kernel through exact DFT index symmetry. It is not a new asymptotic FFT
identity: mathematically it is a
Cooley--Tukey decomposition of a length-`N` DFT into four length-`N/4`
transforms followed by `N/4` length-4 transforms.

What is distinctive is the implementation:

- the four inner transforms are assigned to the four complex-float lanes of
  one 256-bit vector;
- the original interleaved complex input already has the required lane
  layout, so the input does not need an initial global transpose;
- the inner transform is scheduled as mixed-radix register codelets followed
  by regular block-major radix-4 stages;
- the final twiddle multiplication, 4-by-4 transpose, and FFT4 are fused;
- small transforms and selected parent/child boundaries remain in registers;
- all permutations and twiddles are prepared in the reusable plan.

The optimized implementation is named `lane4-avx2-fma`. The AVX, AVX+FMA,
AVX2, and AVX2+FMA entries compile the same 256-bit source at explicit ISA
boundaries. Plain C retains a simpler radix-2 inner FFT. The 128-bit variants
use FFmpeg-style x86inc/NASM assembly with FFT4/FFT8 leaves, block-major
radix-4 stages, and an assembly transpose/twiddle/FFT4 finish.

All samples, roots, twiddles, finish factors, and arithmetic used by the
algorithm are `float`. Wider types appear only in the correctness reference
and benchmark timers.

## 2. Relation to prior literature

The mathematical basis is the general Cooley--Tukey factorization. Frigo and
Johnson describe a composite DFT as smaller DFTs arranged like a
two-dimensional transform with a transposed output in
[The Design and Implementation of FFTW3](https://fftw.org/fftw-paper-ieee.pdf).

The matrix view is also closely related to Bailey's
[four-step FFT](https://www.davidhbailey.com/dhbpapers/fftq.pdf), which
organizes a one-dimensional transform as column transforms, twiddle
multiplication, a transpose, and row transforms. Bailey's motivation was
unit-stride access and fewer passes through hierarchical memory. Lane4 uses
the same type of factorization with one matrix dimension fixed at four, then
fuses the small transpose into the final SIMD codelet.

The most direct SIMD literature connection is the short-vector/vector
recursion work:

- Franchetti, Karner, Kral, and Ueberhuber,
  [Architecture Independent Short Vector FFTs](https://users.ece.cmu.edu/~franzf/papers/icassp01.pdf);
- Franchetti,
  [A Portable Short Vector Version of FFTW](https://citeseerx.ist.psu.edu/document?doi=8abd4533d76a98e7b23b7020e78cc04dc9f90a64&repid=rep1&type=pdf).

That work describes vector codelets as replacing each scalar operation of a
DFT by the same operation on a short vector. In tensor notation this is
`DFT_M ⊗ I_v`: `v` independent transforms are evaluated together. Lane4's
inner transform is exactly the `v=4` case. The implementation also follows
the established FFTW principle of using planned, fixed-size register
codelets rather than expecting a generic loop or compiler to discover the
best dataflow.

[PFFFT](https://android.googlesource.com/platform/external/pffft/+/3b884b26c0c56795c165ddb8b666f442c777a247/pffft.h)
is a useful implementation precedent: it adapts a single-precision FFT to
four-way SIMD on SSE, AltiVec, and NEON and falls back to scalar execution.
Lane4 was independently implemented and does not copy PFFFT code.

Bernstein's
[The tangent FFT](https://cr.yp.to/arith/tangentfft-20070809.pdf) is an
important comparison in this repository, but it is not the source of the
lane4 factorization. Tangent FFT reduces the number of real operations with
scaled transforms. Lane4 deliberately accepts a conventional Cooley--Tukey
arithmetic count in exchange for more regular SIMD execution and memory
traffic.

Therefore the accurate classification is:

```text
mathematics:  radix-4 Cooley--Tukey / fixed-width four-step FFT
SIMD model:   short-vector FFT or DFT_M ⊗ I_4 vector recursion
new work:     the exact lane mapping, fused finish, codelets, tables,
              register carry, and measured x86 schedule in this repository
```

### Is lane4 a new FFT algorithm?

Not in the mathematical or academic sense. Its factorization, asymptotic
complexity, and short-vector execution model all have clear prior art. It
does not introduce a new DFT identity comparable to split radix, tangent FFT,
Rader, or Bluestein.

It is a new implementation and a specialized algorithmic schedule developed
for this repository. The fused finish and exact register/dataflow decisions
may be locally novel, but establishing research novelty would require a much
broader prior-art search and a formal comparison against other lane-sliced
FFT implementations. The defensible description is therefore “an optimized
lane4 Cooley--Tukey implementation,” not “a newly discovered FFT algorithm.”

## 3. Mathematical derivation

The forward unnormalised DFT is

```text
X[k] = sum(n=0..N-1) x[n] W_N^(n k)
W_N  = exp(-2 pi i / N).
```

Let `N=4M` and split both indices into two digits:

```text
n = 4m + r,       0 <= m < M, 0 <= r < 4
k = q + Mp,       0 <= q < M, 0 <= p < 4.
```

Substitution gives

```text
W_N^((4m+r)(q+Mp))
 = W_N^(4mq) W_N^(4mMp) W_N^(rq) W_N^(rMp)
 = W_M^(mq) 1             W_N^(rq) W_4^(rp).
```

The second term is one because it is an integer number of complete turns.
Define four length-`M` transforms

```text
F_r[q] = sum(m=0..M-1) x[4m+r] W_M^(mq).
```

The output is then

```text
X[q+Mp] = sum(r=0..3) F_r[q] W_N^(rq) W_4^(rp).
```

For each `q`, define

```text
A_r[q] = F_r[q] W_N^(rq).
```

The four values

```text
X[q], X[q+M], X[q+2M], X[q+3M]
```

are precisely the four outputs of `DFT_4(A_0[q], A_1[q], A_2[q], A_3[q])`.
This gives the complete algorithm:

1. evaluate four `M`-point transforms `F_0` through `F_3`;
2. multiply lanes 1 through 3 by `W_N^(rq)`; lane zero is unchanged;
3. evaluate one FFT4 across the four lanes for every `q`;
4. store output `p` at `q+pM`.

Nothing approximate has been introduced. This is an algebraic
refactorization of the original DFT.

### 3.1 Inverse transform

The normalized inverse is

```text
x[j] = (1/N) sum(k=0..N-1) X[k] exp(+2 pi i j k/N).
```

Changing the sign in the exponential is equivalent to evaluating the forward
DFT at the negated frequency:

```text
IDFT(X)[j] = DFT(X)[(-j) mod N] / N.
```

`fft_inverse_execute` therefore:

1. calls the selected tuned forward implementation unchanged;
2. keeps output zero in place;
3. swaps output pairs `k` and `N-k`;
4. multiplies every real and imaginary component by `1/N`.

The reversal and normalization are combined into one in-place `O(N)` pass.
This approach automatically provides an inverse for plain C, every SSE/AVX
lane4 version, tangent FFT, radix-2, split-radix, and FFmpeg without adding
direction branches to their hot butterfly loops. It is mathematically exact
apart from the normal float rounding of the forward transform and scaling.

## 4. Why four complex values fit naturally in AVX

The public array is an array of interleaved complex floats:

```text
re(0), im(0), re(1), im(1), ...
```

A 256-bit register contains eight floats, hence four complex values. Loading
32 bytes at `x[4m]` produces

```text
[x[4m+0], x[4m+1], x[4m+2], x[4m+3]].
```

These are exactly the four residue classes required by the decomposition.
During the inner FFT, a vector row has the interpretation

```text
V[q] = [F_0[q], F_1[q], F_2[q], F_3[q]].
```

An inner butterfly applies the same scalar root to all four components, so
every packed add, subtract, multiply, or FMA does useful work for four
independent transforms. Cross-lane shuffles are unnecessary until the final
FFT4.

This layout is different from the common interpretation of one vector as
four adjacent output frequencies. Here the lanes are four independent
transforms until the final stage.

## 5. Planning

`lane4_fft_plan_create` performs all transform-independent work before timing:

1. set `M=N/4` and determine `log2(M)`;
2. construct the mixed-radix input permutation;
3. create stage-major inner-twiddle triplets;
4. create the final factors `W_N^(rq)` for `r=1,2,3`;
5. allocate aligned vector work storage.

### 5.1 Mixed-radix permutation

The optimized inner transform is radix-4. If `log2(M)` is even, its first
codelet is FFT4. If it is odd, the first codelet is FFT8, consisting of one
radix-2 digit and one radix-4 digit. Remaining digits are radix-4.

The plan reverses those mixed-radix digits once. Execution can then load input
groups in the order required by a leaf codelet and immediately write
natural-order leaf outputs. This fuses permutation with useful arithmetic and
avoids a standalone bit/digit-reversal pass.

The stored permutation entries are complex-element offsets, not byte
addresses. That removes a multiply from hot address generation.

### 5.2 Stage-major twiddles

Suppose a radix-4 stage combines four transforms of length `L` into one of
length `4L`. At frequency `j`, the three nontrivial inputs use

```text
W_(4L)^j, W_(4L)^(2j), W_(4L)^(3j).
```

The plan stores these three complex values together, in the exact order in
which execution consumes them. The `j=0` factors are all one and are omitted.
The stage loop therefore advances linearly through a 24-byte float triplet
instead of calculating three root indices.

### 5.3 Finish factors

For four adjacent frequencies, each of the three nonzero lanes needs four
different factors. The AVX plan stores separate vectors containing duplicated
real and imaginary parts:

```text
real = [wr0,wr0, wr1,wr1, wr2,wr2, wr3,wr3]
imag = [wi0,wi0, wi1,wi1, wi2,wi2, wi3,wi3].
```

The hot path loads these vectors directly. It does not execute
`moveldup`/`movehdup` shuffles for every transform. Lane `r=0` has factor one
and has no table entry or multiplication.

## 6. The optimized 256-bit execution schedule

### 6.1 Register leaves

The permuted inputs first enter fixed codelets:

- FFT4 for an even number of inner levels;
- FFT8 for an odd number of inner levels.

The vector FFT4 takes four vectors `a,b,c,d`. Each vector represents the same
time/frequency position in four independent transforms. Define

```text
ac_sum  = a + c
ac_diff = a - c
bd_sum  = b + d
rotated = -i (b - d).
```

The outputs are

```text
y0 = ac_sum  + bd_sum
y1 = ac_diff + rotated
y2 = ac_sum  - bd_sum
y3 = ac_diff - rotated.
```

All equations are component-wise vector operations.

The FFT8 codelet splits even and odd outputs. Its only special roots are
`-i` and the two diagonal roots containing `sqrt(1/2)`, so it uses fixed
sign/swap operations and float multiplications rather than a generic loop.

### 6.2 Fused FFT16 and FFT32 parents

For even inner levels, four FFT4 children and their first radix-4 parent form
one straight-line FFT16 leaf. Keeping the child outputs live lets the compiler
carry most of them directly into the parent. On the test build, only a few
short-lived stack spills remain instead of storing and reloading all sixteen
work rows.

For odd levels, the FFT32 base consists of four FFT8 children and a radix-4
parent. Register pressure is higher, so three children are written to work
storage while all eight rows of the final child remain live and are consumed
by the parent.

This asymmetric choice was measured. Retaining a small part of every child
increased code size and register pressure; retaining one complete final child
was faster.

### 6.3 Block-major radix-4 stages

After the base, each stage visits one contiguous `4L` block at a time:

```text
for each block:
    a = work[block + j]
    b = work[block + L  + j] * W_(4L)^j
    c = work[block + 2L + j] * W_(4L)^(2j)
    d = work[block + 3L + j] * W_(4L)^(3j)
    (a,b,c,d) = FFT4(a,b,c,d)
```

Each object in this pseudocode is a complete four-complex YMM vector. The
loop has:

- no recursion;
- no transform-family switch;
- no per-butterfly kind branch;
- linear work-array streams;
- linear twiddle-table access.

Two adjacent butterflies are unrolled together. Four-way unrolling was
tested, but the extra live state and code size made it slower.

### 6.4 Fused final transpose and FFT4

After the inner FFT, four adjacent work rows form this conceptual matrix:

```text
              lane r=0  lane r=1  lane r=2  lane r=3
frequency q       F0[q]     F1[q]     F2[q]     F3[q]
frequency q+1   F0[q+1]   F1[q+1]   F2[q+1]   F3[q+1]
frequency q+2   F0[q+2]   F1[q+2]   F2[q+2]   F3[q+2]
frequency q+3   F0[q+3]   F1[q+3]   F2[q+3]   F3[q+3]
```

A direct implementation would perform four horizontal FFT4s. Horizontal
arithmetic is shuffle-heavy because each FFT4 crosses SIMD lanes. Lane4
instead transposes the matrix first:

```text
C0 = [F0[q], F0[q+1], F0[q+2], F0[q+3]]
C1 = [F1[q], F1[q+1], F1[q+2], F1[q+3]]
C2 = [F2[q], F2[q+1], F2[q+2], F2[q+3]]
C3 = [F3[q], F3[q+1], F3[q+2], F3[q+3]].
```

The transpose operates on 64-bit complex pairs. `C1`, `C2`, and `C3` are
then multiplied by their four prepacked finish factors. One ordinary vector
FFT4 across `C0..C3` evaluates four independent final FFT4s at once.

Its four vector results contain:

```text
P0 = [X[q],    X[q+1],    X[q+2],    X[q+3]]
P1 = [X[M+q],  X[M+q+1],  X[M+q+2],  X[M+q+3]]
P2 = [X[2M+q], X[2M+q+1], X[2M+q+2], X[2M+q+3]]
P3 = [X[3M+q], X[3M+q+1], X[3M+q+2], X[3M+q+3]].
```

They can be stored directly in natural output order. Algebraically,
transposing before the FFT4 is legal because each of the four `q` positions
is independent. Computationally, it converts four shuffle-bound horizontal
FFTs into one full-width vertical FFT.

This fused finish was the largest improvement over the initial lane4
prototype.

### 6.5 Fixed small-size paths

- `N=16`: the complete inner FFT4 and final FFT4 remain in registers.
- `N=32`: a complete vector FFT8 feeds two finish groups directly.

Larger transforms use the uniform assembly leaf/stage/finish pipeline. The
earlier intrinsic implementation tested fixed 64/128 codelets, but they are
not retained in the intrinsic-free source.

## 7. Mapping complex arithmetic to x86

An AVX vector is interleaved:

```text
[ar0,ai0, ar1,ai1, ar2,ai2, ar3,ai3].
```

For a complex multiplier `wr+i wi`, duplicating the factors gives

```text
real = [wr,wr, wr,wr, wr,wr, wr,wr]
imag = [wi,wi, wi,wi, wi,wi, wi,wi].
```

After swapping each real/imag pair, complex multiplication is

```text
swapped = [ai0,ar0, ai1,ar1, ...]
result  = addsub(value*real, swapped*imag).
```

`addsub` subtracts the even elements and adds the odd elements, producing

```text
[ar*wr-ai*wi, ai*wr+ar*wi, ...].
```

With FMA, the multiply by `real` and alternating add/subtract are fused into
`fmaddsub`; the swapped-imaginary product remains a separate multiply.

Multiplication by `-i` needs no floating-point multiplication. It swaps each
real/imag pair and changes the signs of the new imaginary elements.

AVX2 adds useful integer operations to AVX but no new packed-float FFT
arithmetic. That is why AVX and AVX2 versions of this kernel have nearly the
same instruction stream and benchmark result. FMA is the relevant arithmetic
extension.

## 8. Plain-C and SSE implementations

The portable plan uses rows with explicit real and imaginary lanes:

```text
row.re[4]
row.im[4].
```

The plain-C object contains no intrinsics and is built with automatic
vectorization disabled. This makes it a genuine algorithmic scalar baseline.

Despite using radix-2 for its inner transforms, `lane4-c` is faster than this
repository's conventional radix-2 implementation because:

- each twiddle load and each `start/k` loop iteration serves four independent
  butterflies;
- the fixed four-lane scalar loop exposes independent instruction-level
  parallelism;
- the input permutation is precomputed, whereas conventional radix-2
  dynamically calculates bit reversal and performs conditional swaps;
- lane4 uses one uniform complex multiply, whereas the comparison radix-2
  dispatches through a classified-root switch inside each butterfly;
- the batched final FFT4 produces natural-order output directly.

Thus the gain comes from amortized control, address generation, permutation,
and table traffic rather than a better asymptotic operation count.

The SSE representation maps `row.re` and `row.im` to two 128-bit vectors.
Each inner butterfly therefore evaluates all four residue transforms in
parallel, although a complex row needs two registers rather than one YMM
register. Unlike the scalar version, it uses a mixed-radix digit permutation,
fused FFT4/FFT8 assembly leaves, and block-major radix-4 upper stages.
Replicated real/imaginary roots let the hot loop consume aligned vector
operands without scalar broadcast shuffles. Its finish kernel loads four
rows, transposes the real and imaginary vectors, applies finish factors, runs
the vertical FFT4, and stores natural-order interleaved output.

The public transform scheduler, arithmetic, and hot-path data movement are in
`lane4_sse_stage.asm`, written with FFmpeg's vendored x86inc conventions.
Planning, allocation, outer algorithm selection, and benchmarking remain C.
The assembly uses stage-major root streams, linear row pointers, exact-root
sign folds, and a register-resident FFT8 that occupies all sixteen XMM
registers.

SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 retain separate runtime feature
and API names. For this split real/imag float loop, the later revisions add
no better packed arithmetic operation, so all six names alias the same SSE1
assembly body. Their near-identical timings are expected.

## 9. Work, memory, and output properties

The asymptotic cost remains `O(N log N)`. At the outer level the work is:

```text
four M-point FFTs
3M nontrivial finish twiddle multiplications
M FFT4s
```

The optimized plan allocates approximately:

```text
8N bytes   vector work array
12N bytes  split real/imag finish-factor vectors
~2N bytes  stage-major inner-twiddle triplets
N bytes    mixed-radix permutation
```

This is roughly `23N` bytes plus allocator and plan metadata. Planning and
allocation are outside benchmark timing. The shared portable/SSE plan uses a
different layout: about `8N` work, `6N` finish factors, `2N` ordinary roots,
`8N` stage-major replicated roots, and `2N` permutations, or about `26N`
bytes.

The API is in-place, but execution uses the plan's work array. Forward output
is the conventional natural-order, unnormalised DFT; inverse output is
natural-order and normalized by `1/N`. The current implementation supports:

```text
power-of-two N
16 <= N <= 131072
complex float input
forward and inverse transforms
```

There is no real-input specialization yet.

## 10. Why it is fast despite not minimizing arithmetic

Tangent FFT and modified split-radix algorithms focus on real-operation
count. Lane4 instead optimizes the dependency graph seen by a modern CPU:

- almost every hot instruction operates on four independent complex values;
- cross-lane work is postponed and batched into one final transpose;
- radix-4 reduces the number of full work-array stage passes;
- leaf arithmetic absorbs the input permutation;
- stage twiddles are linear streams rather than derived addresses;
- the hot loops are branch-free and nonrecursive;
- small transforms and selected parent/child edges stay in registers;
- output is produced directly in natural order;
- setup work is moved to the reusable plan.

The rejected experiments reinforce this point. Radix-8 upper stages reduced
the number of passes but increased codelet density and external twiddles;
special branching for exact eighth roots reduced arithmetic but disrupted the
uniform loop; excessive unrolling increased register pressure and code size.
All were slower in the important cache-resident range.

The result is best understood as a modern machine schedule for a standard
factorization, not as a lower-operation FFT.

## 11. Source map and reproduction

The relevant files are:

```text
lane4_fft.c                    optimized 256-bit radix-4 implementation
lane4_portable.c               plain-C implementation and shared plan
lane4_sse_stage.asm            complete x86inc/NASM SSE execution path
fft.c / fft.h                  API selection and runtime ISA checks
Makefile                       explicit per-ISA compilation boundaries
docs/lane-factorized-fft.md    optimization history and cycle results
docs/lane4-isa-variants.md     ISA comparison and benchmark summary
benchmark-simd.csv             raw common-harness measurements
benchmark-inverse.csv          normalized inverse measurements
```

Reproduce the validation and benchmark with:

```sh
make clean
make
make test
taskset -c 2 ./fft_harness --bench --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark-simd.csv
taskset -c 2 ./fft_harness --bench --inverse --min-power 4 --max-power 13 \
  --target-ms 1000 --csv benchmark-inverse.csv
```

The correctness harness compares every forward and inverse implementation
against a long-double direct DFT through 512, then cross-checks larger
supported sizes against the independent radix-2 implementation. ASan and
UBSan builds also pass the complete suite.
