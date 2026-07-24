# Lane2 AArch64 NEON port

## Implementation

`lane2-neon` is a handwritten AArch64 assembly port of the `N=2M`
lane-factorized FFT.  One 128-bit NEON register holds two complete
interleaved complex values:

```text
[F0.re, F0.im, F1.re, F1.im].
```

The production path consists of:

- mixed-radix FFT4/FFT8 gather leaves;
- block-major radix-4 upper stages;
- exact `W8`, `-i`, and `W8^3` special cases;
- a fused two-row transpose, twiddle multiplication, and FFT2 finish;
- a C planner that generates only `float` coefficients and aligned scratch.

The implementation is in [`lane2_neon_stage.S`](../lane2_neon_stage.S) and
[`lane2_neon.c`](../lane2_neon.c).  It is selected only for an AArch64 target
triple and appears as `lane2-neon` in the common harness.

## Handling the missing addsub instruction

AArch64 NEON does not provide the packed `addsubps` operation used by many
x86 complex-multiply sequences.  The port handles that in two ways.

For a generic complex root, the planner stores:

```text
real       = [ wr,  wr,  wr,  wr]
signed_im  = [-wi, +wi, -wi, +wi].
```

An interleaved two-complex multiplication is consequently:

```asm
rev64  swapped.4s, value.4s
fmul   value.4s, value.4s, real.4s
fmla   value.4s, swapped.4s, signed_im.4s
```

No sign mask or separate add/sub is required in the generic product.  Exact
multiplication by `-i` still needs a lane-selective sign operation, so `v31`
holds `[+0,-0,+0,-0]` and the kernel uses `REV64` plus `EOR`.

The initial port reloaded this mask and the three exact diagonal constants at
every stage.  The current code loads `v28-v31` once at public entry and
carries them through the private leaf/stage calls.  These registers are
caller-saved under AAPCS64, and the kernel's maximum live set still leaves
`v25-v27` unused, so the reserved mask causes no spill.  This implements the
same register-carrying idea listed in FFmpeg's x86 FFT TODO, adapted to the
larger AArch64 register file.

## Correctness and build validation

The normal x86 build and complete host test suite continue to pass.  The
AArch64 object assembles with Clang 20.

A freestanding test target builds the production planner and assembly into a
static AArch64 executable.  It compares every power of two from 16 through
8192 against an independent double-precision radix-2 reference.  Under QEMU
10.1 it reports:

```text
PASS: lane2-neon production planner/assembly, N=16..8192
```

With an AArch64-capable Clang, LLD, and QEMU installed:

```sh
make aarch64-asm-check
make aarch64-qemu-test
```

QEMU establishes functional correctness, not native performance.

The N=16 path is now a fused 95-instruction leaf, including argument checks
and dispatch. It loads the natural input with four paired loads, keeps the
paired FFT8 in registers, and writes the top FFT2 directly. FFmpeg's
pre-shuffled and natural-input FFT16 codelets contain 104 and 137
instructions respectively. This gives lane2 a credible N=16 case without
changing the less favorable upper-stage analysis below.

## Static throughput

[`analysis/lane2_neon_mca.s`](../analysis/lane2_neon_mca.s) contains the
steady-state upper butterfly and finish loop.  LLVM-MCA 20.1.8 reports the
following reciprocal throughputs in cycles per loop iteration:

| Core model | upper radix-4 / 8 complex | finish / 4 complex |
|---|---:|---:|
| Cortex-A53 | 19.0 | 9.5 |
| Cortex-A76 | 20.0 | 9.3 |
| Neoverse N1 | 13.5 | 5.5 |
| Apple M1 | 6.3 | 3.2 |
| Apple M3 | 6.3 | 3.2 |

These are static scheduling estimates with assumed L1 hits.  They are useful
for identifying pressure but are not substitutes for measurements on each
core.

The ordinary upper butterfly performs three complex rotations and one packed
radix-4:

```text
19 vector ALU/shuffle instructions
10 Q-register loads (four data, six coefficient)
 4 Q-register stores
```

The load/store pressure is the main concern.  Four lane2 butterflies cover
the same 32 complex values as one FFmpeg `SR_COMBINE` expansion.  At that
normalization:

| Work for 32 complex outputs | lane2-neon | FFmpeg SR combine |
|---|---:|---:|
| vector ALU/shuffle operations | 76 | 72 |
| Q-register loads and stores | 56 | 36 |
| source-level instructions including addressing | 120 | about 93 |

The FFmpeg count comes from the active
`libavutil/aarch64/tx_float_neon.S` macro plus its surrounding paired
loads/stores.  It is not a cycle prediction, but it exposes lane2's larger
coefficient stream.

The reduction from 140 to 120 instructions comes from the wider ARM audit:
four post-index output stores replace four stores plus four pointer additions,
and the coefficient packet advance is folded into its first paired load.
LLVM-MCA's backend-bound reciprocal throughput is unchanged, but dynamic
execution falls by 24,918 instructions (10.4%) at N=8192. Full before/after
counts and the audit are in
[`arm-optimization-audit.md`](arm-optimization-audit.md).

## Compact-root experiment

An ARM-specific alternative was modeled with `LD2R`: store each root as one
interleaved `{wr,wi}` pair, expand it into two vectors at execution, and use
an `EOR` mask to form `[-wi,+wi,...]`.  It reduces coefficient traffic per
upper butterfly from 96 to 24 bytes.  With all data assumed in L1, however,
LLVM-MCA predicts worse throughput:

| Core model | expanded roots | compact roots |
|---|---:|---:|
| Cortex-A53 | 19.0 | 20.0 |
| Cortex-A76 | 20.0 | 21.7 |
| Neoverse N1 | 13.5 | 18.0 |
| Apple M1/M3 | 6.3 | 9.3 |

The expanded table therefore remains the production choice for the important
cache-resident sizes.  A size-dependent compact-root path could still help
once coefficient streams no longer fit the target's L1 cache; that requires
native measurements.

## Likelihood of beating FFmpeg

At code level, this port is credible but not likely to beat FFmpeg at every
size:

- FFmpeg has fully register-resident FFT2 through FFT32 codelets, whereas
  lane2 uses generic gather leaves and a separate finish.
- FFmpeg's split-radix combine has slightly fewer vector operations and
  substantially less coefficient traffic per 32 complex outputs.
- Lane2 performs a conventional radix-4 Cooley--Tukey decomposition, so it
  does not have split-radix's arithmetic-count advantage.
- The common project adapter copies data into and out of FFmpeg's AVTX
  buffers.  Lane2 avoids those two adapter copies, which may make it
  competitive at some small or medium sizes in this particular harness.
- Wide Apple cores match lane2's independent FMA-heavy schedule much better
  than Cortex-A53/A76 according to the static models.

The present expectation is that lane2 may win isolated sizes on Apple or
other wide cores when measured through the common in-place API, but FFmpeg is
the favorite over the complete 16--8192 range.  Native measurements are
needed before claiming a crossover.

The strongest next experiments are fixed FFT16/FFT32 leaves, a
microarchitecture-gated Armv8.3 `FCMLA` product, and a compact-root crossover
for coefficient streams larger than L1.

The fixed FFT16 experiment is now implemented. The more promising general
follow-up is the fused lane4-SoA kernel: separate real/imaginary vectors
reduce a four-way complex product from six lane2 instructions to four and
avoid `REV64` in the generic multiply. See
[`fused-lane4-soa.md`](fused-lane4-soa.md).
