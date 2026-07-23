# Rejected tangent FFT experiments

This report records variants tried during the tangent assembly follow-up but
not retained. Times are medians on the Ryzen 9 3900X and should be read with
the paired FFmpeg result because boost state changed between some short runs.

## AVX2/FMA experiments

### Carrying half of the final FFT8 into FFT128

The final S-family FFT8 child was changed to retain its first four outputs in
a YMM register. Its other four outputs were stored, and the parent FFT32
consumed the live half before the rest of the fixed tree.

This saved one 32-byte store and reload, but added two `vinsertf128`
operations, changed node order, and increased live-register pressure. It
showed no repeatable improvement; cycle runs were approximately tied and
some short runs regressed by about 10%. The simpler fully materialized leaf
was restored.

### Copying fixed results back before `vzeroupper`

FFT32, FFT64, FFT128, and FFT256 were tested with an unrolled AVX copy from
scratch to the public input before `vzeroupper`, replacing the subsequent C
`memcpy`.

It is useful for the smallest fixed paths, and direct in-place FFT16 is a
clear win. At FFT128 the AVX copy made a previously fast fixed path roughly
10% slower. At FFT256 it changed a stable 0.33 µs result to about 0.50 µs.
FFT128 and FFT256 therefore retain the C copy. The small fixed paths retain
only the copy choices that measured positively.

### Compact FFT512 made from fixed FFT256 plus two FFT128 children

A special scheduler computed the fixed FFT256 even child, gathered only the
two remaining S-family FFT128 children, evaluated their seven homogeneous
node batches, and finished with the 512-point root.

It passed the full correctness suite but remained at 0.72–0.73 µs, the same
as the generic grouped schedule and slower than FFmpeg's 0.68–0.69 µs.
Reusing the fixed child lost some batching across sibling subtrees, offsetting
the reduced dispatch. The special scheduler was removed.

### Fully unrolled FFT512

Extending the FFT256 expansion mechanically would emit more than one hundred
four-complex combine chunks plus 43 fixed leaves. Estimated code size exceeds
the useful L1-instruction-cache budget and the compact fixed-child experiment
already showed no gain. This variant was rejected before adding permanent
code; the generic linear loops are the better structure at this size.

### Scalar exact-root split inside a YMM group

Upper-stage vectors contain `k=0` beside three nontrivial factors. Handling
only the exact lane would require scalar extraction/insertion or a mask/blend
sequence while the other lanes still need the general product. Both forms
add instructions without removing the vector multiplies. Exact-root
specialization remains confined to fixed leaves, where the complete
butterfly layout makes it free.

## SSE experiments

### Upper stages without fused leaves

The first SSE version used assembly upper nodes but retained separate
permutation, base, and levels 2–4 passes. It was correct but took about
0.68 µs at 128, 1.67 µs at 512, and 39.9 µs at 8192.

Fixed 4/8/16 leaves reduced those figures to approximately 0.36, 1.56, and
37.8 µs. The upper-only design was discarded.

### Universal leaf-local permutation gather

Gathering two scalar complex values with `movlps`/`movhps`, storing the local
leaf, and executing it immediately was tested at every size.

It hurt smaller transforms:

| N | separate permutation | leaf-local gather |
|---:|---:|---:|
| 128 | about 0.36 µs | about 0.47 µs |
| 512 | about 1.53–1.56 µs | about 1.60 µs |
| 4096 | about 15.36 µs | about 15.50 µs |
| 8192 | about 37.82 µs | about 35.64 µs |

The universal policy was rejected. The retained dispatcher uses the separate
permutation below 8192 and leaf-local gather from 8192 upward.

### Reduced tangent products at every SSE level

Low/high tangent-region branches and reduced products were enabled for every
S and S2 node. They helped large streaming stages but made small stages
slower because each short node paid boundary comparisons and extra control
flow.

S2-wide specialization was reverted. The retained S path uses reduced
products only when the quarter length is at least 16, where the predictable
low/high regions amortize the branch cost.

### Separate SSSE3/SSE4 arithmetic bodies

SSSE3 and SSE4.x do not add a float operation that improves the selected
interleaved-complex butterfly. Blend/insert alternatives either require the
same number of µops or add data movement. Independent public dispatch entries
were retained, but their measured kernel aliases the SSE3 implementation.

## Outcome

The failed variants clarify the useful boundaries:

- register carry works when a complete child naturally matches the parent,
  as in FFT64, but partial-child carry is not automatically profitable;
- fixed straight-line trees pay through FFT256, while FFT512 benefits more
  from compact batching;
- moving copy/permutation work into assembly must be selected by size;
- fewer arithmetic instructions do not compensate for control flow or lane
  repair at small SIMD widths.

The retained implementation and final results are described in
`docs/tangent-assembly-optimization.md`.
