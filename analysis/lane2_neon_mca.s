// Steady-state AArch64 loops extracted from lane2_neon_stage.S.
// Run, for example:
//   llvm-mca -march=aarch64 -mcpu=cortex-a76 \
//       -iterations=100 analysis/lane2_neon_mca.s

        .text

// One nontrivial upper-stage radix-4 butterfly.  It transforms eight
// complex values: two independent packed FFTs over four input rows.
# LLVM-MCA-BEGIN lane2-general
        ldr     q0, [x0]
        ldr     q1, [x4]
        ldr     q2, [x5]
        ldr     q3, [x6]
        ldp     q16, q17, [x12]
        ldp     q18, q19, [x12, #32]
        ldp     q20, q21, [x12, #64]
        rev64   v22.4s, v1.4s
        rev64   v23.4s, v2.4s
        rev64   v24.4s, v3.4s
        fmul    v1.4s, v1.4s, v16.4s
        fmul    v2.4s, v2.4s, v18.4s
        fmul    v3.4s, v3.4s, v20.4s
        fmla    v1.4s, v22.4s, v17.4s
        fmla    v2.4s, v23.4s, v19.4s
        fmla    v3.4s, v24.4s, v21.4s
        fadd    v16.4s, v0.4s, v2.4s
        fsub    v17.4s, v0.4s, v2.4s
        fadd    v18.4s, v1.4s, v3.4s
        fsub    v19.4s, v1.4s, v3.4s
        rev64   v24.4s, v19.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v20.4s, v16.4s, v18.4s
        fadd    v21.4s, v17.4s, v24.4s
        fsub    v22.4s, v16.4s, v18.4s
        fsub    v23.4s, v17.4s, v24.4s
        str     q20, [x0]
        str     q21, [x4]
        str     q22, [x5]
        str     q23, [x6]
        add     x12, x12, #96
        add     x0, x0, #16
        add     x4, x4, #16
        add     x5, x5, #16
        add     x6, x6, #16
# LLVM-MCA-END

// Two adjacent frequencies of the fused lane2 finish.  It consumes and
// produces four complex values.
# LLVM-MCA-BEGIN lane2-finish
        ldp     q0, q1, [x0], #32
        trn1    v2.2d, v0.2d, v1.2d
        trn2    v3.2d, v0.2d, v1.2d
        ldp     q16, q17, [x3], #32
        rev64   v18.4s, v3.4s
        fmul    v3.4s, v3.4s, v16.4s
        fmla    v3.4s, v18.4s, v17.4s
        fadd    v4.4s, v2.4s, v3.4s
        fsub    v5.4s, v2.4s, v3.4s
        str     q4, [x1], #16
        str     q5, [x4], #16
        cmp     x0, x5
        b.lo    lane2_finish_loop
lane2_finish_loop:
# LLVM-MCA-END

// Alternative ARM-native compact roots.  LD2R expands one interleaved
// {wr,wi} pair into two vectors; EOR applies [-,+,-,+] to the imaginary
// vector.  This replaces 96 coefficient bytes with 24 per butterfly.
# LLVM-MCA-BEGIN lane2-general-compact
        ldr     q0, [x0]
        ldr     q1, [x4]
        ldr     q2, [x5]
        ldr     q3, [x6]
        ld2r    { v16.4s, v17.4s }, [x12], #8
        ld2r    { v18.4s, v19.4s }, [x12], #8
        ld2r    { v20.4s, v21.4s }, [x12], #8
        eor     v17.16b, v17.16b, v27.16b
        eor     v19.16b, v19.16b, v27.16b
        eor     v21.16b, v21.16b, v27.16b
        rev64   v22.4s, v1.4s
        rev64   v23.4s, v2.4s
        rev64   v24.4s, v3.4s
        fmul    v1.4s, v1.4s, v16.4s
        fmul    v2.4s, v2.4s, v18.4s
        fmul    v3.4s, v3.4s, v20.4s
        fmla    v1.4s, v22.4s, v17.4s
        fmla    v2.4s, v23.4s, v19.4s
        fmla    v3.4s, v24.4s, v21.4s
        fadd    v16.4s, v0.4s, v2.4s
        fsub    v17.4s, v0.4s, v2.4s
        fadd    v18.4s, v1.4s, v3.4s
        fsub    v19.4s, v1.4s, v3.4s
        rev64   v24.4s, v19.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v20.4s, v16.4s, v18.4s
        fadd    v21.4s, v17.4s, v24.4s
        fsub    v22.4s, v16.4s, v18.4s
        fsub    v23.4s, v17.4s, v24.4s
        str     q20, [x0]
        str     q21, [x4]
        str     q22, [x5]
        str     q23, [x6]
        add     x0, x0, #16
        add     x4, x4, #16
        add     x5, x5, #16
        add     x6, x6, #16
# LLVM-MCA-END

// Compact finish roots contain two interleaved complex values.  LD2 plus
// ZIP1 creates [wr0,wr0,wr1,wr1] and [wi0,wi0,wi1,wi1].
# LLVM-MCA-BEGIN lane2-finish-compact
        ldp     q0, q1, [x0], #32
        trn1    v2.2d, v0.2d, v1.2d
        trn2    v3.2d, v0.2d, v1.2d
        ld2     { v16.2s, v17.2s }, [x3], #16
        zip1    v16.4s, v16.4s, v16.4s
        zip1    v17.4s, v17.4s, v17.4s
        eor     v17.16b, v17.16b, v27.16b
        rev64   v18.4s, v3.4s
        fmul    v3.4s, v3.4s, v16.4s
        fmla    v3.4s, v18.4s, v17.4s
        fadd    v4.4s, v2.4s, v3.4s
        fsub    v5.4s, v2.4s, v3.4s
        str     q4, [x1], #16
        str     q5, [x4], #16
        cmp     x0, x5
        b.lo    lane2_finish_compact_loop
lane2_finish_compact_loop:
# LLVM-MCA-END
