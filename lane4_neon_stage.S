/*
 * Fused lane4 structure-of-arrays leaves for AArch64 NEON.
 *
 * A row is represented by two registers:
 *
 *     re = [z0.re, z1.re, z2.re, z3.re]
 *     im = [z0.im, z1.im, z2.im, z3.im]
 *
 * The N=16 leaf is a radix-4 DIF step, three vector complex products,
 * a 4x4 transpose, and a second radix-4 step.  It reads and writes natural
 * AoS order and never materializes the generic lane4 work layout.
 */

#define PLAN_N          0
#define PLAN_FINISH_RE 56
#define PLAN_FINISH_IM 64
#define PLAN_WORK      72

        .section .rodata
        .p2align 4
.Llane4_fft16_re:
        .float  1.0,  0.9238795042037963867
        .float  0.7071067811865475244,  0.3826834323650897717
        .float  1.0,  0.7071067811865475244
        .float  0.0, -0.7071067811865475244
        .float  1.0,  0.3826834323650897717
        .float -0.7071067811865475244, -0.9238795042037963867
.Llane4_fft16_im:
        .float  0.0, -0.3826834323650897717
        .float -0.7071067811865475244, -0.9238795042037963867
        .float  0.0, -0.7071067811865475244
        .float -1.0, -0.7071067811865475244
        .float  0.0, -0.9238795042037963867
        .float -0.7071067811865475244,  0.3826834323650897717
.Llane4_fft32_re:
        .float  1.0,  0.9807852804032304491
        .float  0.9238795325112867561,  0.8314696123025452371
        .float  0.7071067811865475244,  0.5555702330196022247
        .float  0.3826834323650897717,  0.1950903220161282678
        .float  0.0, -0.1950903220161282678
        .float -0.3826834323650897717, -0.5555702330196022247
        .float -0.7071067811865475244, -0.8314696123025452371
        .float -0.9238795325112867561, -0.9807852804032304491
.Llane4_fft32_im:
        .float  0.0, -0.1950903220161282678
        .float -0.3826834323650897717, -0.5555702330196022247
        .float -0.7071067811865475244, -0.8314696123025452371
        .float -0.9238795325112867561, -0.9807852804032304491
        .float -1.0, -0.9807852804032304491
        .float -0.9238795325112867561, -0.8314696123025452371
        .float -0.7071067811865475244, -0.5555702330196022247
        .float -0.3826834323650897717, -0.1950903220161282678
.Llane4_sqrt_half:
        .float  0.7071067811865475244,  0.7071067811865475244
        .float  0.7071067811865475244,  0.7071067811865475244
.Llane4_neg_sqrt_half:
        .float -0.7071067811865475244, -0.7071067811865475244
        .float -0.7071067811865475244, -0.7071067811865475244

        .text

/*
 * Complex radix-4 on four SoA rows.
 *
 * The eight temporaries must not overlap an input until all temporaries have
 * been produced. Outputs may alias inputs.
 */
        .macro  LANE4_RADIX4_SOA \
                ar, ai, br, bi, cr, ci, dr, di, \
                o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i, \
                t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i
        fadd    \t0r\().4s, \ar\().4s, \cr\().4s
        fadd    \t0i\().4s, \ai\().4s, \ci\().4s
        fsub    \t1r\().4s, \ar\().4s, \cr\().4s
        fsub    \t1i\().4s, \ai\().4s, \ci\().4s
        fadd    \t2r\().4s, \br\().4s, \dr\().4s
        fadd    \t2i\().4s, \bi\().4s, \di\().4s
        fsub    \t3r\().4s, \br\().4s, \dr\().4s
        fsub    \t3i\().4s, \bi\().4s, \di\().4s

        fadd    \o0r\().4s, \t0r\().4s, \t2r\().4s
        fadd    \o0i\().4s, \t0i\().4s, \t2i\().4s
        fadd    \o1r\().4s, \t1r\().4s, \t3i\().4s
        fsub    \o1i\().4s, \t1i\().4s, \t3r\().4s
        fsub    \o2r\().4s, \t0r\().4s, \t2r\().4s
        fsub    \o2i\().4s, \t0i\().4s, \t2i\().4s
        fsub    \o3r\().4s, \t1r\().4s, \t3i\().4s
        fadd    \o3i\().4s, \t1i\().4s, \t3r\().4s
        .endm

/*
 * Transpose four 4-float rows.  The first four arguments are destroyed.
 */
        .macro  LANE4_TRANSPOSE4 r0, r1, r2, r3, c0, c1, c2, c3
        trn1    \c0\().4s, \r0\().4s, \r1\().4s
        trn2    \c1\().4s, \r0\().4s, \r1\().4s
        trn1    \r0\().4s, \r2\().4s, \r3\().4s
        trn2    \r1\().4s, \r2\().4s, \r3\().4s
        trn1    \r2\().2d, \c0\().2d, \r0\().2d
        trn2    \r3\().2d, \c0\().2d, \r0\().2d
        trn1    \c2\().2d, \c1\().2d, \r1\().2d
        trn2    \c3\().2d, \c1\().2d, \r1\().2d
        mov     \c0\().16b, \r2\().16b
        mov     \c1\().16b, \c2\().16b
        mov     \c2\().16b, \r3\().16b
        .endm

        .p2align 6
        .global lane4_neon_execute
        .type   lane4_neon_execute, %function
lane4_neon_execute:
        cbz     x0, .Llane4_error
        cbz     x1, .Llane4_error
        ldr     x9, [x0, #PLAN_N]
        cmp     x9, #16
        b.eq    .Llane4_size16
        cmp     x9, #32
        b.eq    .Llane4_size32
        cmp     x9, #64
        b.eq    .Llane4_size64
        b.gt    .Llane4_generic
        b       .Llane4_error

.Llane4_size16:
        ldr     x2, [x0, #PLAN_FINISH_RE]
        ldr     x3, [x0, #PLAN_FINISH_IM]
.Llane4_fft16_core:
        mov     x9, x1

        ld2     { v0.4s, v1.4s }, [x1], #32
        ld2     { v2.4s, v3.4s }, [x1], #32
        ld2     { v4.4s, v5.4s }, [x1], #32
        ld2     { v6.4s, v7.4s }, [x1]

        LANE4_RADIX4_SOA \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v16, v17, v18, v19, v20, v21, v22, v23

        /* Y1 *= W16^k. */
        ldr     q24, [x2], #16
        ldr     q25, [x3], #16
        fmul    v16.4s, v2.4s, v24.4s
        fmul    v17.4s, v3.4s, v24.4s
        fmls    v16.4s, v3.4s, v25.4s
        fmla    v17.4s, v2.4s, v25.4s

        /* Y2 *= W16^(2k). */
        ldr     q24, [x2], #16
        ldr     q25, [x3], #16
        fmul    v18.4s, v4.4s, v24.4s
        fmul    v19.4s, v5.4s, v24.4s
        fmls    v18.4s, v5.4s, v25.4s
        fmla    v19.4s, v4.4s, v25.4s

        /* Y3 *= W16^(3k). */
        ldr     q24, [x2]
        ldr     q25, [x3]
        fmul    v20.4s, v6.4s, v24.4s
        fmul    v21.4s, v7.4s, v24.4s
        fmls    v20.4s, v7.4s, v25.4s
        fmla    v21.4s, v6.4s, v25.4s

        /*
         * Rows are [X_l(0), X_l(1), X_l(2), X_l(3)].  Transpose them so
         * the second radix-4 operates on l and directly produces four
         * natural-order output blocks.
         */
        LANE4_TRANSPOSE4 v0, v16, v18, v20, v22, v23, v24, v25
        LANE4_TRANSPOSE4 v1, v17, v19, v21, v26, v27, v28, v29

        LANE4_RADIX4_SOA \
                v22, v26, v23, v27, v24, v28, v25, v29, \
                v16, v17, v18, v19, v20, v21, v22, v23, \
                v0, v1, v2, v3, v4, v5, v6, v7

        st2     { v16.4s, v17.4s }, [x9], #32
        st2     { v18.4s, v19.4s }, [x9], #32
        st2     { v20.4s, v21.4s }, [x9], #32
        st2     { v22.4s, v23.4s }, [x9]
        mov     w0, wzr
        ret

.Llane4_size32:
        /*
         * Radix-2 DIF produces even/odd length-16 transforms in work.
         * The two FFT16 leaves are shared with N=16/N=64.  ZIP performs the
         * final Stockham autosort directly into natural AoS output.
         */
        stp     x19, x20, [sp, #-48]!
        stp     x21, x22, [sp, #16]
        str     x30, [sp, #32]
        ldr     x19, [x0, #PLAN_WORK]
        mov     x22, x1
        adrp    x2, .Llane4_fft32_re
        add     x2, x2, :lo12:.Llane4_fft32_re
        adrp    x3, .Llane4_fft32_im
        add     x3, x3, :lo12:.Llane4_fft32_im
        mov     x5, x1
        add     x6, x1, #128
        mov     x10, x19
        add     x11, x19, #128
        mov     w14, #4

.Llane4_size32_split:
        ld2     { v0.4s, v1.4s }, [x5], #32
        ld2     { v2.4s, v3.4s }, [x6], #32
        fadd    v4.4s, v0.4s, v2.4s
        fadd    v5.4s, v1.4s, v3.4s
        fsub    v6.4s, v0.4s, v2.4s
        fsub    v7.4s, v1.4s, v3.4s
        st2     { v4.4s, v5.4s }, [x10], #32
        ldr     q24, [x2], #16
        ldr     q25, [x3], #16
        fmul    v26.4s, v6.4s, v24.4s
        fmul    v27.4s, v7.4s, v24.4s
        fmls    v26.4s, v7.4s, v25.4s
        fmla    v27.4s, v6.4s, v25.4s
        st2     { v26.4s, v27.4s }, [x11], #32
        subs    w14, w14, #1
        b.ne    .Llane4_size32_split

        adrp    x20, .Llane4_fft16_re
        add     x20, x20, :lo12:.Llane4_fft16_re
        adrp    x21, .Llane4_fft16_im
        add     x21, x21, :lo12:.Llane4_fft16_im
        mov     x1, x19
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core
        add     x1, x19, #128
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core

        mov     x4, x19
        add     x5, x19, #128
        mov     x9, x22
        mov     w10, #4

.Llane4_size32_autosort:
        ld2     { v0.4s, v1.4s }, [x4], #32
        ld2     { v2.4s, v3.4s }, [x5], #32
        zip1    v4.4s, v0.4s, v2.4s
        zip1    v5.4s, v1.4s, v3.4s
        zip2    v6.4s, v0.4s, v2.4s
        zip2    v7.4s, v1.4s, v3.4s
        st2     { v4.4s, v5.4s }, [x9], #32
        st2     { v6.4s, v7.4s }, [x9], #32
        subs    w10, w10, #1
        b.ne    .Llane4_size32_autosort

        ldr     x30, [sp, #32]
        ldp     x21, x22, [sp, #16]
        ldp     x19, x20, [sp], #48
        mov     w0, wzr
        ret

.Llane4_size64:
        /*
         * One fused radix-4 DIF pass creates four contiguous length-16
         * transforms in the plan work area.  Four register-resident FFT16
         * leaves consume them, and the final 4x4 transposes perform the
         * Stockham autosort while writing natural-order output.
         */
        stp     x19, x20, [sp, #-48]!
        stp     x21, x22, [sp, #16]
        str     x30, [sp, #32]
        ldr     x19, [x0, #PLAN_WORK]
        mov     x22, x1
        ldr     x2, [x0, #PLAN_FINISH_RE]
        ldr     x3, [x0, #PLAN_FINISH_IM]

        mov     x5, x1
        add     x6, x1, #128
        add     x7, x1, #256
        add     x8, x1, #384
        mov     x10, x19
        add     x11, x19, #128
        add     x12, x19, #256
        add     x13, x19, #384
        mov     w14, #4

.Llane4_size64_split:
        ld2     { v0.4s, v1.4s }, [x5], #32
        ld2     { v2.4s, v3.4s }, [x6], #32
        ld2     { v4.4s, v5.4s }, [x7], #32
        ld2     { v6.4s, v7.4s }, [x8], #32

        LANE4_RADIX4_SOA \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v16, v17, v18, v19, v20, v21, v22, v23

        st2     { v0.4s, v1.4s }, [x10], #32

        ldr     q24, [x2]
        ldr     q25, [x3]
        fmul    v26.4s, v2.4s, v24.4s
        fmul    v27.4s, v3.4s, v24.4s
        fmls    v26.4s, v3.4s, v25.4s
        fmla    v27.4s, v2.4s, v25.4s
        st2     { v26.4s, v27.4s }, [x11], #32

        ldr     q24, [x2, #64]
        ldr     q25, [x3, #64]
        fmul    v28.4s, v4.4s, v24.4s
        fmul    v29.4s, v5.4s, v24.4s
        fmls    v28.4s, v5.4s, v25.4s
        fmla    v29.4s, v4.4s, v25.4s
        st2     { v28.4s, v29.4s }, [x12], #32

        ldr     q24, [x2, #128]
        ldr     q25, [x3, #128]
        fmul    v30.4s, v6.4s, v24.4s
        fmul    v31.4s, v7.4s, v24.4s
        fmls    v30.4s, v7.4s, v25.4s
        fmla    v31.4s, v6.4s, v25.4s
        st2     { v30.4s, v31.4s }, [x13], #32

        add     x2, x2, #16
        add     x3, x3, #16
        subs    w14, w14, #1
        b.ne    .Llane4_size64_split

        adrp    x20, .Llane4_fft16_re
        add     x20, x20, :lo12:.Llane4_fft16_re
        adrp    x21, .Llane4_fft16_im
        add     x21, x21, :lo12:.Llane4_fft16_im

        mov     x1, x19
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core
        add     x1, x19, #128
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core
        add     x1, x19, #256
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core
        add     x1, x19, #384
        mov     x2, x20
        mov     x3, x21
        bl      .Llane4_fft16_core

        mov     x4, x19
        add     x5, x19, #128
        add     x6, x19, #256
        add     x7, x19, #384
        mov     x9, x22
        mov     w10, #4

.Llane4_size64_autosort:
        ld2     { v0.4s, v1.4s }, [x4], #32
        ld2     { v2.4s, v3.4s }, [x5], #32
        ld2     { v4.4s, v5.4s }, [x6], #32
        ld2     { v6.4s, v7.4s }, [x7], #32
        LANE4_TRANSPOSE4 v0, v2, v4, v6, v16, v18, v20, v22
        LANE4_TRANSPOSE4 v1, v3, v5, v7, v17, v19, v21, v23
        st2     { v16.4s, v17.4s }, [x9], #32
        st2     { v18.4s, v19.4s }, [x9], #32
        st2     { v20.4s, v21.4s }, [x9], #32
        st2     { v22.4s, v23.4s }, [x9], #32
        subs    w10, w10, #1
        b.ne    .Llane4_size64_autosort

        ldr     x30, [sp, #32]
        ldp     x21, x22, [sp, #16]
        ldp     x19, x20, [sp], #48
        mov     w0, wzr
        ret

.Llane4_generic:
        stp     x19, x20, [sp, #-128]!
        stp     x21, x22, [sp, #16]
        stp     x23, x24, [sp, #32]
        str     x30, [sp, #48]
        stp     d8, d9, [sp, #64]
        stp     d10, d11, [sp, #80]
        stp     d12, d13, [sp, #96]
        stp     d14, d15, [sp, #112]
        mov     x19, x0
        mov     x20, x1

        mov     x0, x20
        ldr     x1, [x19, #32]
        ldr     x2, [x19, #PLAN_WORK]
        ldr     x3, [x19, #8]
        ldr     w9, [x19, #16]
        tbz     w9, #0, .Llane4_generic_base4
        bl      .Llane4_base8
        mov     x21, #8
        b       .Llane4_generic_stage_test

.Llane4_generic_base4:
        bl      .Llane4_base4
        mov     x21, #4

.Llane4_generic_stage_test:
        ldr     x9, [x19, #8]
        cmp     x21, x9
        b.hs    .Llane4_generic_finish
        ldr     x22, [x19, #48]

.Llane4_generic_stage_loop:
        ldr     x0, [x19, #PLAN_WORK]
        mov     x1, x21
        ldr     x2, [x19, #8]
        mov     x3, x22
        bl      .Llane4_stage
        add     x9, x21, x21, lsl #1
        sub     x9, x9, #3
        add     x22, x22, x9, lsl #5
        lsl     x21, x21, #2
        ldr     x9, [x19, #8]
        cmp     x21, x9
        b.lo    .Llane4_generic_stage_loop

.Llane4_generic_finish:
        ldr     x0, [x19, #PLAN_WORK]
        mov     x1, x20
        ldr     x2, [x19, #8]
        ldr     x3, [x19, #PLAN_FINISH_RE]
        ldr     x4, [x19, #PLAN_FINISH_IM]
        bl      .Llane4_finish
        ldr     x30, [sp, #48]
        ldp     d14, d15, [sp, #112]
        ldp     d12, d13, [sp, #96]
        ldp     d10, d11, [sp, #80]
        ldp     d8, d9, [sp, #64]
        ldp     x23, x24, [sp, #32]
        ldp     x21, x22, [sp, #16]
        ldp     x19, x20, [sp], #128
        mov     w0, wzr
        ret

.Llane4_error:
        mov     w0, #-1
        ret

/*
 * x0=data, x1=mixed permutation, x2=work, x3=row count.
 */
        .p2align 6
.Llane4_base4:
.Llane4_base4_loop:
        ldp     w9, w10, [x1], #8
        ldp     w11, w12, [x1], #8
        add     x13, x0, x9, lsl #3
        ld2     { v0.4s, v1.4s }, [x13]
        add     x13, x0, x10, lsl #3
        ld2     { v2.4s, v3.4s }, [x13]
        add     x13, x0, x11, lsl #3
        ld2     { v4.4s, v5.4s }, [x13]
        add     x13, x0, x12, lsl #3
        ld2     { v6.4s, v7.4s }, [x13]
        LANE4_RADIX4_SOA \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v16, v17, v18, v19, v20, v21, v22, v23
        stp     q0, q1, [x2], #32
        stp     q2, q3, [x2], #32
        stp     q4, q5, [x2], #32
        stp     q6, q7, [x2], #32
        subs    x3, x3, #4
        b.ne    .Llane4_base4_loop
        ret

/*
 * Register-resident FFT8 over four split-complex transforms.
 * x0=data, x1=mixed permutation, x2=work, x3=row count.
 */
        .p2align 6
.Llane4_base8:
.Llane4_base8_loop:
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v0.4s, v1.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v2.4s, v3.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v4.4s, v5.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v6.4s, v7.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v8.4s, v9.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v10.4s, v11.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v12.4s, v13.4s }, [x13]
        ldr     w9, [x1], #4
        add     x13, x0, x9, lsl #3
        ld2     { v14.4s, v15.4s }, [x13]

        fadd    v16.4s, v0.4s, v2.4s
        fadd    v17.4s, v1.4s, v3.4s
        fsub    v24.4s, v0.4s, v2.4s
        fsub    v25.4s, v1.4s, v3.4s
        fadd    v18.4s, v4.4s, v6.4s
        fadd    v19.4s, v5.4s, v7.4s
        fsub    v26.4s, v4.4s, v6.4s
        fsub    v27.4s, v5.4s, v7.4s
        fadd    v20.4s, v8.4s, v10.4s
        fadd    v21.4s, v9.4s, v11.4s
        fsub    v28.4s, v8.4s, v10.4s
        fsub    v29.4s, v9.4s, v11.4s
        fadd    v22.4s, v12.4s, v14.4s
        fadd    v23.4s, v13.4s, v15.4s
        fsub    v30.4s, v12.4s, v14.4s
        fsub    v31.4s, v13.4s, v15.4s

        mov     v0.16b, v24.16b
        mov     v1.16b, v25.16b
        adrp    x9, .Llane4_sqrt_half
        add     x9, x9, :lo12:.Llane4_sqrt_half
        ldp     q8, q9, [x9]
        fadd    v2.4s, v26.4s, v27.4s
        fsub    v3.4s, v27.4s, v26.4s
        fmul    v2.4s, v2.4s, v8.4s
        fmul    v3.4s, v3.4s, v8.4s
        mov     v4.16b, v29.16b
        fneg    v5.4s, v28.4s
        fsub    v6.4s, v31.4s, v30.4s
        fadd    v7.4s, v30.4s, v31.4s
        fmul    v6.4s, v6.4s, v8.4s
        fmul    v7.4s, v7.4s, v9.4s

        LANE4_RADIX4_SOA \
                v16, v17, v18, v19, v20, v21, v22, v23, \
                v16, v17, v18, v19, v20, v21, v22, v23, \
                v24, v25, v26, v27, v28, v29, v30, v31
        LANE4_RADIX4_SOA \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v24, v25, v26, v27, v28, v29, v30, v31

        stp     q16, q17, [x2], #32
        stp     q0, q1, [x2], #32
        stp     q18, q19, [x2], #32
        stp     q2, q3, [x2], #32
        stp     q20, q21, [x2], #32
        stp     q4, q5, [x2], #32
        stp     q22, q23, [x2], #32
        stp     q6, q7, [x2], #32
        subs    x3, x3, #8
        b.ne    .Llane4_base8_loop
        ret

/*
 * x0=work, x1=previous row count, x2=total row count,
 * x3=replicated root stream.
 */
        .p2align 6
.Llane4_stage:
        lsl     x5, x1, #5
        lsl     x6, x2, #5
        add     x13, x0, x6

.Llane4_stage_block:
        add     x7, x0, x5
        add     x8, x7, x5
        add     x9, x8, x5
        add     x10, x9, x5
        ldp     q0, q1, [x0]
        ldp     q2, q3, [x7]
        ldp     q4, q5, [x8]
        ldp     q6, q7, [x9]
        LANE4_RADIX4_SOA \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v0, v1, v2, v3, v4, v5, v6, v7, \
                v16, v17, v18, v19, v20, v21, v22, v23
        stp     q0, q1, [x0], #32
        stp     q2, q3, [x7], #32
        stp     q4, q5, [x8], #32
        stp     q6, q7, [x9], #32
        mov     x12, x3
        sub     x11, x1, #1
        cbz     x11, .Llane4_stage_next

.Llane4_stage_butterfly:
        ldp     q0, q1, [x0]
        ldp     q2, q3, [x7]
        ldp     q4, q5, [x8]
        ldp     q6, q7, [x9]
        ldp     q24, q25, [x12], #32
        ldp     q26, q27, [x12], #32
        ldp     q28, q29, [x12], #32
        fmul    v16.4s, v2.4s, v24.4s
        fmul    v17.4s, v3.4s, v24.4s
        fmls    v16.4s, v3.4s, v25.4s
        fmla    v17.4s, v2.4s, v25.4s
        fmul    v18.4s, v4.4s, v26.4s
        fmul    v19.4s, v5.4s, v26.4s
        fmls    v18.4s, v5.4s, v27.4s
        fmla    v19.4s, v4.4s, v27.4s
        fmul    v20.4s, v6.4s, v28.4s
        fmul    v21.4s, v7.4s, v28.4s
        fmls    v20.4s, v7.4s, v29.4s
        fmla    v21.4s, v6.4s, v29.4s
        LANE4_RADIX4_SOA \
                v0, v1, v16, v17, v18, v19, v20, v21, \
                v0, v1, v16, v17, v18, v19, v20, v21, \
                v2, v3, v4, v5, v6, v7, v22, v23
        stp     q0, q1, [x0], #32
        stp     q16, q17, [x7], #32
        stp     q18, q19, [x8], #32
        stp     q20, q21, [x9], #32
        subs    x11, x11, #1
        b.ne    .Llane4_stage_butterfly

.Llane4_stage_next:
        mov     x0, x10
        cmp     x0, x13
        b.lo    .Llane4_stage_block
        ret

/*
 * x0=work, x1=output, x2=inner row count,
 * x3=finish real roots, x4=finish imaginary roots.
 */
        .p2align 6
.Llane4_finish:
        lsl     x5, x2, #3
        add     x7, x1, x5
        add     x8, x7, x5
        add     x9, x8, x5
        lsl     x6, x2, #5
        add     x11, x0, x6
        lsl     x12, x2, #2

.Llane4_finish_loop:
        ldp     q0, q4, [x0, #0]
        ldp     q1, q5, [x0, #32]
        ldp     q2, q6, [x0, #64]
        ldp     q3, q7, [x0, #96]
        LANE4_TRANSPOSE4 v0, v1, v2, v3, v16, v17, v18, v19
        LANE4_TRANSPOSE4 v4, v5, v6, v7, v20, v21, v22, v23

        ldr     q24, [x3]
        ldr     q25, [x4]
        fmul    v0.4s, v17.4s, v24.4s
        fmul    v1.4s, v21.4s, v24.4s
        fmls    v0.4s, v21.4s, v25.4s
        fmla    v1.4s, v17.4s, v25.4s
        ldr     q24, [x3, x12]
        ldr     q25, [x4, x12]
        fmul    v2.4s, v18.4s, v24.4s
        fmul    v3.4s, v22.4s, v24.4s
        fmls    v2.4s, v22.4s, v25.4s
        fmla    v3.4s, v18.4s, v25.4s
        add     x13, x3, x12, lsl #1
        ldr     q24, [x13]
        add     x13, x4, x12, lsl #1
        ldr     q25, [x13]
        fmul    v4.4s, v19.4s, v24.4s
        fmul    v5.4s, v23.4s, v24.4s
        fmls    v4.4s, v23.4s, v25.4s
        fmla    v5.4s, v19.4s, v25.4s

        LANE4_RADIX4_SOA \
                v16, v20, v0, v1, v2, v3, v4, v5, \
                v16, v17, v18, v19, v20, v21, v22, v23, \
                v6, v7, v24, v25, v26, v27, v28, v29
        st2     { v16.4s, v17.4s }, [x1], #32
        st2     { v18.4s, v19.4s }, [x7], #32
        st2     { v20.4s, v21.4s }, [x8], #32
        st2     { v22.4s, v23.4s }, [x9], #32
        add     x0, x0, #128
        add     x3, x3, #16
        add     x4, x4, #16
        cmp     x0, x11
        b.lo    .Llane4_finish_loop
        ret

        .size   lane4_neon_execute, .-lane4_neon_execute

        .section .note.GNU-stack,"",%progbits
