/*
 * Two-lane AArch64 NEON FFT.
 *
 * Every Q register carries two complete interleaved complex values:
 *
 *   [F0.re, F0.im, F1.re, F1.im]
 *
 * The inner FFT evaluates the even and odd input-residue transforms
 * together.  The final loop transposes two adjacent rows, applies two
 * frequency-specific roots, and evaluates two FFT2 butterflies in parallel.
 */

#define PLAN_INNER_SIZE       8
#define PLAN_INNER_LEVELS    16
#define PLAN_MIXED_PERM      24
#define PLAN_REPLICATED_ROOT 32
#define PLAN_FINISH_ROOT     40
#define PLAN_WORK            48

        .section .rodata
        .p2align 4
.Llane2_sign_odd:
        .word   0x00000000, 0x80000000, 0x00000000, 0x80000000
.Llane2_positive_diagonal:
        .float  0.7071067811865475244
        .float  0.7071067811865475244
        .float  0.7071067811865475244
        .float  0.7071067811865475244
.Llane2_diagonal_signed_im:
        .float  0.7071067811865475244
        .float -0.7071067811865475244
        .float  0.7071067811865475244
        .float -0.7071067811865475244
.Llane2_negative_diagonal:
        .float -0.7071067811865475244
        .float -0.7071067811865475244
        .float -0.7071067811865475244
        .float -0.7071067811865475244

        .text

/*
 * Inputs v0-v3; outputs v20-v23; scratch v16-v19 and v24.
 * v31 contains the sign mask for multiplication by -i.
 */
        .macro  LANE2_RADIX4
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
        .endm

        .macro  LANE2_LOAD_ROWS
        ldr     q0, [x0]
        ldr     q1, [x4]
        ldr     q2, [x5]
        ldr     q3, [x6]
        .endm

        .macro  LANE2_STORE_ROWS
        str     q20, [x0], #16
        str     q21, [x4], #16
        str     q22, [x5], #16
        str     q23, [x6], #16
        .endm

/*
 * Rotate the three nonzero radix-4 legs with one linear 96-byte root
 * packet.  The real and signed-imaginary vectors allow a baseline-NEON
 * complex product to use REV64, FMUL, FMLA.
 */
        .macro  LANE2_GENERAL_BUTTERFLY
        LANE2_LOAD_ROWS
        ldp     q16, q17, [x12], #96
        ldp     q18, q19, [x12, #-64]
        ldp     q20, q21, [x12, #-32]
        rev64   v22.4s, v1.4s
        rev64   v23.4s, v2.4s
        rev64   v24.4s, v3.4s
        fmul    v1.4s, v1.4s, v16.4s
        fmul    v2.4s, v2.4s, v18.4s
        fmul    v3.4s, v3.4s, v20.4s
        fmla    v1.4s, v22.4s, v17.4s
        fmla    v2.4s, v23.4s, v19.4s
        fmla    v3.4s, v24.4s, v21.4s
        LANE2_RADIX4
        LANE2_STORE_ROWS
        .endm

        .p2align 6
        .global lane2_neon_execute
        .type   lane2_neon_execute, %function
lane2_neon_execute:
        cbz     x0, .Lexecute_error
        cbz     x1, .Lexecute_error
        ldr     x9, [x0, #PLAN_REPLICATED_ROOT]
        cbz     x9, .Lexecute_error
        ldr     x9, [x0, #PLAN_INNER_SIZE]
        cmp     x9, #8
        b.eq    lane2_neon_direct16_path

        stp     x29, x30, [sp, #-48]!
        mov     x29, sp
        stp     x19, x20, [sp, #16]
        stp     x21, x22, [sp, #32]
        mov     x19, x0
        mov     x20, x1

        /*
         * AArch64 has no packed addsub instruction.  Keep the selective
         * sign mask and the exact diagonal roots live for the complete
         * transform instead of reloading them at each internal stage.
         * These are caller-saved vector registers and every BL target below
         * is private assembly under our control.
         */
        adrp    x9, .Llane2_sign_odd
        add     x9, x9, :lo12:.Llane2_sign_odd
        ldp     q31, q28, [x9]
        ldp     q29, q30, [x9, #32]

        mov     x0, x20
        ldr     x1, [x19, #PLAN_MIXED_PERM]
        ldr     x2, [x19, #PLAN_WORK]
        ldr     x3, [x19, #PLAN_INNER_SIZE]
        ldr     w9, [x19, #PLAN_INNER_LEVELS]
        tbz     w9, #0, .Lexecute_base4
        bl      .Llane2_base8
        mov     x21, #8
        b       .Lexecute_stage_test

.Lexecute_base4:
        bl      .Llane2_base4
        mov     x21, #4

.Lexecute_stage_test:
        ldr     x9, [x19, #PLAN_INNER_SIZE]
        cmp     x21, x9
        b.hs    .Lexecute_finish
        ldr     x22, [x19, #PLAN_REPLICATED_ROOT]

.Lexecute_stage_loop:
        ldr     x0, [x19, #PLAN_WORK]
        mov     x1, x21
        ldr     x2, [x19, #PLAN_INNER_SIZE]
        mov     x3, x22
        bl      .Llane2_stage
        add     x9, x21, x21, lsl #1
        sub     x9, x9, #3
        add     x22, x22, x9, lsl #5
        lsl     x21, x21, #2
        ldr     x9, [x19, #PLAN_INNER_SIZE]
        cmp     x21, x9
        b.lo    .Lexecute_stage_loop

.Lexecute_finish:
        ldr     x0, [x19, #PLAN_WORK]
        mov     x1, x20
        ldr     x2, [x19, #PLAN_INNER_SIZE]
        ldr     x3, [x19, #PLAN_FINISH_ROOT]
        bl      .Llane2_finish

        ldp     x21, x22, [sp, #32]
        ldp     x19, x20, [sp, #16]
        ldp     x29, x30, [sp], #48
        mov     w0, wzr
        ret

.Lexecute_error:
        mov     w0, #-1
        ret
        .size   lane2_neon_execute, .-lane2_neon_execute

/*
 * Fused N=16 lane2 codelet.
 *
 * Four paired loads put the even and odd N=8 residues in the same Q
 * register.  The inner FFT8 stays entirely in registers and the top FFT2
 * writes natural-order output directly, eliminating the generic gather,
 * work-buffer stores/loads, two BL/RET pairs, and loop control.
 *
 * x0=plan, x1=in-place data.  This is an internal tail-called entry.
 */
        .macro  LANE2_DIRECT16_FINISH first, second
        trn1    v0.2d, \first\().2d, \second\().2d
        trn2    v1.2d, \first\().2d, \second\().2d
        ldp     q16, q17, [x3], #32
        rev64   v18.4s, v1.4s
        fmul    v1.4s, v1.4s, v16.4s
        fmla    v1.4s, v18.4s, v17.4s
        fadd    v2.4s, v0.4s, v1.4s
        fsub    v3.4s, v0.4s, v1.4s
        str     q2, [x1], #16
        str     q3, [x4], #16
        .endm

        .p2align 6
        .type   lane2_neon_direct16_path, %function
lane2_neon_direct16_path:
        adrp    x9, .Llane2_sign_odd
        add     x9, x9, :lo12:.Llane2_sign_odd
        ldp     q31, q28, [x9]
        ldp     q29, q30, [x9, #32]
        ldr     x3, [x0, #PLAN_FINISH_ROOT]

        /*
         * Register order is the mixed 2x4 reversal of the eight paired
         * input rows: 0,4,1,5,2,6,3,7.  Sequential LDPs still cover the
         * natural input without a permutation table.
         */
        ldp     q0, q2, [x1, #0]
        ldp     q4, q6, [x1, #32]
        ldp     q1, q3, [x1, #64]
        ldp     q5, q7, [x1, #96]

        fadd    v16.4s, v0.4s, v1.4s
        fsub    v20.4s, v0.4s, v1.4s
        fadd    v17.4s, v2.4s, v3.4s
        fsub    v21.4s, v2.4s, v3.4s
        fadd    v18.4s, v4.4s, v5.4s
        fsub    v22.4s, v4.4s, v5.4s
        fadd    v19.4s, v6.4s, v7.4s
        fsub    v23.4s, v6.4s, v7.4s

        rev64   v24.4s, v21.4s
        fmul    v21.4s, v21.4s, v28.4s
        fmla    v21.4s, v24.4s, v29.4s
        rev64   v22.4s, v22.4s
        eor     v22.16b, v22.16b, v31.16b
        rev64   v24.4s, v23.4s
        fmul    v23.4s, v23.4s, v30.4s
        fmla    v23.4s, v24.4s, v29.4s

        fadd    v0.4s, v16.4s, v18.4s
        fsub    v1.4s, v16.4s, v18.4s
        fadd    v2.4s, v17.4s, v19.4s
        fsub    v3.4s, v17.4s, v19.4s
        rev64   v24.4s, v3.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v4.4s, v0.4s, v2.4s
        fadd    v5.4s, v1.4s, v24.4s
        fsub    v6.4s, v0.4s, v2.4s
        fsub    v7.4s, v1.4s, v24.4s

        fadd    v16.4s, v20.4s, v22.4s
        fsub    v17.4s, v20.4s, v22.4s
        fadd    v18.4s, v21.4s, v23.4s
        fsub    v19.4s, v21.4s, v23.4s
        rev64   v24.4s, v19.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v20.4s, v16.4s, v18.4s
        fadd    v21.4s, v17.4s, v24.4s
        fsub    v22.4s, v16.4s, v18.4s
        fsub    v23.4s, v17.4s, v24.4s

        add     x4, x1, #64
        LANE2_DIRECT16_FINISH v4, v20
        LANE2_DIRECT16_FINISH v5, v21
        LANE2_DIRECT16_FINISH v6, v22
        LANE2_DIRECT16_FINISH v7, v23
        mov     w0, wzr
        ret
        .size   lane2_neon_direct16_path, .-lane2_neon_direct16_path

/*
 * x0=data, x1=mixed permutation, x2=work, x3=inner row count.
 */
        .p2align 6
.Llane2_base4:
        mov     x8, xzr
.Lbase4_loop:
        add     x13, x1, x8, lsl #2
        ldp     w9, w10, [x13]
        ldp     w11, w12, [x13, #8]
        add     x14, x0, x9, lsl #3
        ldr     q0, [x14]
        add     x14, x0, x10, lsl #3
        ldr     q1, [x14]
        add     x15, x0, x11, lsl #3
        ldr     q2, [x15]
        add     x15, x0, x12, lsl #3
        ldr     q3, [x15]

        LANE2_RADIX4
        add     x15, x2, x8, lsl #4
        stp     q20, q21, [x15]
        stp     q22, q23, [x15, #32]
        add     x8, x8, #4
        cmp     x8, x3
        b.lo    .Lbase4_loop
        ret

/*
 * x0=data, x1=mixed permutation, x2=work, x3=inner row count.
 */
        .p2align 6
.Llane2_base8:
        mov     x8, xzr
.Lbase8_loop:
        add     x17, x1, x8, lsl #2
        ldp     w9, w10, [x17]
        ldp     w11, w12, [x17, #8]
        ldp     w13, w14, [x17, #16]
        ldp     w15, w16, [x17, #24]
        add     x17, x0, x9, lsl #3
        ldr     q0, [x17]
        add     x17, x0, x10, lsl #3
        ldr     q1, [x17]
        add     x17, x0, x11, lsl #3
        ldr     q2, [x17]
        add     x17, x0, x12, lsl #3
        ldr     q3, [x17]
        add     x17, x0, x13, lsl #3
        ldr     q4, [x17]
        add     x17, x0, x14, lsl #3
        ldr     q5, [x17]
        add     x17, x0, x15, lsl #3
        ldr     q6, [x17]
        add     x17, x0, x16, lsl #3
        ldr     q7, [x17]

        fadd    v16.4s, v0.4s, v1.4s
        fsub    v20.4s, v0.4s, v1.4s
        fadd    v17.4s, v2.4s, v3.4s
        fsub    v21.4s, v2.4s, v3.4s
        fadd    v18.4s, v4.4s, v5.4s
        fsub    v22.4s, v4.4s, v5.4s
        fadd    v19.4s, v6.4s, v7.4s
        fsub    v23.4s, v6.4s, v7.4s

        rev64   v24.4s, v21.4s
        fmul    v21.4s, v21.4s, v28.4s
        fmla    v21.4s, v24.4s, v29.4s
        rev64   v22.4s, v22.4s
        eor     v22.16b, v22.16b, v31.16b
        rev64   v24.4s, v23.4s
        fmul    v23.4s, v23.4s, v30.4s
        fmla    v23.4s, v24.4s, v29.4s

        fadd    v0.4s, v16.4s, v18.4s
        fsub    v1.4s, v16.4s, v18.4s
        fadd    v2.4s, v17.4s, v19.4s
        fsub    v3.4s, v17.4s, v19.4s
        rev64   v24.4s, v3.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v4.4s, v0.4s, v2.4s
        fadd    v5.4s, v1.4s, v24.4s
        fsub    v6.4s, v0.4s, v2.4s
        fsub    v7.4s, v1.4s, v24.4s

        fadd    v16.4s, v20.4s, v22.4s
        fsub    v17.4s, v20.4s, v22.4s
        fadd    v18.4s, v21.4s, v23.4s
        fsub    v19.4s, v21.4s, v23.4s
        rev64   v24.4s, v19.4s
        eor     v24.16b, v24.16b, v31.16b
        fadd    v20.4s, v16.4s, v18.4s
        fadd    v21.4s, v17.4s, v24.4s
        fsub    v22.4s, v16.4s, v18.4s
        fsub    v23.4s, v17.4s, v24.4s

        add     x17, x2, x8, lsl #4
        str     q4,  [x17]
        str     q20, [x17, #16]
        str     q5,  [x17, #32]
        str     q21, [x17, #48]
        str     q6,  [x17, #64]
        str     q22, [x17, #80]
        str     q7,  [x17, #96]
        str     q23, [x17, #112]
        add     x8, x8, #8
        cmp     x8, x3
        b.lo    .Lbase8_loop
        ret

/*
 * x0=work, x1=previous row count, x2=inner row count, x3=stage roots.
 */
        .p2align 6
.Llane2_stage:
        lsl     x1, x1, #4
        lsl     x2, x2, #4
        add     x13, x0, x2

.Lstage_block_loop:
        add     x4, x0, x1
        add     x5, x4, x1
        add     x6, x5, x1
        add     x7, x6, x1
        mov     x8, x4
        lsr     x9, x1, #1
        add     x9, x0, x9

        LANE2_LOAD_ROWS
        LANE2_RADIX4
        LANE2_STORE_ROWS
        mov     x12, x3

        LANE2_GENERAL_BUTTERFLY

.Lstage_before_special:
        cmp     x0, x9
        b.hs    .Lstage_at_special
        LANE2_GENERAL_BUTTERFLY
        LANE2_GENERAL_BUTTERFLY
        b       .Lstage_before_special

.Lstage_at_special:
        LANE2_LOAD_ROWS
        rev64   v16.4s, v1.4s
        fmul    v1.4s, v1.4s, v28.4s
        fmla    v1.4s, v16.4s, v29.4s
        rev64   v2.4s, v2.4s
        eor     v2.16b, v2.16b, v31.16b
        rev64   v16.4s, v3.4s
        fmul    v3.4s, v3.4s, v30.4s
        fmla    v3.4s, v16.4s, v29.4s
        LANE2_RADIX4
        LANE2_STORE_ROWS
        add     x12, x12, #96

        LANE2_GENERAL_BUTTERFLY

.Lstage_after_special:
        cmp     x0, x8
        b.hs    .Lstage_next_block
        LANE2_GENERAL_BUTTERFLY
        LANE2_GENERAL_BUTTERFLY
        b       .Lstage_after_special

.Lstage_next_block:
        mov     x0, x7
        cmp     x0, x13
        b.lo    .Lstage_block_loop
        ret

/*
 * x0=work, x1=output, x2=inner row count, x3=packed finish roots.
 */
        .p2align 6
.Llane2_finish:
        add     x4, x1, x2, lsl #3
        add     x5, x0, x2, lsl #4
.Lfinish_loop:
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
        b.lo    .Lfinish_loop
        ret

        /*
         * Marker used by the instruction-trace harness.  The public symbol
         * ends before the private leaves, so its ordinary ELF size alone
         * does not cover all dynamically executed lane2 code.
         */
        .hidden lane2_neon_code_end
        .type   lane2_neon_code_end, %function
lane2_neon_code_end:
        .size   lane2_neon_code_end, .-lane2_neon_code_end

        .section .note.GNU-stack,"",%progbits
