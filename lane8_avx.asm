; Eight-lane split-complex FFT for AVX2/FMA x86-64.
;
; One work row is
;   ymm real = [F0.re ... F7.re]
;   ymm imag = [F0.im ... F7.im]
; and therefore evaluates eight decimated residue transforms together.
; Unlike the interleaved AVX representation, a split-complex rotation of
; eight values takes two multiplies plus two FMAs and no lane-swap uops.

%define private_prefix lane8
%include "libavutil/x86/x86inc.asm"

SECTION_RODATA 32

align 32
sqrt_half:
    times 8 dd 0.7071067811865475244
neg_sqrt_half:
    times 8 dd -0.7071067811865475244
sign_all:
    times 8 dd 0x80000000

SECTION .text

%define PLAN_N                0
%define PLAN_INNER_SIZE       8
%define PLAN_INNER_LEVELS    16
%define PLAN_PERMUTATION     24
%define PLAN_ROOT            32
%define PLAN_FINISH_RE       40
%define PLAN_FINISH_IM       48
%define PLAN_WORK            56

; In-place sum/difference using no temporary vector.
%macro BUTTERFLY_NO_TEMP 2
    addps %1, %2
    addps %2, %2
    subps %2, %1
    xorps %2, [sign_all]
%endmacro

; Load eight adjacent AoS complex floats and split them into two YMM vectors.
; The final vpermq repairs the 128-bit-local ordering produced by vshufps.
%macro LOAD_INPUT8 3
    movups m14, [%3]
    movups m15, [%3 + 32]
    shufps %1, m14, m15, 0x88
    shufps %2, m14, m15, 0xdd
    vpermq %1, %1, 0xd8
    vpermq %2, %2, 0xd8
%endmacro

; Last-row variant when all sixteen vector registers are live.
%macro LOAD_INPUT8_LAST 1
    shufps m15, m15, [%1], 0xdd
    shufps m14, m14, [%1 + 32], 0x88
    vpermq m15, m15, 0x8d
    vpermq m14, m14, 0xd8
%endmacro

; (%1 + i%2) *= replicated (%3.re + i%3.im).
; m14/m15 are scratch.  The four arithmetic instructions are the FMA
; lower bound for a generic split-complex rotation.
%macro COMPLEX_MULTIPLY_FMA 3
    mulps   m14, %1, [%3 + 32]
    fmaddps m14, %2, [%3], m14
    mulps   m15, %2, [%3 + 32]
    fmsubps %1, %1, [%3], m15
    movaps  %2, m14
%endmacro

; Inputs m0:m1, m2:m3, m4:m5, m6:m7 are split complex rows.
%macro RADIX4_STORE_POINTERS 4
    movaps m8, m0
    movaps m9, m1
    addps  m8, m4
    addps  m9, m5
    subps  m0, m4
    subps  m1, m5

    movaps m10, m2
    movaps m11, m3
    addps  m10, m6
    addps  m11, m7
    subps  m2, m6
    subps  m3, m7

    movaps m12, m8
    movaps m13, m9
    addps  m12, m10
    addps  m13, m11
    subps  m8, m10
    subps  m9, m11

    movaps m10, m0
    movaps m11, m1
    addps  m10, m3
    subps  m11, m2
    subps  m0, m3
    addps  m1, m2

    movaps [%1], m12
    movaps [%1 + 32], m13
    movaps [%2], m10
    movaps [%2 + 32], m11
    movaps [%3], m8
    movaps [%3 + 32], m9
    movaps [%4], m0
    movaps [%4 + 32], m1
%endmacro

%macro LOAD_POINTER_ROWS 0
    movaps m0, [workq]
    movaps m1, [workq + 32]
    movaps m2, [p1q]
    movaps m3, [p1q + 32]
    movaps m4, [p2q]
    movaps m5, [p2q + 32]
    movaps m6, [p3q]
    movaps m7, [p3q + 32]
%endmacro

%macro ADVANCE_POINTER_ROWS 0
    add workq, 64
    add p1q, 64
    add p2q, 64
    add p3q, 64
%endmacro

%macro STAGE_GENERAL_BUTTERFLY 0
    LOAD_POINTER_ROWS
    COMPLEX_MULTIPLY_FMA m2, m3, rootsq
    COMPLEX_MULTIPLY_FMA m4, m5, rootsq + 64
    COMPLEX_MULTIPLY_FMA m6, m7, rootsq + 128
    RADIX4_STORE_POINTERS workq, p1q, p2q, p3q
    add rootsq, 192
    ADVANCE_POINTER_ROWS
%endmacro

INIT_YMM fma3
cglobal avx_base4, 4, 7, 16, data, permutation, work, count, index, input, a
    xor indexq, indexq
.loop:
    mov inputd, [permutationq + indexq*4 + 0]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m0, m1, inputq
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m2, m3, inputq
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m4, m5, inputq
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m6, m7, inputq
    mov aq, indexq
    shl aq, 6
    add aq, workq
    RADIX4_STORE_POINTERS aq, aq + 64, aq + 128, aq + 192
    add indexq, 4
    cmp indexq, countq
    jb .loop
    RET

; Register-resident FFT8 leaf.  This is the same reduced-operation DIF
; codelet structure as FFmpeg's short transforms, expanded across eight
; independent residue transforms.
INIT_YMM fma3
cglobal avx_base8, 4, 7, 16, data, permutation, work, count, index, input, a
    xor indexq, indexq
.loop:
    mov inputd, [permutationq + indexq*4 + 0]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m0, m1, inputq
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m2, m3, inputq
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m4, m5, inputq
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m6, m7, inputq
    mov inputd, [permutationq + indexq*4 + 16]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m8, m9, inputq
    mov inputd, [permutationq + indexq*4 + 20]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m10, m11, inputq
    mov inputd, [permutationq + indexq*4 + 24]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT8 m12, m13, inputq
    mov inputd, [permutationq + indexq*4 + 28]
    lea inputq, [dataq + inputq*8]
    movups m14, [inputq]
    movups m15, [inputq + 32]
    LOAD_INPUT8_LAST inputq

    BUTTERFLY_NO_TEMP m0, m2
    BUTTERFLY_NO_TEMP m1, m3
    BUTTERFLY_NO_TEMP m4, m6
    BUTTERFLY_NO_TEMP m5, m7
    BUTTERFLY_NO_TEMP m8, m10
    BUTTERFLY_NO_TEMP m9, m11
    BUTTERFLY_NO_TEMP m12, m14
    BUTTERFLY_NO_TEMP m13, m15

    addps  m6, m7
    addps  m7, m7
    subps  m7, m6
    mulps  m6, [sqrt_half]
    mulps  m7, [sqrt_half]
    xorps  m10, [sign_all]
    addps  m14, m15
    addps  m15, m15
    subps  m15, m14
    mulps  m14, [neg_sqrt_half]
    mulps  m15, [sqrt_half]

    BUTTERFLY_NO_TEMP m0, m8
    BUTTERFLY_NO_TEMP m1, m9
    BUTTERFLY_NO_TEMP m4, m12
    BUTTERFLY_NO_TEMP m5, m13
    BUTTERFLY_NO_TEMP m0, m4
    BUTTERFLY_NO_TEMP m1, m5
    BUTTERFLY_NO_TEMP m8, m13
    BUTTERFLY_NO_TEMP m9, m12

    BUTTERFLY_NO_TEMP m2, m11
    BUTTERFLY_NO_TEMP m3, m10
    BUTTERFLY_NO_TEMP m6, m15
    BUTTERFLY_NO_TEMP m7, m14
    BUTTERFLY_NO_TEMP m2, m6
    BUTTERFLY_NO_TEMP m3, m7
    BUTTERFLY_NO_TEMP m11, m14
    BUTTERFLY_NO_TEMP m10, m15

    mov aq, indexq
    shl aq, 6
    add aq, workq
    movaps [aq + 0], m0
    movaps [aq + 32], m1
    movaps [aq + 64], m2
    movaps [aq + 96], m3
    movaps [aq + 128], m8
    movaps [aq + 160], m12
    movaps [aq + 192], m11
    movaps [aq + 224], m15
    movaps [aq + 256], m4
    movaps [aq + 288], m5
    movaps [aq + 320], m6
    movaps [aq + 352], m7
    movaps [aq + 384], m13
    movaps [aq + 416], m9
    movaps [aq + 448], m14
    movaps [aq + 480], m10

    add indexq, 8
    cmp indexq, countq
    jb .loop
    RET

INIT_YMM fma3
cglobal avx_stage, 4, 12, 16, work, previous, inner, root, p1, p2, p3, rowend, next, innerend, roots, special
    shl previousq, 6
    shl innerq, 6
    lea innerendq, [workq + innerq]

.block_loop:
    lea p1q, [workq + previousq]
    lea p2q, [p1q + previousq]
    lea p3q, [p2q + previousq]
    lea nextq, [p3q + previousq]
    mov rowendq, p1q
    mov specialq, previousq
    shr specialq, 1
    add specialq, workq
    LOAD_POINTER_ROWS
    RADIX4_STORE_POINTERS workq, p1q, p2q, p3q
    ADVANCE_POINTER_ROWS
    mov rootsq, rootq

.before_special:
    cmp workq, specialq
    jae .special
    STAGE_GENERAL_BUTTERFLY
    jmp .before_special

.special:
    ; k = previous/2: W8, -i, W8^3.
    LOAD_POINTER_ROWS
    movaps m14, m2
    movaps m15, m3
    addps  m2, m3
    subps  m15, m14
    mulps  m2, [sqrt_half]
    mulps  m15, [sqrt_half]
    movaps m3, m15

    movaps m14, m4
    movaps m4, m5
    movaps m5, m14
    xorps  m5, [sign_all]

    movaps m14, m7
    subps  m14, m6
    addps  m6, m7
    mulps  m14, [sqrt_half]
    mulps  m6, [neg_sqrt_half]
    movaps m7, m6
    movaps m6, m14
    RADIX4_STORE_POINTERS workq, p1q, p2q, p3q
    add rootsq, 192
    ADVANCE_POINTER_ROWS

.after_special:
    cmp workq, rowendq
    jae .next_block
    STAGE_GENERAL_BUTTERFLY
    jmp .after_special

.next_block:
    mov workq, nextq
    cmp workq, innerendq
    jb .block_loop
    RET

; Four-by-four XMM transpose using two explicit scratch registers.
%macro TRANSPOSE4_XMM 6
    movaps %5, %1
    movaps %6, %3
    unpcklps %1, %2
    unpckhps %5, %2
    unpcklps %3, %4
    unpckhps %6, %4
    movaps %2, %1
    movlhps %1, %3
    movhlps %3, %2
    movaps %2, %5
    movlhps %2, %6
    movhlps %6, %5
    movaps %4, %6
%endmacro

; Destructive FFT4 on four arbitrary split-complex XMM values.
; Results are E0=%1:%5, E1=%3:%4, E2=%2:%6, E3=%8:%7.
%macro FFT4_SPLIT_NO_TEMP 8
    BUTTERFLY_NO_TEMP %1, %3
    BUTTERFLY_NO_TEMP %5, %7
    BUTTERFLY_NO_TEMP %2, %4
    BUTTERFLY_NO_TEMP %6, %8
    BUTTERFLY_NO_TEMP %1, %2
    BUTTERFLY_NO_TEMP %5, %6
    BUTTERFLY_NO_TEMP %3, %8
    BUTTERFLY_NO_TEMP %7, %4
%endmacro

; Store four complex values held in two XMM vectors. %3 is scratch.
%macro STORE_COMPLEX4_XMM 4
    movaps %3, %1
    unpcklps %1, %2
    unpckhps %3, %2
    movups [%4], %1
    movups [%4 + 16], %3
%endmacro

; Split-complex XMM multiply using xm0/xm8 as temporaries.  Factor pointers
; are advanced by one lane block after each use.
%macro FINISH_MULTIPLY 2
    mulps   xm0, %1, [finishimq]
    fmaddps xm0, %2, [finishreq], xm0
    mulps   xm8, %2, [finishimq]
    fmsubps %1, %1, [finishreq], xm8
    movaps  %2, xm0
    add finishreq, innerq
    add finishimq, innerq
%endmacro

; work, output, inner_size, finish_re, finish_im
INIT_YMM fma3
cglobal avx_finish, 5, 7, 16, 64, work, output, inner, finishre, finishim, end, p
    shl innerq, 2
    lea endq, [finishreq + innerq]

.loop:
    ; Real columns 0..3.
    movaps xm0, [workq + 0]
    movaps xm1, [workq + 64]
    movaps xm2, [workq + 128]
    movaps xm3, [workq + 192]
    TRANSPOSE4_XMM xm0, xm1, xm2, xm3, xm4, xm5

    ; Real columns 4..7.
    movaps m4, [workq + 0]
    vextractf128 xm4, m4, 1
    movaps m5, [workq + 64]
    vextractf128 xm5, m5, 1
    movaps m6, [workq + 128]
    vextractf128 xm6, m6, 1
    movaps m7, [workq + 192]
    vextractf128 xm7, m7, 1
    TRANSPOSE4_XMM xm4, xm5, xm6, xm7, xm8, xm9

    ; Imaginary columns 0..3.
    movaps xm8, [workq + 32]
    movaps xm9, [workq + 96]
    movaps xm10, [workq + 160]
    movaps xm11, [workq + 224]
    TRANSPOSE4_XMM xm8, xm9, xm10, xm11, xm12, xm13

    ; Imaginary columns 4..7.  Spill two live re columns to make the
    ; transpose scratch explicit and keep the hot transform spill-free.
    movaps [rsp + 0], xm0
    movaps [rsp + 16], xm1
    movaps m12, [workq + 32]
    vextractf128 xm12, m12, 1
    movaps m13, [workq + 96]
    vextractf128 xm13, m13, 1
    movaps m14, [workq + 160]
    vextractf128 xm14, m14, 1
    movaps m15, [workq + 224]
    vextractf128 xm15, m15, 1
    TRANSPOSE4_XMM xm12, xm13, xm14, xm15, xm0, xm1
    movaps xm0, [rsp + 0]
    movaps xm1, [rsp + 16]

    ; TRANSPOSE4_XMM naturally returns columns 0,2,1,3.  Put both halves
    ; into strict lane order before applying lane-indexed final twiddles.
    movaps [rsp + 0], xm1
    movaps xm1, xm2
    movaps xm2, [rsp + 0]
    movaps [rsp + 0], xm5
    movaps xm5, xm6
    movaps xm6, [rsp + 0]
    movaps [rsp + 0], xm9
    movaps xm9, xm10
    movaps xm10, [rsp + 0]
    movaps [rsp + 0], xm13
    movaps xm13, xm14
    movaps xm14, [rsp + 0]

    ; Keep lane zero on the stack so xm0/xm8 are the two FMA temporaries.
    movaps [rsp + 0], xm0
    movaps [rsp + 16], xm8
    mov [rsp + 32], finishreq
    mov [rsp + 40], finishimq
    FINISH_MULTIPLY xm1, xm9
    FINISH_MULTIPLY xm2, xm10
    FINISH_MULTIPLY xm3, xm11
    FINISH_MULTIPLY xm4, xm12
    FINISH_MULTIPLY xm5, xm13
    FINISH_MULTIPLY xm6, xm14
    FINISH_MULTIPLY xm7, xm15
    mov finishreq, [rsp + 32]
    mov finishimq, [rsp + 40]
    movaps xm0, [rsp + 0]
    movaps xm8, [rsp + 16]

    ; DIF FFT8: pair distance-four values, rotate the odd half, then
    ; evaluate its even- and odd-output FFT4s independently.
    BUTTERFLY_NO_TEMP xm0, xm4
    BUTTERFLY_NO_TEMP xm8, xm12
    BUTTERFLY_NO_TEMP xm1, xm5
    BUTTERFLY_NO_TEMP xm9, xm13
    BUTTERFLY_NO_TEMP xm2, xm6
    BUTTERFLY_NO_TEMP xm10, xm14
    BUTTERFLY_NO_TEMP xm3, xm7
    BUTTERFLY_NO_TEMP xm11, xm15

    addps xm5, xm13
    addps xm13, xm13
    subps xm13, xm5
    mulps xm5, [sqrt_half]
    mulps xm13, [sqrt_half]
    xorps xm6, [sign_all]
    addps xm7, xm15
    addps xm15, xm15
    subps xm15, xm7
    mulps xm7, [neg_sqrt_half]
    mulps xm15, [sqrt_half]

    FFT4_SPLIT_NO_TEMP xm0, xm1, xm2, xm3, xm8, xm9, xm10, xm11
    FFT4_SPLIT_NO_TEMP xm4, xm5, xm14, xm15, xm12, xm13, xm6, xm7

    ; The destructive codelets leave the natural outputs in the register
    ; pairs documented by FFT4_SPLIT_NO_TEMP.
    movaps [rsp + 48], xm7             ; save k=7 real
    mov pq, outputq
    STORE_COMPLEX4_XMM xm0, xm8, xm7, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm4, xm12, xm0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm2, xm3, xm0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm14, xm15, xm0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm1, xm9, xm0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm5, xm13, xm0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX4_XMM xm11, xm10, xm0, pq
    lea pq, [pq + innerq*2]
    movaps xm1, [rsp + 48]
    STORE_COMPLEX4_XMM xm1, xm6, xm0, pq

    add workq, 256
    add outputq, 32
    add finishreq, 16
    add finishimq, 16
    cmp finishreq, endq
    jb .loop
    RET

; Eight-by-eight float transpose. Inputs m0..m7, outputs m8..m15.
%macro TRANSPOSE8_TO_HIGH 0
    unpcklps m8,  m0, m1
    unpckhps m9,  m0, m1
    unpcklps m10, m2, m3
    unpckhps m11, m2, m3
    unpcklps m12, m4, m5
    unpckhps m13, m4, m5
    unpcklps m14, m6, m7
    unpckhps m15, m6, m7

    shufps m0, m8,  m10, 0x44
    shufps m1, m8,  m10, 0xee
    shufps m2, m9,  m11, 0x44
    shufps m3, m9,  m11, 0xee
    shufps m4, m12, m14, 0x44
    shufps m5, m12, m14, 0xee
    shufps m6, m13, m15, 0x44
    shufps m7, m13, m15, 0xee

    vperm2f128 m8,  m0, m4, 0x20
    vperm2f128 m12, m0, m4, 0x31
    vperm2f128 m9,  m1, m5, 0x20
    vperm2f128 m13, m1, m5, 0x31
    vperm2f128 m10, m2, m6, 0x20
    vperm2f128 m14, m2, m6, 0x31
    vperm2f128 m11, m3, m7, 0x20
    vperm2f128 m15, m3, m7, 0x31
%endmacro

; Multiply a split YMM pair with one scratch register. %1 is real, %2 imag.
%macro FINISH_MULTIPLY_ONE_TEMP 2
    movaps m0, %1
    mulps %1, %1, [finishreq]
    fnmaddps %1, %2, [finishimq], %1
    mulps %2, %2, [finishreq]
    fmaddps %2, m0, [finishimq], %2
    add finishreq, innerq
    add finishimq, innerq
%endmacro

; Interleave and store eight complex values with one scratch register.
%macro STORE_COMPLEX8 4
    unpcklps %3, %1, %2
    unpckhps %2, %1, %2
    vperm2f128 %1, %3, %2, 0x20
    vperm2f128 %3, %3, %2, 0x31
    movups [%4], %1
    movups [%4 + 32], %3
%endmacro

; Full-width finalizer.  It transposes eight q rows at a time so the final
; FFT8 evaluates eight frequencies per instruction instead of four.
INIT_YMM fma3
cglobal avx_finish8, 5, 7, 16, 320, work, output, inner, finishre, finishim, end, p
    shl innerq, 2
    lea endq, [finishreq + innerq]
.loop:
    movaps m0, [workq + 0]
    movaps m1, [workq + 64]
    movaps m2, [workq + 128]
    movaps m3, [workq + 192]
    movaps m4, [workq + 256]
    movaps m5, [workq + 320]
    movaps m6, [workq + 384]
    movaps m7, [workq + 448]
    TRANSPOSE8_TO_HIGH
    movaps [rsp + 0], m8
    movaps [rsp + 32], m9
    movaps [rsp + 64], m10
    movaps [rsp + 96], m11
    movaps [rsp + 128], m12
    movaps [rsp + 160], m13
    movaps [rsp + 192], m14
    movaps [rsp + 224], m15

    movaps m0, [workq + 32]
    movaps m1, [workq + 96]
    movaps m2, [workq + 160]
    movaps m3, [workq + 224]
    movaps m4, [workq + 288]
    movaps m5, [workq + 352]
    movaps m6, [workq + 416]
    movaps m7, [workq + 480]
    TRANSPOSE8_TO_HIGH

    mov [rsp + 256], finishreq
    mov [rsp + 264], finishimq
    movaps m1, [rsp + 32]
    FINISH_MULTIPLY_ONE_TEMP m1, m9
    movaps m2, [rsp + 64]
    FINISH_MULTIPLY_ONE_TEMP m2, m10
    movaps m3, [rsp + 96]
    FINISH_MULTIPLY_ONE_TEMP m3, m11
    movaps m4, [rsp + 128]
    FINISH_MULTIPLY_ONE_TEMP m4, m12
    movaps m5, [rsp + 160]
    FINISH_MULTIPLY_ONE_TEMP m5, m13
    movaps m6, [rsp + 192]
    FINISH_MULTIPLY_ONE_TEMP m6, m14
    movaps m7, [rsp + 224]
    FINISH_MULTIPLY_ONE_TEMP m7, m15
    mov finishreq, [rsp + 256]
    mov finishimq, [rsp + 264]
    movaps m0, [rsp + 0]

    BUTTERFLY_NO_TEMP m0, m4
    BUTTERFLY_NO_TEMP m8, m12
    BUTTERFLY_NO_TEMP m1, m5
    BUTTERFLY_NO_TEMP m9, m13
    BUTTERFLY_NO_TEMP m2, m6
    BUTTERFLY_NO_TEMP m10, m14
    BUTTERFLY_NO_TEMP m3, m7
    BUTTERFLY_NO_TEMP m11, m15

    addps m5, m13
    addps m13, m13
    subps m13, m5
    mulps m5, [sqrt_half]
    mulps m13, [sqrt_half]
    xorps m6, [sign_all]
    addps m7, m15
    addps m15, m15
    subps m15, m7
    mulps m7, [neg_sqrt_half]
    mulps m15, [sqrt_half]

    FFT4_SPLIT_NO_TEMP m0, m1, m2, m3, m8, m9, m10, m11
    FFT4_SPLIT_NO_TEMP m4, m5, m14, m15, m12, m13, m6, m7

    movaps [rsp + 288], m7
    mov pq, outputq
    STORE_COMPLEX8 m0, m8, m7, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m4, m12, m0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m2, m3, m0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m14, m15, m0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m1, m9, m0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m5, m13, m0, pq
    lea pq, [pq + innerq*2]
    STORE_COMPLEX8 m11, m10, m0, pq
    lea pq, [pq + innerq*2]
    movaps m1, [rsp + 288]
    STORE_COMPLEX8 m1, m6, m0, pq

    add workq, 512
    add outputq, 64
    add finishreq, 32
    add finishimq, 32
    cmp finishreq, endq
    jb .loop
    RET

; Public monolithic dispatcher. Planning stays in scalar C; all execution
; data movement and floating-point work is in the assembly routines above.
global mangle(lane8_avx_execute)
align 64
mangle(lane8_avx_execute):
    test rdi, rdi
    jz .error
    test rsi, rsi
    jz .error
    cmp qword [rdi + PLAN_INNER_SIZE], 4
    jb .error

    push rbp
    push rbx
    push r12
    push r13
    sub rsp, 8
    mov rbp, rdi
    mov r12, rsi

    mov rdi, r12
    mov rsi, [rbp + PLAN_PERMUTATION]
    mov rdx, [rbp + PLAN_WORK]
    mov rcx, [rbp + PLAN_INNER_SIZE]
    test byte [rbp + PLAN_INNER_LEVELS], 1
    jnz .base8
    call mangle(lane8_avx_base4_fma3)
    mov ebx, 4
    jmp .stage_test
.base8:
    call mangle(lane8_avx_base8_fma3)
    mov ebx, 8

.stage_test:
    cmp rbx, [rbp + PLAN_INNER_SIZE]
    jae .finish
    mov r13, [rbp + PLAN_ROOT]
.stage_loop:
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, rbx
    mov rdx, [rbp + PLAN_INNER_SIZE]
    mov rcx, r13
    call mangle(lane8_avx_stage_fma3)
    lea rax, [rbx + rbx*2 - 3]
    shl rax, 6
    add r13, rax
    shl rbx, 2
    cmp rbx, [rbp + PLAN_INNER_SIZE]
    jb .stage_loop

.finish:
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, r12
    mov rdx, [rbp + PLAN_INNER_SIZE]
    mov rcx, [rbp + PLAN_FINISH_RE]
    mov r4q, [rbp + PLAN_FINISH_IM]
    ; The N=32 transform has only four inner frequencies.
    cmp qword [rbp + PLAN_INNER_SIZE], 4
    je .finish4
    call mangle(lane8_avx_finish8_fma3)
    jmp .finish_done
.finish4:
    call mangle(lane8_avx_finish_fma3)
.finish_done:
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    xor eax, eax
    vzeroupper
    ret
.error:
    mov eax, -1
    ret
