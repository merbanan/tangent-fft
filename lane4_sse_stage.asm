; Lane-factorized radix-4 stage for x86-64.
;
; This uses FFmpeg's x86inc calling-convention and register macros.  Four
; independent complex FFTs are stored as separate real/imaginary XMM vectors.
; Twiddle values are replicated in the plan so the hot loop can consume them
; with aligned loads and no scalar broadcast shuffles.

%define private_prefix lane4
%include "libavutil/x86/x86inc.asm"

SECTION_RODATA 16

align 16
sqrt_half:
    times 4 dd 0.7071067811865475244
neg_sqrt_half:
    times 4 dd -0.7071067811865475244
sign_all:
    times 4 dd 0x80000000

SECTION .text

; lane4_portable_plan offsets. Planning and allocation remain in C; execution
; enters here directly so every SSE-flavour public transform is assembly.
%define PLAN_N                0
%define PLAN_INNER_SIZE       8
%define PLAN_INNER_LEVELS    16
%define PLAN_MIXED_PERM      32
%define PLAN_REPLICATED_ROOT 48
%define PLAN_FINISH_RE       56
%define PLAN_FINISH_IM       64
%define PLAN_WORK            72

; Later SSE revisions do not add an operation needed by this float kernel.
; Keep distinct public CPUID/API entries while sharing the exact SSE1 body.
global mangle(lane4_sse_execute)
global mangle(lane4_sse2_execute)
global mangle(lane4_sse3_execute)
global mangle(lane4_ssse3_execute)
global mangle(lane4_sse41_execute)
global mangle(lane4_sse42_execute)
align 64
mangle(lane4_sse_execute):
mangle(lane4_sse2_execute):
mangle(lane4_sse3_execute):
mangle(lane4_ssse3_execute):
mangle(lane4_sse41_execute):
mangle(lane4_sse42_execute):
    test rdi, rdi
    jz .execute_error
    test rsi, rsi
    jz .execute_error
    cmp qword [rdi + PLAN_REPLICATED_ROOT], 0
    je .execute_error
    cmp qword [rdi + PLAN_N], 16
    je mangle(lane4_sse_fused16)
    cmp qword [rdi + PLAN_N], 32
    je mangle(lane4_sse_fused32)
    cmp qword [rdi + PLAN_N], 64
    je mangle(lane4_sse_fused64)

    push rbp
    push rbx
    push r12
    push r13
    sub rsp, 8
    mov rbp, rdi
    mov r12, rsi

    mov rdi, r12
    mov rsi, [rbp + PLAN_MIXED_PERM]
    mov rdx, [rbp + PLAN_WORK]
    mov rcx, [rbp + PLAN_INNER_SIZE]
    test byte [rbp + PLAN_INNER_LEVELS], 1
    jnz .execute_base8
    call mangle(lane4_sse_base4_sse)
    mov ebx, 4
    jmp .execute_stage_test

.execute_base8:
    call mangle(lane4_sse_base8_sse)
    mov ebx, 8

.execute_stage_test:
    cmp rbx, [rbp + PLAN_INNER_SIZE]
    jae .execute_finish
    mov r13, [rbp + PLAN_REPLICATED_ROOT]

.execute_stage_loop:
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, rbx
    mov rdx, [rbp + PLAN_INNER_SIZE]
    mov rcx, r13
    call mangle(lane4_sse_stage_sse)
    lea rax, [rbx + rbx*2 - 3]
    shl rax, 5
    add r13, rax
    shl rbx, 2
    cmp rbx, [rbp + PLAN_INNER_SIZE]
    jb .execute_stage_loop

.execute_finish:
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, r12
    mov rdx, [rbp + PLAN_INNER_SIZE]
    mov rcx, [rbp + PLAN_FINISH_RE]
    mov r4q, [rbp + PLAN_FINISH_IM]
    call mangle(lane4_sse_finish_sse)
    add rsp, 8
    pop r13
    pop r12
    pop rbx
    pop rbp
    xor eax, eax
    ret

.execute_error:
    mov eax, -1
    ret

%macro LOAD_INPUT4 3
    movups %1, [%3]
    movups m15, [%3 + 16]
    movaps %2, %1
    shufps %1, m15, 0x88
    shufps %2, m15, 0xdd
%endmacro

; Two-register AoS-to-SoA load for the final row when all other XMM
; registers are live.
%macro LOAD_INPUT4_LAST 3
    movups %1, [%3]
    movups %2, [%3 + 16]
    shufps %2, %1, 0xdd
    shufps %2, %2, 0x4e
    shufps %1, [%3 + 16], 0x88
%endmacro

; In-place sum/difference without a temporary register.
%macro BUTTERFLY_NO_TEMP 2
    addps %1, %2
    addps %2, %2
    subps %2, %1
    xorps %2, [sign_all]
%endmacro

; Complex vector multiply (%1+i%2) by the replicated coefficient at %3.
; Memory operands let x86 fuse coefficient loads with the multiplies.
%macro COMPLEX_MULTIPLY 3
    movaps m14, %1
    movaps m15, %2
    mulps  %1, [%3]
    mulps  %2, [%3]
    mulps  m14, [%3 + 16]
    mulps  m15, [%3 + 16]
    subps  %1, m15
    addps  %2, m14
%endmacro

%macro RADIX4_STORE_CONTIGUOUS 0
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

    movaps [aq + 0], m12
    movaps [aq + 16], m13
    movaps [aq + 32], m10
    movaps [aq + 48], m11
    movaps [aq + 64], m8
    movaps [aq + 80], m9
    movaps [aq + 96], m0
    movaps [aq + 112], m1
%endmacro

; Same arithmetic with four independent row pointers.  Upper stages advance
; these pointers linearly, avoiding repeated base+index reconstruction.
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
    movaps [%1 + 16], m13
    movaps [%2], m10
    movaps [%2 + 16], m11
    movaps [%3], m8
    movaps [%3 + 16], m9
    movaps [%4], m0
    movaps [%4 + 16], m1
%endmacro

; Keep the four row bases fixed and advance one shared byte offset.  This
; removes three integer adds from each upper-stage butterfly.
%macro LOAD_POINTER_ROWS 0
    movaps m0, [workq + rowendq]
    movaps m1, [workq + rowendq + 16]
    movaps m2, [p1q + rowendq]
    movaps m3, [p1q + rowendq + 16]
    movaps m4, [p2q + rowendq]
    movaps m5, [p2q + rowendq + 16]
    movaps m6, [p3q + rowendq]
    movaps m7, [p3q + rowendq + 16]
%endmacro

%macro ADVANCE_POINTER_ROWS 0
    add rowendq, 32
%endmacro

%macro STAGE_GENERAL_BUTTERFLY 0
    LOAD_POINTER_ROWS
    COMPLEX_MULTIPLY m2, m3, rootsq
    COMPLEX_MULTIPLY m4, m5, rootsq + 32
    COMPLEX_MULTIPLY m6, m7, rootsq + 64
    RADIX4_STORE_POINTERS \
        workq + rowendq, p1q + rowendq, \
        p2q + rowendq, p3q + rowendq
    add rootsq, 96
    ADVANCE_POINTER_ROWS
%endmacro

; data, mixed-radix permutation, work, row count
INIT_XMM sse
cglobal sse_base4, 4, 7, 16, data, permutation, work, count, index, input, a
    xor indexq, indexq
.base4_loop:
    mov inputd, [permutationq + indexq*4]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m0, m1, inputq
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m2, m3, inputq
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m4, m5, inputq
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m6, m7, inputq
    mov aq, indexq
    shl aq, 5
    add aq, workq
    RADIX4_STORE_CONTIGUOUS
    add indexq, 4
    cmp indexq, countq
    jb .base4_loop
    RET

; Register-resident vector FFT8.  The no-temporary butterflies let all eight
; split-complex rows occupy the sixteen architectural XMM registers.
; data, mixed-radix permutation, work, row count
INIT_XMM sse
cglobal sse_base8, 4, 7, 16, data, permutation, work, count, index, input, a
    xor indexq, indexq
.base8_loop:
    mov inputd, [permutationq + indexq*4 + 0]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m0, m1, inputq
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m2, m3, inputq
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m4, m5, inputq
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m6, m7, inputq
    mov inputd, [permutationq + indexq*4 + 16]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m8, m9, inputq
    mov inputd, [permutationq + indexq*4 + 20]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m10, m11, inputq
    mov inputd, [permutationq + indexq*4 + 24]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4 m12, m13, inputq
    mov inputd, [permutationq + indexq*4 + 28]
    lea inputq, [dataq + inputq*8]
    LOAD_INPUT4_LAST m14, m15, inputq

    BUTTERFLY_NO_TEMP m0, m2
    BUTTERFLY_NO_TEMP m1, m3
    BUTTERFLY_NO_TEMP m4, m6
    BUTTERFLY_NO_TEMP m5, m7
    BUTTERFLY_NO_TEMP m8, m10
    BUTTERFLY_NO_TEMP m9, m11
    BUTTERFLY_NO_TEMP m12, m14
    BUTTERFLY_NO_TEMP m13, m15

    ; Odd-half twiddles W8, -i, W8^3 without scratch registers.
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

    ; Even-frequency FFT4: logical rows 0, 2, 4, 6.
    BUTTERFLY_NO_TEMP m0, m8
    BUTTERFLY_NO_TEMP m1, m9
    BUTTERFLY_NO_TEMP m4, m12
    BUTTERFLY_NO_TEMP m5, m13
    BUTTERFLY_NO_TEMP m0, m4
    BUTTERFLY_NO_TEMP m1, m5
    BUTTERFLY_NO_TEMP m8, m13
    BUTTERFLY_NO_TEMP m9, m12

    ; Odd-frequency FFT4.  W8^2 and W8^3 left their logical components
    ; swapped, which is consumed directly by this register assignment.
    BUTTERFLY_NO_TEMP m2, m11
    BUTTERFLY_NO_TEMP m3, m10
    BUTTERFLY_NO_TEMP m6, m15
    BUTTERFLY_NO_TEMP m7, m14
    BUTTERFLY_NO_TEMP m2, m6
    BUTTERFLY_NO_TEMP m3, m7
    BUTTERFLY_NO_TEMP m11, m14
    BUTTERFLY_NO_TEMP m10, m15

    mov aq, indexq
    shl aq, 5
    add aq, workq
    movaps [aq + 0], m0
    movaps [aq + 16], m1
    movaps [aq + 32], m2
    movaps [aq + 48], m3
    movaps [aq + 64], m8
    movaps [aq + 80], m12
    movaps [aq + 96], m11
    movaps [aq + 112], m15
    movaps [aq + 128], m4
    movaps [aq + 144], m5
    movaps [aq + 160], m6
    movaps [aq + 176], m7
    movaps [aq + 192], m13
    movaps [aq + 208], m9
    movaps [aq + 224], m14
    movaps [aq + 240], m10

    add indexq, 8
    cmp indexq, countq
    jb .base8_loop
    RET

INIT_XMM sse
cglobal sse_stage, 4, 12, 16, work, previous, inner, root, p1, p2, p3, rowend, next, innerend, roots, special
    shl previousq, 5
    shl innerq, 5
    lea innerendq, [workq + innerq]

.block_loop:
    lea p1q, [workq + previousq]
    lea p2q, [p1q + previousq]
    lea p3q, [p2q + previousq]
    lea nextq, [p3q + previousq]
    xor rowendq, rowendq
    mov specialq, previousq
    shr specialq, 1
    LOAD_POINTER_ROWS
    RADIX4_STORE_POINTERS \
        workq + rowendq, p1q + rowendq, \
        p2q + rowendq, p3q + rowendq
    ADVANCE_POINTER_ROWS
    mov rootsq, rootq

.before_special:
    STAGE_GENERAL_BUTTERFLY
    cmp rowendq, specialq
    jb .before_special

    ; k = previous/2 has factors W8, -i, and W8^3.  FFmpeg handles
    ; equivalent exact roots with sign folds and reduced multiplications.
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
    RADIX4_STORE_POINTERS \
        workq + rowendq, p1q + rowendq, \
        p2q + rowendq, p3q + rowendq
    add rootsq, 96
    ADVANCE_POINTER_ROWS

.after_special:
    cmp rowendq, previousq
    jae .next_block
    STAGE_GENERAL_BUTTERFLY
    jmp .after_special

.next_block:
    mov workq, nextq
    cmp workq, innerendq
    jb .block_loop
    RET

; Transpose four vectors held in %1-%4.
; m14 and m15 are scratch.
%macro TRANSPOSE4 4
    movaps m14, %1
    movaps m15, %3
    unpcklps %1, %2
    unpckhps m14, %2
    unpcklps %3, %4
    unpckhps m15, %4
    movaps %2, %1
    movlhps %1, %3
    movhlps %3, %2
    movaps %2, m14
    movlhps %2, m15
    movhlps m15, m14
    movaps %4, m15
%endmacro

; Multiply the split-complex vector %1:%2 by the vector factors at %3:%4.
%macro COMPLEX_MULTIPLY_SPLIT 4
    movaps m14, %1
    movaps m15, %2
    mulps  %1, %3
    mulps  %2, %3
    mulps  m14, %4
    mulps  m15, %4
    subps  %1, m15
    addps  %2, m14
%endmacro

; Interleave a real/imaginary pair and store four complex values.
%macro STORE_COMPLEX4 3
    movaps m14, %1
    unpcklps %1, %2
    unpckhps m14, %2
    movups [%3], %1
    movups [%3 + 16], m14
%endmacro

; work, output, inner_size, finish_re, finish_im
INIT_XMM sse
cglobal sse_finish, 5, 9, 16, work, output, inner, finishre, finishim, out1, out2, out3, end
    lea out1q, [outputq + innerq*8]
    lea out2q, [out1q + innerq*8]
    lea out3q, [out2q + innerq*8]
    shl innerq, 2
    lea endq, [finishreq + innerq]

.finish_loop:
    movaps m0, [workq + 0]
    movaps m1, [workq + 32]
    movaps m2, [workq + 64]
    movaps m3, [workq + 96]
    movaps m4, [workq + 16]
    movaps m5, [workq + 48]
    movaps m6, [workq + 80]
    movaps m7, [workq + 112]
    TRANSPOSE4 m0, m1, m2, m3
    TRANSPOSE4 m4, m5, m6, m7

    COMPLEX_MULTIPLY_SPLIT \
        m2, m6, [finishreq], [finishimq]
    COMPLEX_MULTIPLY_SPLIT \
        m1, m5, [finishreq + innerq], [finishimq + innerq]
    COMPLEX_MULTIPLY_SPLIT \
        m3, m7, [finishreq + innerq*2], [finishimq + innerq*2]

    movaps m8, m0
    movaps m9, m4
    addps  m8, m1
    addps  m9, m5
    subps  m0, m1
    subps  m4, m5

    movaps m10, m2
    movaps m11, m6
    addps  m10, m3
    addps  m11, m7
    subps  m2, m3
    subps  m6, m7

    movaps m12, m8
    movaps m13, m9
    addps  m12, m10
    addps  m13, m11
    subps  m8, m10
    subps  m9, m11

    movaps m10, m0
    movaps m11, m4
    addps  m10, m6
    subps  m11, m2
    subps  m0, m6
    addps  m4, m2

    STORE_COMPLEX4 m12, m13, outputq
    STORE_COMPLEX4 m10, m11, out1q
    STORE_COMPLEX4 m8, m9, out2q
    STORE_COMPLEX4 m0, m4, out3q

    add workq, 128
    add outputq, 32
    add out1q, 32
    add out2q, 32
    add out3q, 32
    add finishreq, 16
    add finishimq, 16
    cmp finishreq, endq
    jb .finish_loop
    RET

; ---------------------------------------------------------------------------
; Fixed-size fused leaves shared by SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2
; ---------------------------------------------------------------------------

; Fixed-register SoA radix-4. Inputs and outputs are m0:m1 ... m6:m7;
; m8-m15 are temporary.
%macro RADIX4_SOA_FIXED 0
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

    movaps m2, m10
    movaps m3, m11
    movaps m4, m8
    movaps m5, m9
    movaps m6, m0
    movaps m7, m1
    movaps m0, m12
    movaps m1, m13
%endmacro

global mangle(lane4_sse_fused16)
align 64
mangle(lane4_sse_fused16):
    mov rdx, [rdi + PLAN_FINISH_RE]
    mov rcx, [rdi + PLAN_FINISH_IM]
    mov rdi, rsi

    LOAD_INPUT4 m0, m1, rdi
    LOAD_INPUT4 m2, m3, rdi + 32
    LOAD_INPUT4 m4, m5, rdi + 64
    LOAD_INPUT4 m6, m7, rdi + 96
    RADIX4_SOA_FIXED

    COMPLEX_MULTIPLY_SPLIT m2, m3, [rdx],      [rcx]
    COMPLEX_MULTIPLY_SPLIT m4, m5, [rdx + 16], [rcx + 16]
    COMPLEX_MULTIPLY_SPLIT m6, m7, [rdx + 32], [rcx + 32]

    TRANSPOSE4 m0, m2, m4, m6
    TRANSPOSE4 m1, m3, m5, m7
    ; TRANSPOSE4's historical output order is 0,2,1,3 because the generic
    ; finish consumes that order directly.  The fixed radix macro wants
    ; natural 0,1,2,3 rows.
    movaps m14, m2
    movaps m2, m4
    movaps m4, m14
    movaps m14, m3
    movaps m3, m5
    movaps m5, m14
    RADIX4_SOA_FIXED

    STORE_COMPLEX4 m0, m1, rdi
    STORE_COMPLEX4 m2, m3, rdi + 32
    STORE_COMPLEX4 m4, m5, rdi + 64
    STORE_COMPLEX4 m6, m7, rdi + 96
    xor eax, eax
    ret

; N=32 has one FFT8 SoA leaf and no upper stage.  Keeping this fixed path
; removes generic stage tests and the reusable-executor frame.
global mangle(lane4_sse_fused32)
align 64
mangle(lane4_sse_fused32):
    push rbp
    push r12
    sub rsp, 8
    mov rbp, rdi
    mov r12, rsi
    mov rdi, r12
    mov rsi, [rbp + PLAN_MIXED_PERM]
    mov rdx, [rbp + PLAN_WORK]
    mov rcx, 8
    call mangle(lane4_sse_base8_sse)
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, r12
    mov rdx, 8
    mov rcx, [rbp + PLAN_FINISH_RE]
    mov r4q, [rbp + PLAN_FINISH_IM]
    call mangle(lane4_sse_finish_sse)
    add rsp, 8
    pop r12
    pop rbp
    xor eax, eax
    ret

; N=64 is exactly one FFT4 leaf layer, one radix-4 stage, and finish.
global mangle(lane4_sse_fused64)
align 64
mangle(lane4_sse_fused64):
    push rbp
    push r12
    sub rsp, 8
    mov rbp, rdi
    mov r12, rsi
    mov rdi, r12
    mov rsi, [rbp + PLAN_MIXED_PERM]
    mov rdx, [rbp + PLAN_WORK]
    mov rcx, 16
    call mangle(lane4_sse_base4_sse)
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, 4
    mov rdx, 16
    mov rcx, [rbp + PLAN_REPLICATED_ROOT]
    call mangle(lane4_sse_stage_sse)
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, r12
    mov rdx, 16
    mov rcx, [rbp + PLAN_FINISH_RE]
    mov r4q, [rbp + PLAN_FINISH_IM]
    call mangle(lane4_sse_finish_sse)
    add rsp, 8
    pop r12
    pop rbp
    xor eax, eax
    ret
