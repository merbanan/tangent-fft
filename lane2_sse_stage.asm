; Two-lane SSE FFT for x86-64.
;
; Each XMM row contains two complete interleaved complex values:
;
;   [F0.re, F0.im, F1.re, F1.im]
;
; The inner transform evaluates the two even/odd residue FFTs together.
; The finish transposes two adjacent rows, applies the odd-residue twiddles,
; and evaluates two final FFT2s in parallel.

%define private_prefix lane2
%include "libavutil/x86/x86inc.asm"

SECTION_RODATA 16

align 16
sign_odd:
    dd 0, 0x80000000, 0, 0x80000000
align 16
w8:
    times 4 dd 0.7071067811865475244
    dd 0.7071067811865475244, -0.7071067811865475244
    dd 0.7071067811865475244, -0.7071067811865475244
align 16
w8_3:
    times 4 dd -0.7071067811865475244
    dd 0.7071067811865475244, -0.7071067811865475244
    dd 0.7071067811865475244, -0.7071067811865475244

SECTION .text

%define PLAN_INNER_SIZE       8
%define PLAN_INNER_LEVELS    16
%define PLAN_MIXED_PERM      24
%define PLAN_REPLICATED_ROOT 32
%define PLAN_FINISH_ROOT     40
%define PLAN_WORK            48

global mangle(lane2_sse_execute)
align 64
mangle(lane2_sse_execute):
    test rdi, rdi
    jz .execute_error
    test rsi, rsi
    jz .execute_error
    cmp qword [rdi + PLAN_REPLICATED_ROOT], 0
    je .execute_error

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
    jz .execute_base4
    call mangle(lane2_sse_base8_sse)
    mov ebx, 8
    jmp .execute_stage_test

.execute_base4:
    call mangle(lane2_sse_base4_sse)
    mov ebx, 4

.execute_stage_test:
    cmp rbx, [rbp + PLAN_INNER_SIZE]
    jae .execute_finish
    mov r13, [rbp + PLAN_REPLICATED_ROOT]

.execute_stage_loop:
    mov rdi, [rbp + PLAN_WORK]
    mov rsi, rbx
    mov rdx, [rbp + PLAN_INNER_SIZE]
    mov rcx, r13
    call mangle(lane2_sse_stage_sse)
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
    mov rcx, [rbp + PLAN_FINISH_ROOT]
    call mangle(lane2_sse_finish_sse)
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

; Multiply two interleaved complex values in %1 by the factors at %2.
; The second vector stores [-wi0,+wi0,-wi1,+wi1], allowing baseline SSE
; to use a regular add instead of addsubps.
%macro COMPLEX_MULTIPLY 3
    movaps %3, %1
    shufps %3, %3, 0xb1
    mulps %1, [%2]
    mulps %3, [%2 + 16]
    addps %1, %3
%endmacro

; Rotate all three nonzero radix-4 legs together. m4-m6 carry independent
; swapped values until the multiplies complete, then become the radix-4
; output scratch registers. This breaks the loop-carried m8 dependency of
; three consecutive COMPLEX_MULTIPLY expansions without adding spills.
%macro COMPLEX_MULTIPLY_THREE 1
    movaps m4, m1
    movaps m5, m2
    movaps m6, m3
    shufps m4, m4, 0xb1
    shufps m5, m5, 0xb1
    shufps m6, m6, 0xb1
    mulps m1, [%1 + 0]
    mulps m2, [%1 + 32]
    mulps m3, [%1 + 64]
    mulps m4, [%1 + 16]
    mulps m5, [%1 + 48]
    mulps m6, [%1 + 80]
    addps m1, m4
    addps m2, m5
    addps m3, m6
%endmacro

; Forward FFT4 over four vectors, followed by pointer stores.
; Inputs are m0=a, m1=b, m2=c, m3=d.
%macro RADIX4_STORE_POINTERS 4
    movaps m4, m0
    addps m4, m2
    subps m0, m2
    movaps m5, m1
    addps m5, m3
    subps m1, m3

    movaps m6, m4
    addps m6, m5
    subps m4, m5

    shufps m1, m1, 0xb1
    xorps m1, [sign_odd]
    movaps m7, m0
    addps m7, m1
    subps m0, m1

    movaps [%1], m6
    movaps [%2], m7
    movaps [%3], m4
    movaps [%4], m0
%endmacro

%macro RADIX4_STORE_CONTIGUOUS 1
    movaps m4, m0
    addps m4, m2
    subps m0, m2
    movaps m5, m1
    addps m5, m3
    subps m1, m3

    movaps m6, m4
    addps m6, m5
    subps m4, m5

    shufps m1, m1, 0xb1
    xorps m1, [sign_odd]
    movaps m7, m0
    addps m7, m1
    subps m0, m1

    movaps [%1 + 0], m6
    movaps [%1 + 16], m7
    movaps [%1 + 32], m4
    movaps [%1 + 48], m0
%endmacro

; Destructive FFT4 with results stored at every second row. The four input
; registers may be named arbitrarily; m12-m15 are scratch.
%macro RADIX4_STORE_STRIDE2 6
    movaps m12, %1
    addps m12, %3
    subps %1, %3
    movaps m13, %2
    addps m13, %4
    subps %2, %4

    movaps m14, m12
    addps m14, m13
    subps m12, m13

    shufps %2, %2, 0xb1
    xorps %2, [sign_odd]
    movaps m15, %1
    addps m15, %2
    subps %1, %2

    movaps [%5 + %6 + 0], m14
    movaps [%5 + %6 + 32], m15
    movaps [%5 + %6 + 64], m12
    movaps [%5 + %6 + 96], %1
%endmacro

; Register-resident FFT8 leaf. Mixed-radix permutation presents four
; distance-four pairs. Their sums feed the even FFT4 while their differences,
; after W8/-i/W8^3 rotations, feed the odd FFT4.
; data, mixed-radix permutation, work, row count
align 64
INIT_XMM sse
cglobal sse_base8, 4, 7, 16, data, permutation, work, count, index, input, a
    xor indexq, indexq
.base8_loop:
    mov inputd, [permutationq + indexq*4 + 0]
    lea inputq, [dataq + inputq*8]
    movups m0, [inputq]
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    movups m1, [inputq]
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    movups m2, [inputq]
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    movups m3, [inputq]
    mov inputd, [permutationq + indexq*4 + 16]
    lea inputq, [dataq + inputq*8]
    movups m4, [inputq]
    mov inputd, [permutationq + indexq*4 + 20]
    lea inputq, [dataq + inputq*8]
    movups m5, [inputq]
    mov inputd, [permutationq + indexq*4 + 24]
    lea inputq, [dataq + inputq*8]
    movups m6, [inputq]
    mov inputd, [permutationq + indexq*4 + 28]
    lea inputq, [dataq + inputq*8]
    movups m7, [inputq]

    movaps m8, m0
    addps m0, m1
    subps m8, m1
    movaps m9, m2
    addps m2, m3
    subps m9, m3
    movaps m10, m4
    addps m4, m5
    subps m10, m5
    movaps m11, m6
    addps m6, m7
    subps m11, m7

    COMPLEX_MULTIPLY m9, w8, m12
    shufps m10, m10, 0xb1
    xorps m10, [sign_odd]
    COMPLEX_MULTIPLY m11, w8_3, m12

    mov aq, indexq
    shl aq, 4
    add aq, workq
    RADIX4_STORE_STRIDE2 m0, m2, m4, m6, aq, 0
    RADIX4_STORE_STRIDE2 m8, m9, m10, m11, aq, 16

    add indexq, 8
    cmp indexq, countq
    jb .base8_loop
    RET

; data, mixed-radix permutation, work, row count
align 64
INIT_XMM sse
cglobal sse_base4, 4, 7, 8, data, permutation, work, count, index, input, a
    xor indexq, indexq
.base4_loop:
    mov inputd, [permutationq + indexq*4]
    lea inputq, [dataq + inputq*8]
    movups m0, [inputq]
    mov inputd, [permutationq + indexq*4 + 4]
    lea inputq, [dataq + inputq*8]
    movups m1, [inputq]
    mov inputd, [permutationq + indexq*4 + 8]
    lea inputq, [dataq + inputq*8]
    movups m2, [inputq]
    mov inputd, [permutationq + indexq*4 + 12]
    lea inputq, [dataq + inputq*8]
    movups m3, [inputq]
    mov aq, indexq
    shl aq, 4
    add aq, workq
    RADIX4_STORE_CONTIGUOUS aq
    add indexq, 4
    cmp indexq, countq
    jb .base4_loop
    RET

%macro LOAD_POINTER_ROWS 0
    movaps m0, [workq]
    movaps m1, [p1q]
    movaps m2, [p2q]
    movaps m3, [p3q]
%endmacro

%macro ADVANCE_POINTER_ROWS 0
    add workq, 16
    add p1q, 16
    add p2q, 16
    add p3q, 16
%endmacro

%macro STAGE_GENERAL_BUTTERFLY 0
    LOAD_POINTER_ROWS
    COMPLEX_MULTIPLY_THREE rootsq
    RADIX4_STORE_POINTERS workq, p1q, p2q, p3q
    add rootsq, 96
    ADVANCE_POINTER_ROWS
%endmacro

; work, previous row count, total inner row count, stage roots
align 64
INIT_XMM sse
cglobal sse_stage, 4, 11, 9, work, previous, inner, root, p1, p2, p3, next, rowend, roots, special
    shl previousq, 4
    shl innerq, 4
    add innerq, workq

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

    ; There are an odd number of frequencies on each side of k=previous/2.
    ; Consume one and then unroll the remaining even count by two.
    STAGE_GENERAL_BUTTERFLY

.before_special:
    cmp workq, specialq
    jae .at_special
    STAGE_GENERAL_BUTTERFLY
    STAGE_GENERAL_BUTTERFLY
    jmp .before_special

.at_special:
    ; W8, -i, and W8^3 are exact roots. Avoid six coefficient-vector
    ; memory operands and reduce the middle multiply to a swap/sign fold.
    LOAD_POINTER_ROWS
    COMPLEX_MULTIPLY m1, w8, m8
    shufps m2, m2, 0xb1
    xorps m2, [sign_odd]
    COMPLEX_MULTIPLY m3, w8_3, m8
    RADIX4_STORE_POINTERS workq, p1q, p2q, p3q
    add rootsq, 96
    ADVANCE_POINTER_ROWS

    STAGE_GENERAL_BUTTERFLY

.after_special:
    cmp workq, rowendq
    jae .next_block
    STAGE_GENERAL_BUTTERFLY
    STAGE_GENERAL_BUTTERFLY
    jmp .after_special

.next_block:
    mov workq, nextq
    cmp workq, innerq
    jb .block_loop
    RET

; work, output, inner row count, packed finish roots
align 64
INIT_XMM sse
cglobal sse_finish, 4, 6, 6, work, output, inner, finish, out1, end
    lea out1q, [outputq + innerq*8]
    shl innerq, 4
    lea endq, [workq + innerq]

.finish_loop:
    movaps m0, [workq]
    movaps m1, [workq + 16]

    ; Transpose two rows at the granularity of 64-bit complex pairs.
    movaps m2, m0
    movlhps m2, m1
    movaps m3, m1
    movhlps m3, m0

    COMPLEX_MULTIPLY m3, finishq, m4
    movaps m5, m2
    addps m2, m3
    subps m5, m3
    movups [outputq], m2
    movups [out1q], m5

    add workq, 32
    add outputq, 16
    add out1q, 16
    add finishq, 32
    cmp workq, endq
    jb .finish_loop
    RET
