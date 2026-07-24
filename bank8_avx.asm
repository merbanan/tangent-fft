; Dual-bank radix-8 FFT for x86-64 System V.
;
; A work row contains two native AoS YMM banks:
;   bank A = four interleaved complex residues 0..3
;   bank B = four interleaved complex residues 4..7
; Inner radix-4 stages advance both banks together and load every twiddle
; vector once for both.  The finish transposes four frequencies in each bank,
; applies the seven outer factors, and performs a register-resident FFT8.

bits 64
default rel

%define PLAN_N               0
%define PLAN_INNER_SIZE      8
%define PLAN_INNER_LEVELS   16
%define PLAN_PERMUTATION    24
%define PLAN_TWIDDLE        32
%define PLAN_FINISH_FACTOR  40
%define PLAN_WORK           48

section .rodata align=32
align 32
bank8_sign_odd:
    dd 0, 0x80000000, 0, 0x80000000
    dd 0, 0x80000000, 0, 0x80000000
bank8_positive_diagonal:
    times 8 dd 0.7071067811865475244

section .text

; Inputs and outputs may alias.  The last four operands are scratch registers.
%macro RADIX4_MEM 12
    vaddps      %9,  %1, %3
    vsubps      %10, %1, %3
    vaddps      %11, %2, %4
    vsubps      %12, %2, %4
    vpermilps   %12, %12, 0xb1
    vxorps      %12, %12, [bank8_sign_odd]
    vaddps      %5, %9,  %11
    vaddps      %6, %10, %12
    vsubps      %7, %9,  %11
    vsubps      %8, %10, %12
%endmacro

%macro COMPLEX_MULTIPLY_FMA 4
    vpermilps       %2, %1, 0xb1
    vmulps          %2, %2, %4
    vfmaddsub132ps  %1, %2, %3
%endmacro

; The real and imaginary coefficient vectors are already in ymm8 and ymm9.
%macro COMPLEX_MULTIPLY_PAIR 4
    vpermilps       %3, %1, 0xb1
    vmulps          %3, %3, ymm9
    vfmaddsub132ps  %1, %3, ymm8
    vpermilps       %4, %2, 0xb1
    vmulps          %4, %4, ymm9
    vfmaddsub132ps  %2, %4, ymm8
%endmacro

; Natural-input split-radix FFT8 used by the outer residue transform.
; It computes an FFT4 over the even residues and two FFT2s over residues
; (1,5) and (3,7).  This uses 34 vector instructions instead of 46 for the
; direct radix-2 closure.  Result registers are documented below.
%macro FFT8_SPLIT_NATURAL 0
    RADIX4_MEM ymm0,ymm2,ymm4,ymm6, ymm0,ymm2,ymm4,ymm6, \
               ymm8,ymm9,ymm10,ymm11

    ; A0/A1 = x1 +/- x5; B0/B1 = x3 +/- x7.
    vaddps      ymm8, ymm1, ymm5
    vsubps      ymm5, ymm1, ymm5
    vaddps      ymm9, ymm3, ymm7
    vsubps      ymm7, ymm3, ymm7

    ; T1 = sqrt(1/2) * ((A1-B1) - i*(A1+B1))
    ; T3 = sqrt(1/2) * (-(A1-B1) - i*(A1+B1)).
    vaddps      ymm10, ymm5, ymm7
    vsubps      ymm11, ymm5, ymm7
    vpermilps   ymm10, ymm10, 0xb1
    vxorps      ymm10, ymm10, [bank8_sign_odd]
    vaddps      ymm12, ymm11, ymm10
    vsubps      ymm13, ymm10, ymm11
    vmulps      ymm12, ymm12, [bank8_positive_diagonal]
    vmulps      ymm13, ymm13, [bank8_positive_diagonal]
    vaddps      ymm1, ymm2, ymm12       ; X1
    vsubps      ymm3, ymm2, ymm12       ; X5
    vaddps      ymm5, ymm6, ymm13       ; X3
    vsubps      ymm7, ymm6, ymm13       ; X7

    ; X0/X4 and X2/X6 from A0 and B0.
    vaddps      ymm10, ymm8, ymm9
    vsubps      ymm11, ymm8, ymm9
    vpermilps   ymm11, ymm11, 0xb1
    vxorps      ymm11, ymm11, [bank8_sign_odd]
    vsubps      ymm8, ymm0, ymm10       ; X4
    vaddps      ymm0, ymm0, ymm10       ; X0
    vsubps      ymm9, ymm4, ymm11       ; X6
    vaddps      ymm4, ymm4, ymm11       ; X2
    ; X0..X7 are now ymm0,ymm1,ymm4,ymm5,ymm8,ymm3,ymm9,ymm7.
%endmacro

; Transpose four AoS rows in %1..%4 into four frequency vectors.
%macro TRANSPOSE4 4
    vunpcklpd   ymm8,  %1, %2
    vunpckhpd   ymm9,  %1, %2
    vunpcklpd   ymm10, %3, %4
    vunpckhpd   ymm11, %3, %4
    vinsertf128 %1, ymm8,  xmm10, 1
    vinsertf128 %2, ymm9,  xmm11, 1
    vperm2f128  %3, ymm8,  ymm10, 0x31
    vperm2f128  %4, ymm9,  ymm11, 0x31
%endmacro

; ---------------------------------------------------------------------------
; Paired mixed-radix leaves
; ---------------------------------------------------------------------------

; data, permutation, work, number of rows
align 64
bank8_base4:
.loop:
    mov     r8d,  [rsi]
    mov     r9d,  [rsi + 4]
    mov     r10d, [rsi + 8]
    mov     r11d, [rsi + 12]
    vmovups ymm0, [rdi + r8*8]
    vmovups ymm1, [rdi + r9*8]
    vmovups ymm2, [rdi + r10*8]
    vmovups ymm3, [rdi + r11*8]
    RADIX4_MEM ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
               ymm4,ymm5,ymm6,ymm7
    vmovaps [rdx],       ymm0
    vmovaps [rdx + 64],  ymm1
    vmovaps [rdx + 128], ymm2
    vmovaps [rdx + 192], ymm3

    vmovups ymm0, [rdi + r8*8 + 32]
    vmovups ymm1, [rdi + r9*8 + 32]
    vmovups ymm2, [rdi + r10*8 + 32]
    vmovups ymm3, [rdi + r11*8 + 32]
    RADIX4_MEM ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
               ymm4,ymm5,ymm6,ymm7
    vmovaps [rdx + 32],  ymm0
    vmovaps [rdx + 96],  ymm1
    vmovaps [rdx + 160], ymm2
    vmovaps [rdx + 224], ymm3
    add     rsi, 16
    add     rdx, 256
    sub     rcx, 4
    jnz .loop
    ret

; data, permutation, work, number of rows
align 64
bank8_base8:
    push    rbp
    push    rbx
    push    r12
.loop:
    mov     r8d,  [rsi]
    mov     r9d,  [rsi + 8]
    mov     r10d, [rsi + 16]
    mov     r11d, [rsi + 24]
    mov     eax,  [rsi + 4]
    mov     ebx,  [rsi + 12]
    mov     ebp,  [rsi + 20]
    mov     r12d, [rsi + 28]
    vmovups ymm0, [rdi + r8*8]
    vmovups ymm1, [rdi + r9*8]
    vmovups ymm2, [rdi + r10*8]
    vmovups ymm3, [rdi + r11*8]
    vmovups ymm4, [rdi + rax*8]
    vmovups ymm5, [rdi + rbx*8]
    vmovups ymm6, [rdi + rbp*8]
    vmovups ymm7, [rdi + r12*8]
    FFT8_SPLIT_NATURAL
    vmovaps [rdx],       ymm0
    vmovaps [rdx + 64],  ymm1
    vmovaps [rdx + 128], ymm4
    vmovaps [rdx + 192], ymm5
    vmovaps [rdx + 256], ymm8
    vmovaps [rdx + 320], ymm3
    vmovaps [rdx + 384], ymm9
    vmovaps [rdx + 448], ymm7

    vmovups ymm0, [rdi + r8*8 + 32]
    vmovups ymm1, [rdi + r9*8 + 32]
    vmovups ymm2, [rdi + r10*8 + 32]
    vmovups ymm3, [rdi + r11*8 + 32]
    vmovups ymm4, [rdi + rax*8 + 32]
    vmovups ymm5, [rdi + rbx*8 + 32]
    vmovups ymm6, [rdi + rbp*8 + 32]
    vmovups ymm7, [rdi + r12*8 + 32]
    FFT8_SPLIT_NATURAL
    vmovaps [rdx + 32],  ymm0
    vmovaps [rdx + 96],  ymm1
    vmovaps [rdx + 160], ymm4
    vmovaps [rdx + 224], ymm5
    vmovaps [rdx + 288], ymm8
    vmovaps [rdx + 352], ymm3
    vmovaps [rdx + 416], ymm9
    vmovaps [rdx + 480], ymm7
    add     rsi, 32
    add     rdx, 512
    sub     rcx, 8
    jnz .loop
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ---------------------------------------------------------------------------
; Paired upper radix-4 stages
; ---------------------------------------------------------------------------

; work, previous length in rows, total rows, shared twiddle stream
align 64
bank8_stage:
    shl     rsi, 6
    shl     rdx, 6
    lea     r11, [rdi + rdx]
.block_loop:
    lea     r9, [rdi + rsi]

    vmovaps ymm0, [rdi]
    vmovaps ymm1, [r9]
    vmovaps ymm2, [r9 + rsi]
    vmovaps ymm3, [r9 + rsi*2]
    vmovaps ymm4, [rdi + 32]
    vmovaps ymm5, [r9 + 32]
    vmovaps ymm6, [r9 + rsi + 32]
    vmovaps ymm7, [r9 + rsi*2 + 32]
    RADIX4_MEM ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
               ymm8,ymm9,ymm10,ymm12
    RADIX4_MEM ymm4,ymm5,ymm6,ymm7, ymm4,ymm5,ymm6,ymm7, \
               ymm8,ymm9,ymm10,ymm12
    vmovaps [rdi], ymm0
    vmovaps [r9],  ymm1
    vmovaps [r9 + rsi],   ymm2
    vmovaps [r9 + rsi*2], ymm3
    vmovaps [rdi + 32], ymm4
    vmovaps [r9 + 32],  ymm5
    vmovaps [r9 + rsi + 32],   ymm6
    vmovaps [r9 + rsi*2 + 32], ymm7

    add     rdi, 64
    add     r9,  64
    mov     r10, rcx
    mov     rax, rsi
    shr     rax, 6
    dec     rax
    jz      .block_done
.butterfly_loop:
    vmovaps ymm0, [rdi]
    vmovaps ymm1, [r9]
    vmovaps ymm2, [r9 + rsi]
    vmovaps ymm3, [r9 + rsi*2]
    vmovaps ymm4, [rdi + 32]
    vmovaps ymm5, [r9 + 32]
    vmovaps ymm6, [r9 + rsi + 32]
    vmovaps ymm7, [r9 + rsi*2 + 32]

    vmovaps ymm8, [r10]
    vmovaps ymm9, [r10 + 32]
    COMPLEX_MULTIPLY_PAIR ymm1, ymm5, ymm10, ymm11
    vmovaps ymm8, [r10 + 64]
    vmovaps ymm9, [r10 + 96]
    COMPLEX_MULTIPLY_PAIR ymm2, ymm6, ymm12, ymm13
    vmovaps ymm8, [r10 + 128]
    vmovaps ymm9, [r10 + 160]
    COMPLEX_MULTIPLY_PAIR ymm3, ymm7, ymm14, ymm15

    RADIX4_MEM ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
               ymm8,ymm9,ymm10,ymm12
    RADIX4_MEM ymm4,ymm5,ymm6,ymm7, ymm4,ymm5,ymm6,ymm7, \
               ymm8,ymm9,ymm10,ymm12
    vmovaps [rdi], ymm0
    vmovaps [r9],  ymm1
    vmovaps [r9 + rsi],   ymm2
    vmovaps [r9 + rsi*2], ymm3
    vmovaps [rdi + 32], ymm4
    vmovaps [r9 + 32],  ymm5
    vmovaps [r9 + rsi + 32],   ymm6
    vmovaps [r9 + rsi*2 + 32], ymm7
    add     rdi, 64
    add     r9,  64
    add     r10, 192
    dec     rax
    jnz .butterfly_loop
.block_done:
    lea     rdi, [r9 + rsi*2]
    cmp     rdi, r11
    jb .block_loop
    ret

; ---------------------------------------------------------------------------
; Four-frequency, eight-residue outer finish
; ---------------------------------------------------------------------------

; work, data, inner size, finish-factor stream
align 64
bank8_finish:
    mov     r8, rdx
    shl     r8, 3
    lea     r9,  [rsi + r8*2]
    lea     r10, [r9 + r8*2]
    lea     r11, [r10 + r8*2]
    mov     rax, rdx
    shl     rax, 6
    add     rax, rdi
.loop:
    vmovaps     ymm0, [rdi]
    vmovaps     ymm4, [rdi + 32]
    vmovaps     ymm1, [rdi + 64]
    vmovaps     ymm5, [rdi + 96]
    vmovaps     ymm2, [rdi + 128]
    vmovaps     ymm6, [rdi + 160]
    vmovaps     ymm3, [rdi + 192]
    vmovaps     ymm7, [rdi + 224]
    TRANSPOSE4 ymm0,ymm1,ymm2,ymm3
    TRANSPOSE4 ymm4,ymm5,ymm6,ymm7

    COMPLEX_MULTIPLY_FMA ymm1, ymm14, [rcx],       [rcx + 32]
    COMPLEX_MULTIPLY_FMA ymm2, ymm14, [rcx + 64],  [rcx + 96]
    COMPLEX_MULTIPLY_FMA ymm3, ymm14, [rcx + 128], [rcx + 160]
    COMPLEX_MULTIPLY_FMA ymm4, ymm14, [rcx + 192], [rcx + 224]
    COMPLEX_MULTIPLY_FMA ymm5, ymm14, [rcx + 256], [rcx + 288]
    COMPLEX_MULTIPLY_FMA ymm6, ymm14, [rcx + 320], [rcx + 352]
    COMPLEX_MULTIPLY_FMA ymm7, ymm14, [rcx + 384], [rcx + 416]
    FFT8_SPLIT_NATURAL

    vmovups [rsi],      ymm0
    vmovups [rsi + r8], ymm1
    vmovups [r9],       ymm4
    vmovups [r9 + r8],  ymm5
    vmovups [r10],      ymm8
    vmovups [r10 + r8], ymm3
    vmovups [r11],      ymm9
    vmovups [r11 + r8], ymm7
    add     rdi, 256
    add     rsi, 32
    add     r9,  32
    add     r10, 32
    add     r11, 32
    add     rcx, 448
    cmp     rdi, rax
    jb .loop
    ret

; ---------------------------------------------------------------------------
; Public execution
; ---------------------------------------------------------------------------

global bank8_avx_execute
align 64
bank8_avx_execute:
    test    rdi, rdi
    jz      .error
    test    rsi, rsi
    jz      .error
    cmp     qword [rdi + PLAN_N], 32
    jb      .error

    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8
    mov     rbp, rdi
    mov     r12, rsi
    mov     r13, [rbp + PLAN_INNER_SIZE]
    mov     r14, [rbp + PLAN_WORK]
    mov     r15, [rbp + PLAN_TWIDDLE]

    mov     rdi, r12
    mov     rsi, [rbp + PLAN_PERMUTATION]
    mov     rdx, r14
    mov     rcx, r13
    test    byte [rbp + PLAN_INNER_LEVELS], 1
    jnz     .base8
    call    bank8_base4
    mov     ebx, 4
    jmp     .stage_test
.base8:
    call    bank8_base8
    mov     ebx, 8
.stage_test:
    cmp     rbx, r13
    jae     .finish
.stage_loop:
    mov     rdi, r14
    mov     rsi, rbx
    mov     rdx, r13
    mov     rcx, r15
    call    bank8_stage
    lea     rax, [rbx - 1]
    imul    rax, rax, 192
    add     r15, rax
    shl     rbx, 2
    cmp     rbx, r13
    jb      .stage_loop
.finish:
    mov     rdi, r14
    mov     rsi, r12
    mov     rdx, r13
    mov     rcx, [rbp + PLAN_FINISH_FACTOR]
    call    bank8_finish
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    vzeroupper
    xor     eax, eax
    ret
.error:
    mov     eax, -1
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
