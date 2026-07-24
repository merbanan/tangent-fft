; Handwritten four-lane FFT for x86-64 System V.
;
; Each YMM register holds four interleaved complex floats.  Scalar C performs
; planning and builds aligned replicated coefficient streams; every execution
; load, shuffle, butterfly, complex product, and store is implemented here.
; The AVX and AVX2 API variants share an AVX body because this dataflow needs
; no AVX2-only primitive.  The FMA variants use three-instruction complex
; products; the plain AVX variants use vmulps/vaddsubps.

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
lane4_sign_odd:
    dd 0, 0x80000000, 0, 0x80000000
    dd 0, 0x80000000, 0, 0x80000000
lane4_sign_all:
    times 8 dd 0x80000000
lane4_positive_diagonal:
    times 8 dd 0.7071067811865475244
lane4_negative_diagonal:
    times 8 dd -0.7071067811865475244

section .text

%macro MULTIPLY_MINUS_I 2
    vpermilps %1, %1, 0xb1
    vxorps    %1, %1, %2
%endmacro

; Four full-width complex butterflies.  Inputs may alias outputs, provided the
; four scratch registers are distinct from both groups.
%macro RADIX4 13
    vaddps      %10, %1, %3
    vsubps      %11, %1, %3
    vaddps      %12, %2, %4
    vsubps      %13, %2, %4
    MULTIPLY_MINUS_I %13, %9
    vaddps      %5, %10, %12
    vaddps      %6, %11, %13
    vsubps      %7, %10, %12
    vsubps      %8, %11, %13
%endmacro

%macro COMPLEX_MULTIPLY_AVX 4
    vpermilps   %2, %1, 0xb1
    vmulps      %1, %1, %3
    vmulps      %2, %2, %4
    vaddsubps   %1, %1, %2
%endmacro

%macro COMPLEX_MULTIPLY_FMA 4
    vpermilps       %2, %1, 0xb1
    vmulps          %2, %2, %4
    vfmaddsub132ps  %1, %2, %3
%endmacro

; In-place sum/difference without an extra vector register.
%macro BUTTERFLY_NO_TEMP 2
    vaddps %1, %1, %2
    vaddps %2, %2, %2
    vsubps %2, %2, %1
    vxorps %2, %2, [lane4_sign_all]
%endmacro

; ---------------------------------------------------------------------------
; Register-resident N=16 and N=32 transforms
; ---------------------------------------------------------------------------

; Transpose four rows, apply finish factors, perform the outer FFT4, and store.
; rdi=data, rdx=finish factors, ymm8=-i mask.  Used only by the N=32 leaf.
%macro FINISH_SMALL 9
    vunpcklpd   ymm9,  %1, %2
    vunpckhpd   ymm10, %1, %2
    vunpcklpd   ymm11, %3, %4
    vunpckhpd   ymm12, %3, %4
    vinsertf128 ymm13, ymm9,  xmm11, 1
    vperm2f128  ymm9,  ymm9,  ymm11, 0x31
    vinsertf128 ymm14, ymm10, xmm12, 1
    vperm2f128  ymm10, ymm10, ymm12, 0x31
    COMPLEX_MULTIPLY_AVX ymm14, ymm15, [rdx + %5],       [rdx + %5 + 32]
    COMPLEX_MULTIPLY_AVX ymm9,  ymm15, [rdx + %5 + 64],  [rdx + %5 + 96]
    COMPLEX_MULTIPLY_AVX ymm10, ymm15, [rdx + %5 + 128], [rdx + %5 + 160]
    vaddps      ymm11, ymm13, ymm9
    vsubps      ymm12, ymm13, ymm9
    vaddps      ymm13, ymm14, ymm10
    vsubps      ymm14, ymm14, ymm10
    vpermilps   ymm14, ymm14, 0xb1
    vxorps      ymm14, ymm14, ymm8
    vaddps      ymm9,  ymm11, ymm13
    vaddps      ymm10, ymm12, ymm14
    vsubps      ymm11, ymm11, ymm13
    vsubps      ymm12, ymm12, ymm14
    vmovups     [rdi + %6], ymm9
    vmovups     [rdi + %7], ymm10
    vmovups     [rdi + %8], ymm11
    vmovups     [rdi + %9], ymm12
%endmacro

; void lane4_asm_fft16(data, finish_factor)
global lane4_asm_fft16
align 64
lane4_asm_fft16:
    vmovaps     ymm15, [lane4_sign_odd]
    vmovups     ymm0, [rdi]
    vmovups     ymm1, [rdi + 32]
    vmovups     ymm2, [rdi + 64]
    vmovups     ymm3, [rdi + 96]
    RADIX4 ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7

    vunpcklpd   ymm4, ymm0, ymm1
    vunpckhpd   ymm5, ymm0, ymm1
    vunpcklpd   ymm6, ymm2, ymm3
    vunpckhpd   ymm7, ymm2, ymm3
    vinsertf128 ymm8,  ymm4, xmm6, 1
    vperm2f128  ymm10, ymm4, ymm6, 0x31
    vinsertf128 ymm9,  ymm5, xmm7, 1
    vperm2f128  ymm11, ymm5, ymm7, 0x31

    COMPLEX_MULTIPLY_AVX ymm9,  ymm14, [rsi],       [rsi + 32]
    COMPLEX_MULTIPLY_AVX ymm10, ymm14, [rsi + 64],  [rsi + 96]
    COMPLEX_MULTIPLY_AVX ymm11, ymm14, [rsi + 128], [rsi + 160]
    RADIX4 ymm8,ymm9,ymm10,ymm11, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7

    vmovups     [rdi],      ymm0
    vmovups     [rdi + 32], ymm1
    vmovups     [rdi + 64], ymm2
    vmovups     [rdi + 96], ymm3
    vzeroupper
    ret

; void lane4_asm_fft32(data, permutation, finish_factor)
global lane4_asm_fft32
align 64
lane4_asm_fft32:
    mov         r8d, [rsi]
    vmovups     ymm8,  [rdi + r8*8]
    mov         r8d, [rsi + 4]
    vmovups     ymm9,  [rdi + r8*8]
    mov         r8d, [rsi + 8]
    vmovups     ymm10, [rdi + r8*8]
    mov         r8d, [rsi + 12]
    vmovups     ymm11, [rdi + r8*8]
    mov         r8d, [rsi + 16]
    vmovups     ymm12, [rdi + r8*8]
    mov         r8d, [rsi + 20]
    vmovups     ymm13, [rdi + r8*8]
    mov         r8d, [rsi + 24]
    vmovups     ymm14, [rdi + r8*8]
    mov         r8d, [rsi + 28]
    vmovups     ymm15, [rdi + r8*8]

    vaddps      ymm0, ymm8,  ymm9
    vsubps      ymm1, ymm8,  ymm9
    vaddps      ymm2, ymm10, ymm11
    vsubps      ymm3, ymm10, ymm11
    vaddps      ymm4, ymm12, ymm13
    vsubps      ymm5, ymm12, ymm13
    vaddps      ymm6, ymm14, ymm15
    vsubps      ymm7, ymm14, ymm15

    vmovaps     ymm8, [lane4_sign_odd]
    RADIX4 ymm0,ymm2,ymm4,ymm6, ymm0,ymm2,ymm4,ymm6, \
           ymm8, ymm9,ymm10,ymm11,ymm12
    COMPLEX_MULTIPLY_AVX ymm3, ymm9, \
                         [lane4_positive_diagonal], [lane4_negative_diagonal]
    MULTIPLY_MINUS_I ymm5, ymm8
    COMPLEX_MULTIPLY_AVX ymm7, ymm9, \
                         [lane4_negative_diagonal], [lane4_negative_diagonal]
    RADIX4 ymm1,ymm3,ymm5,ymm7, ymm1,ymm3,ymm5,ymm7, \
           ymm8, ymm9,ymm10,ymm11,ymm12

    FINISH_SMALL ymm0,ymm1,ymm2,ymm3, 0,   0,64,128,192
    FINISH_SMALL ymm4,ymm5,ymm6,ymm7, 192, 32,96,160,224
    vzeroupper
    ret

; ---------------------------------------------------------------------------
; Mixed-radix vector leaves
; ---------------------------------------------------------------------------

; data, permutation, work, number of rows
align 64
lane4_base4:
    vmovaps ymm15, [lane4_sign_odd]
.loop:
    mov     r8d, [rsi]
    vmovups ymm0, [rdi + r8*8]
    mov     r8d, [rsi + 4]
    vmovups ymm1, [rdi + r8*8]
    mov     r8d, [rsi + 8]
    vmovups ymm2, [rdi + r8*8]
    mov     r8d, [rsi + 12]
    vmovups ymm3, [rdi + r8*8]
    RADIX4 ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7
    vmovaps [rdx],      ymm0
    vmovaps [rdx + 32], ymm1
    vmovaps [rdx + 64], ymm2
    vmovaps [rdx + 96], ymm3
    add     rsi, 16
    add     rdx, 128
    sub     rcx, 4
    jnz .loop
    ret

; data, permutation, work, number of rows
align 64
lane4_base8:
    vmovaps ymm15, [lane4_sign_odd]
.loop:
    mov     r8d, [rsi]
    vmovups ymm0, [rdi + r8*8]
    mov     r8d, [rsi + 4]
    vmovups ymm1, [rdi + r8*8]
    mov     r8d, [rsi + 8]
    vmovups ymm2, [rdi + r8*8]
    mov     r8d, [rsi + 12]
    vmovups ymm3, [rdi + r8*8]
    mov     r8d, [rsi + 16]
    vmovups ymm4, [rdi + r8*8]
    mov     r8d, [rsi + 20]
    vmovups ymm5, [rdi + r8*8]
    mov     r8d, [rsi + 24]
    vmovups ymm6, [rdi + r8*8]
    mov     r8d, [rsi + 28]
    vmovups ymm7, [rdi + r8*8]

    BUTTERFLY_NO_TEMP ymm0, ymm1
    BUTTERFLY_NO_TEMP ymm2, ymm3
    BUTTERFLY_NO_TEMP ymm4, ymm5
    BUTTERFLY_NO_TEMP ymm6, ymm7
    RADIX4 ymm0,ymm2,ymm4,ymm6, ymm0,ymm2,ymm4,ymm6, \
           ymm15, ymm8,ymm9,ymm10,ymm11
    COMPLEX_MULTIPLY_AVX ymm3, ymm8, \
                         [lane4_positive_diagonal], [lane4_negative_diagonal]
    MULTIPLY_MINUS_I ymm5, ymm15
    COMPLEX_MULTIPLY_AVX ymm7, ymm8, \
                         [lane4_negative_diagonal], [lane4_negative_diagonal]
    RADIX4 ymm1,ymm3,ymm5,ymm7, ymm1,ymm3,ymm5,ymm7, \
           ymm15, ymm8,ymm9,ymm10,ymm11

    vmovaps [rdx],       ymm0
    vmovaps [rdx + 32],  ymm1
    vmovaps [rdx + 64],  ymm2
    vmovaps [rdx + 96],  ymm3
    vmovaps [rdx + 128], ymm4
    vmovaps [rdx + 160], ymm5
    vmovaps [rdx + 192], ymm6
    vmovaps [rdx + 224], ymm7
    add     rsi, 32
    add     rdx, 256
    sub     rcx, 8
    jnz .loop
    ret

; ---------------------------------------------------------------------------
; Upper radix-4 stages
; ---------------------------------------------------------------------------

; work, previous length in rows, total rows, replicated twiddle stream
%macro DEFINE_STAGE 2
align 64
%1:
    shl     rsi, 5
    shl     rdx, 5
    lea     r11, [rdi + rdx]
    vmovaps ymm15, [lane4_sign_odd]
.block_loop:
    lea     r9,  [rdi + rsi]

    vmovaps ymm0, [rdi]
    vmovaps ymm1, [r9]
    vmovaps ymm2, [r9 + rsi]
    vmovaps ymm3, [r9 + rsi*2]
    RADIX4 ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7
    vmovaps [rdi], ymm0
    vmovaps [r9],  ymm1
    vmovaps [r9 + rsi],   ymm2
    vmovaps [r9 + rsi*2], ymm3

    add     rdi, 32
    add     r9,  32
    mov     r10, rcx
    mov     rdx, rsi
    shr     rdx, 5
    dec     rdx
    jz      .block_done
.butterfly_loop:
    vmovaps ymm0, [rdi]
    vmovaps ymm1, [r9]
    vmovaps ymm2, [r9 + rsi]
    vmovaps ymm3, [r9 + rsi*2]
%if %2
    COMPLEX_MULTIPLY_FMA ymm1, ymm5, [r10],       [r10 + 32]
    COMPLEX_MULTIPLY_FMA ymm2, ymm6, [r10 + 64],  [r10 + 96]
    COMPLEX_MULTIPLY_FMA ymm3, ymm7, [r10 + 128], [r10 + 160]
%else
    COMPLEX_MULTIPLY_AVX ymm1, ymm5, [r10],       [r10 + 32]
    COMPLEX_MULTIPLY_AVX ymm2, ymm6, [r10 + 64],  [r10 + 96]
    COMPLEX_MULTIPLY_AVX ymm3, ymm7, [r10 + 128], [r10 + 160]
%endif
    RADIX4 ymm0,ymm1,ymm2,ymm3, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7
    vmovaps [rdi], ymm0
    vmovaps [r9],  ymm1
    vmovaps [r9 + rsi],   ymm2
    vmovaps [r9 + rsi*2], ymm3
    add     rdi, 32
    add     r9,  32
    add     r10, 192
    dec     rdx
    jnz     .butterfly_loop
.block_done:
    lea     rdi, [r9 + rsi*2]
    cmp     rdi, r11
    jb .block_loop
    ret
%endmacro

DEFINE_STAGE lane4_stage_avx, 0
DEFINE_STAGE lane4_stage_fma, 1

; ---------------------------------------------------------------------------
; Four-frequency outer finish
; ---------------------------------------------------------------------------

; work, data, inner size, finish-factor stream
%macro DEFINE_FINISH 2
align 64
%1:
    mov     r8, rdx
    shl     r8, 3
    lea     r9,  [rsi + r8]
    lea     r10, [r9 + r8]
    lea     rax, [r10 + r8]
    mov     r11, rdx
    shl     r11, 5
    add     r11, rdi
    vmovaps ymm15, [lane4_sign_odd]
.loop:
    vmovaps     ymm0, [rdi]
    vmovaps     ymm1, [rdi + 32]
    vmovaps     ymm2, [rdi + 64]
    vmovaps     ymm3, [rdi + 96]
    vunpcklpd   ymm4, ymm0, ymm1
    vunpckhpd   ymm5, ymm0, ymm1
    vunpcklpd   ymm6, ymm2, ymm3
    vunpckhpd   ymm7, ymm2, ymm3
    vinsertf128 ymm8,  ymm4, xmm6, 1
    vperm2f128  ymm10, ymm4, ymm6, 0x31
    vinsertf128 ymm9,  ymm5, xmm7, 1
    vperm2f128  ymm11, ymm5, ymm7, 0x31
%if %2
    COMPLEX_MULTIPLY_FMA ymm9,  ymm12, [rcx],       [rcx + 32]
    COMPLEX_MULTIPLY_FMA ymm10, ymm13, [rcx + 64],  [rcx + 96]
    COMPLEX_MULTIPLY_FMA ymm11, ymm14, [rcx + 128], [rcx + 160]
%else
    COMPLEX_MULTIPLY_AVX ymm9,  ymm12, [rcx],       [rcx + 32]
    COMPLEX_MULTIPLY_AVX ymm10, ymm13, [rcx + 64],  [rcx + 96]
    COMPLEX_MULTIPLY_AVX ymm11, ymm14, [rcx + 128], [rcx + 160]
%endif
    RADIX4 ymm8,ymm9,ymm10,ymm11, ymm0,ymm1,ymm2,ymm3, \
           ymm15, ymm4,ymm5,ymm6,ymm7
    vmovups [rsi], ymm0
    vmovups [r9],  ymm1
    vmovups [r10], ymm2
    vmovups [rax], ymm3
    add     rdi, 128
    add     rsi, 32
    add     r9,  32
    add     r10, 32
    add     rax, 32
    add     rcx, 192
    cmp     rdi, r11
    jb .loop
    ret
%endmacro

DEFINE_FINISH lane4_finish_avx, 0
DEFINE_FINISH lane4_finish_fma, 1

; ---------------------------------------------------------------------------
; Public plan execution
; ---------------------------------------------------------------------------

%macro DEFINE_EXECUTE 4
global %1
global %2
align 64
%1:
%2:
    test    rdi, rdi
    jz      .error
    test    rsi, rsi
    jz      .error
    cmp     qword [rdi + PLAN_N], 16
    je      .size16
    cmp     qword [rdi + PLAN_N], 32
    je      .size32
    cmp     qword [rdi + PLAN_N], 64
    je      .size64

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
    call    lane4_base4
    mov     ebx, 4
    jmp     .stage_test
.base8:
    call    lane4_base8
    mov     ebx, 8
.stage_test:
    cmp     rbx, r13
    jae     .finish
.stage_loop:
    mov     rdi, r14
    mov     rsi, rbx
    mov     rdx, r13
    mov     rcx, r15
    call    %3
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
    call    %4
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
.size16:
    mov     rdx, [rdi + PLAN_FINISH_FACTOR]
    mov     rdi, rsi
    mov     rsi, rdx
    xor     eax, eax
    jmp     lane4_asm_fft16
.size32:
    mov     rdx, [rdi + PLAN_FINISH_FACTOR]
    mov     rcx, [rdi + PLAN_PERMUTATION]
    mov     rdi, rsi
    mov     rsi, rcx
    xor     eax, eax
    jmp     lane4_asm_fft32
.size64:
    push    rbp
    push    r12
    sub     rsp, 8
    mov     rbp, rdi
    mov     r12, rsi
    mov     rdi, r12
    mov     rsi, [rbp + PLAN_PERMUTATION]
    mov     rdx, [rbp + PLAN_WORK]
    mov     rcx, 16
    call    lane4_base4
    mov     rdi, [rbp + PLAN_WORK]
    mov     rsi, 4
    mov     rdx, 16
    mov     rcx, [rbp + PLAN_TWIDDLE]
    call    %3
    mov     rdi, [rbp + PLAN_WORK]
    mov     rsi, r12
    mov     rdx, 16
    mov     rcx, [rbp + PLAN_FINISH_FACTOR]
    call    %4
    add     rsp, 8
    pop     r12
    pop     rbp
    vzeroupper
    xor     eax, eax
    ret
.error:
    mov     eax, -1
    ret
%endmacro

DEFINE_EXECUTE lane4_avx_fft_execute, lane4_avx2_fft_execute, \
               lane4_stage_avx, lane4_finish_avx
DEFINE_EXECUTE lane4_avx_fma_fft_execute, lane4_fft_execute, \
               lane4_stage_fma, lane4_finish_fma
