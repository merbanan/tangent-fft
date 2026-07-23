; AVX tangent-FFT combine kernels for the x86-64 System V ABI.
;
; The structure follows the useful part of FFmpeg's x86 AVTX design:
; four complex samples occupy one YMM register, the four split-radix quarters
; stay in separate streams, and complex rotations use lane swaps plus sign
; masks.  Unlike a conventional FFT, the S/S2/S4 kernels retain the tangent
; FFT's reduced-multiply 1+i*tan and cot-i forms.

bits 64
default rel

section .rodata align=32
align 32
sign_even: dd 0x80000000, 0, 0x80000000, 0
           dd 0x80000000, 0, 0x80000000, 0
sign_odd:  dd 0, 0x80000000, 0, 0x80000000
           dd 0, 0x80000000, 0, 0x80000000
sign_all:  times 8 dd 0x80000000
sqrt_two:  dd 1.4142135623730951
align 16
sign_lane3: dd 0, 0, 0, 0x80000000
base_s4_scale: dd 1.0, 1.0, 1.4142135623730951, 1.4142135623730951

section .text

; rdi=input, rsi=output, rdx=uint32 permutation, rcx=count
; Gather four complete complex-float values as four 64-bit lanes.
global tangent_x86_permute
tangent_x86_permute:
    xor         rax, rax
.permute_loop:
    vpcmpeqd    ymm1, ymm1, ymm1
    vmovdqu     xmm2, [rdx + rax*4]
    vgatherdpd  ymm0, [rdi + xmm2*8], ymm1
    vmovdqu     [rsi + rax*8], ymm0
    add         rax, 4
    cmp         rax, rcx
    jb          .permute_loop
    vzeroupper
    ret

; rdi=data, rsi=offsets in complex samples, rdx=count
global tangent_x86_batch_base
tangent_x86_batch_base:
    xor         rax, rax
.base_loop:
    cmp         rax, rdx
    jae         .base_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovq       xmm0, [r8]
    vmovq       xmm1, [r8 + 8]
    vaddps      xmm2, xmm0, xmm1
    vsubps      xmm3, xmm0, xmm1
    vmovq       [r8], xmm2
    vmovq       [r8 + 8], xmm3
    inc         rax
    jmp         .base_loop
.base_done:
    vzeroupper
    ret

global tangent_x86_batch_base_s4
tangent_x86_batch_base_s4:
    vbroadcastss xmm4, [sqrt_two]
    xor         rax, rax
.base_s4_loop:
    cmp         rax, rdx
    jae         .base_s4_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovq       xmm0, [r8]
    vmovq       xmm1, [r8 + 8]
    vaddps      xmm2, xmm0, xmm1
    vsubps      xmm3, xmm0, xmm1
    vmulps      xmm3, xmm3, xmm4
    vmovq       [r8], xmm2
    vmovq       [r8 + 8], xmm3
    inc         rax
    jmp         .base_s4_loop
.base_s4_done:
    vzeroupper
    ret

; Duplicate four packed floats as [x0,x0,x1,x1,x2,x2,x3,x3].
; %1 = destination YMM, %2 = source address, %3/%4 = temporary XMM.
%macro DUP4 4
    vmovups     xmm%3, %2
    vpermilps   xmm%4, xmm%3, 0x50
    vpermilps   xmm%3, xmm%3, 0xfa
    vinsertf128 %1, ymm%4, xmm%3, 1
%endmacro

; Form the four output streams from u0=%1, u1=%2, a=%3 and b=%4.
; Clobbers %3-%8.  Output registers are %3=o0, %5=o1, %4=o2, %6=o3.
%macro FINISH_UNSCALED 8
    vaddps      %5, %3, %4                 ; sum
    vsubps      %6, %3, %4                 ; difference
    vaddps      %3, %1, %5                 ; o0
    vsubps      %4, %1, %5                 ; o2
    vpermilps   %6, %6, 0xb1               ; difference.imre
    vxorps      %7, %6, [sign_odd]          ; -i*difference
    vxorps      %8, %6, [sign_even]         ; +i*difference
    vaddps      %5, %2, %7                 ; o1
    vaddps      %6, %2, %8                 ; o3
%endmacro

; rdi=data, rsi=quarter, rdx=factor, rcx=count
global tangent_x86_normal
tangent_x86_normal:
    test        rcx, rcx
    jz          .normal_done
    shl         rsi, 3
    lea         r8, [rdi + rsi]             ; quarter 1
    lea         r9, [r8 + rsi]              ; quarter 2
    lea         r10, [r9 + rsi]             ; quarter 3
    xor         rax, rax
.normal_loop:
    vmovups     ymm0, [rdi + rax*8]
    vmovups     ymm1, [r8 + rax*8]
    vmovups     ymm2, [r9 + rax*8]
    vmovups     ymm3, [r10 + rax*8]
    vmovups     ymm4, [rdx + rax*8]

    vmovsldup   ymm5, ymm4                  ; cosine
    vmovshdup   ymm6, ymm4                  ; sine

    vpermilps   ymm7, ymm2, 0xb1
    vmulps      ymm2, ymm2, ymm5
    vmulps      ymm7, ymm7, ymm6
    vaddsubps   ymm2, ymm2, ymm7            ; a = factor*z

    vpermilps   ymm7, ymm3, 0xb1
    vmulps      ymm3, ymm3, ymm5
    vmulps      ymm7, ymm7, ymm6
    vxorps      ymm7, ymm7, [sign_all]
    vaddsubps   ymm3, ymm3, ymm7            ; b = conj(factor)*zp

    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7

    vmovups     [rdi + rax*8], ymm2
    vmovups     [r8 + rax*8], ymm4
    vmovups     [r9 + rax*8], ymm3
    vmovups     [r10 + rax*8], ymm5
    add         rax, 4
    cmp         rax, rcx
    jb          .normal_loop
.normal_done:
    vzeroupper
    ret

; Shared unscaled S kernel generator.
; rdi=data, rsi=quarter, rdx=value, rcx=count
%macro S_KERNEL 2
global %1
%1:
    test        rcx, rcx
    jz          %%done
    shl         rsi, 3
    lea         r8, [rdi + rsi]
    lea         r9, [r8 + rsi]
    lea         r10, [r9 + rsi]
    xor         rax, rax
%%loop:
    vmovups     ymm0, [rdi + rax*8]
    vmovups     ymm1, [r8 + rax*8]
    vmovups     ymm2, [r9 + rax*8]
    vmovups     ymm3, [r10 + rax*8]
    DUP4        ymm4, [rdx + rax*4], 5, 6
%if %2 = 0
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_even]
    vxorps      ymm6, ymm6, [sign_odd]
    vmulps      ymm5, ymm5, ymm4
    vmulps      ymm6, ymm6, ymm4
    vaddps      ymm2, ymm2, ymm5             ; z*(1+i*v)
    vaddps      ymm3, ymm3, ymm6             ; zp*(1-i*v)
%else
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_odd]
    vxorps      ymm6, ymm6, [sign_even]
    vmulps      ymm2, ymm2, ymm4
    vmulps      ymm3, ymm3, ymm4
    vaddps      ymm2, ymm2, ymm5             ; z*(v-i)
    vaddps      ymm3, ymm3, ymm6             ; zp*(v+i)
%endif
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmovups     [rdi + rax*8], ymm2
    vmovups     [r8 + rax*8], ymm4
    vmovups     [r9 + rax*8], ymm3
    vmovups     [r10 + rax*8], ymm5
    add         rax, 4
    cmp         rax, rcx
    jb          %%loop
%%done:
    vzeroupper
    ret
%endmacro

S_KERNEL tangent_x86_s_low, 0
S_KERNEL tangent_x86_s_high, 1

; Shared S2 kernel generator.
; rdi=data, rsi=quarter, rdx=value, rcx=low scale,
; r8=high scale, r9=count
%macro S2_KERNEL 2
global %1
%1:
    test        r9, r9
    jz          %%done
    shl         rsi, 3
    lea         r10, [rdi + rsi]
    lea         r11, [r10 + rsi]
    add         rsi, r11                     ; quarter 3 pointer
    xor         rax, rax
%%loop:
    vmovups     ymm0, [rdi + rax*8]
    vmovups     ymm1, [r10 + rax*8]
    vmovups     ymm2, [r11 + rax*8]
    vmovups     ymm3, [rsi + rax*8]
    DUP4        ymm4, [rdx + rax*4], 5, 6
%if %2 = 0
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_even]
    vxorps      ymm6, ymm6, [sign_odd]
    vmulps      ymm5, ymm5, ymm4
    vmulps      ymm6, ymm6, ymm4
    vaddps      ymm2, ymm2, ymm5
    vaddps      ymm3, ymm3, ymm6
%else
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_odd]
    vxorps      ymm6, ymm6, [sign_even]
    vmulps      ymm2, ymm2, ymm4
    vmulps      ymm3, ymm3, ymm4
    vaddps      ymm2, ymm2, ymm5
    vaddps      ymm3, ymm3, ymm6
%endif
    vaddps      ymm4, ymm2, ymm3             ; low
    vsubps      ymm5, ymm2, ymm3             ; high
    DUP4        ymm6, [rcx + rax*4], 2, 3
    DUP4        ymm7, [r8 + rax*4], 2, 3
    vmulps      ymm4, ymm4, ymm6
    vmulps      ymm5, ymm5, ymm7

    vaddps      ymm2, ymm0, ymm4
    vsubps      ymm3, ymm0, ymm4
    vpermilps   ymm5, ymm5, 0xb1
    vxorps      ymm6, ymm5, [sign_odd]
    vxorps      ymm7, ymm5, [sign_even]
    vaddps      ymm4, ymm1, ymm6
    vaddps      ymm5, ymm1, ymm7

    vmovups     [rdi + rax*8], ymm2
    vmovups     [r10 + rax*8], ymm4
    vmovups     [r11 + rax*8], ymm3
    vmovups     [rsi + rax*8], ymm5
    add         rax, 4
    cmp         rax, r9
    jb          %%loop
%%done:
    vzeroupper
    ret
%endmacro

S2_KERNEL tangent_x86_s2_low, 0
S2_KERNEL tangent_x86_s2_high, 1

; Shared S4 kernel generator.
; rdi=data, rsi=quarter, rdx=value, rcx=scale0, r8=scale1, r9=scale2,
; [rsp+8]=scale3, [rsp+16]=count
%macro S4_KERNEL 2
global %1
%1:
    push        rbx
    push        r12
    push        r13
    mov         rbx, [rsp + 32]              ; scale3
    mov         r13, [rsp + 40]              ; count
    test        r13, r13
    jz          %%done
    shl         rsi, 3
    lea         r10, [rdi + rsi]
    lea         r11, [r10 + rsi]
    lea         r12, [r11 + rsi]
    xor         rax, rax
%%loop:
    vmovups     ymm0, [rdi + rax*8]
    vmovups     ymm1, [r10 + rax*8]
    vmovups     ymm2, [r11 + rax*8]
    vmovups     ymm3, [r12 + rax*8]
    DUP4        ymm4, [rdx + rax*4], 5, 6
%if %2 = 0
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_even]
    vxorps      ymm6, ymm6, [sign_odd]
    vmulps      ymm5, ymm5, ymm4
    vmulps      ymm6, ymm6, ymm4
    vaddps      ymm2, ymm2, ymm5
    vaddps      ymm3, ymm3, ymm6
%else
    vpermilps   ymm5, ymm2, 0xb1
    vpermilps   ymm6, ymm3, 0xb1
    vxorps      ymm5, ymm5, [sign_odd]
    vxorps      ymm6, ymm6, [sign_even]
    vmulps      ymm2, ymm2, ymm4
    vmulps      ymm3, ymm3, ymm4
    vaddps      ymm2, ymm2, ymm5
    vaddps      ymm3, ymm3, ymm6
%endif
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7

    DUP4        ymm0, [rcx + rax*4], 7, 8
    DUP4        ymm1, [r8 + rax*4], 7, 8
    vmulps      ymm2, ymm2, ymm0
    vmulps      ymm4, ymm4, ymm1
    DUP4        ymm0, [r9 + rax*4], 7, 8
    DUP4        ymm1, [rbx + rax*4], 7, 8
    vmulps      ymm3, ymm3, ymm0
    vmulps      ymm5, ymm5, ymm1

    vmovups     [rdi + rax*8], ymm2
    vmovups     [r10 + rax*8], ymm4
    vmovups     [r11 + rax*8], ymm3
    vmovups     [r12 + rax*8], ymm5
    add         rax, 4
    cmp         rax, r13
    jb          %%loop
%%done:
    vzeroupper
    pop         r13
    pop         r12
    pop         rbx
    ret
%endmacro

S4_KERNEL tangent_x86_s4_low, 0
S4_KERNEL tangent_x86_s4_high, 1

; Load and multiply z/zp by a shared complex-factor vector.
; %1/%2 are z/zp, %3/%4 are duplicated real/imaginary factor lanes,
; %5/%6 are temporaries.
%macro APPLY_FACTOR 6
    vpermilps   %5, %1, 0xb1
    vmulps      %5, %5, %4
    vfmaddsub213ps %1, %3, %5
    vpermilps   %6, %2, 0xb1
    vmulps      %6, %6, %4
    vfmsubadd213ps %2, %3, %6
%endmacro

; Tangent multiplication with one real coefficient per complex value. The
; vector beginning at the N/8 boundary mixes the two forms and uses the
; general complex multiply; all other vectors use two multiplies total.
%macro APPLY_TANGENT 9
    cmp         %4, %5
    jb          %%low
    ja          %%high
    vmovsldup   %8, %3
    vmovshdup   %9, %3
    APPLY_FACTOR %1, %2, %8, %9, %6, %7
    jmp         %%done
%%low:
    vmovshdup   %8, %3
    vpermilps   %6, %1, 0xb1
    vpermilps   %7, %2, 0xb1
    vxorps      %6, %6, [sign_even]
    vxorps      %7, %7, [sign_odd]
    vfmadd231ps %1, %6, %8
    vfmadd231ps %2, %7, %8
    jmp         %%done
%%high:
    vmovsldup   %8, %3
    vpermilps   %6, %1, 0xb1
    vpermilps   %7, %2, 0xb1
    vxorps      %6, %6, [sign_odd]
    vxorps      %7, %7, [sign_even]
    vfmadd213ps %1, %8, %6
    vfmadd213ps %2, %8, %7
%%done:
%endmacro

; Fixed-leaf tables are eight tables x five levels x four complex values.
; Every 32-byte level slot is directly vector-loadable; real scale values are
; duplicated in the real and imaginary lanes by the C plan builder.
%define LEAF_TABLE_STRIDE 160
%define LEAF_SCALED       (0 * LEAF_TABLE_STRIDE)
%define LEAF_TANGENT      (1 * LEAF_TABLE_STRIDE)
%define LEAF_S2_LOW       (2 * LEAF_TABLE_STRIDE)
%define LEAF_S2_HIGH      (3 * LEAF_TABLE_STRIDE)
%define LEAF_S4_0         (4 * LEAF_TABLE_STRIDE)
%define LEAF_S4_1         (5 * LEAF_TABLE_STRIDE)
%define LEAF_S4_2         (6 * LEAF_TABLE_STRIDE)
%define LEAF_S4_3         (7 * LEAF_TABLE_STRIDE)
%define LEAF_LEVEL(level) ((level) * 32)

%define KIND_NORMAL 0
%define KIND_S      1
%define KIND_S2     2
%define KIND_S4     3

; r8 points at the start of one fixed leaf.
%macro LEAF_BASE 2
    vmovq       xmm0, [r8 + %1]
    vmovq       xmm1, [r8 + %1 + 8]
    vaddps      xmm2, xmm0, xmm1
    vsubps      xmm3, xmm0, xmm1
%if %2 = KIND_S4
    vbroadcastss xmm4, [sqrt_two]
    vmulps      xmm3, xmm3, xmm4
%endif
    vmovq       [r8 + %1], xmm2
    vmovq       [r8 + %1 + 8], xmm3
%endmacro

; Complete one combine. %1 is the byte offset in the leaf, %2 is the level
; (2/3/4), and %3 is the tangent transform kind.
%macro LEAF_COMBINE 3
%if %2 = 2
    vmovq       xmm0, [r8 + %1]
    vmovq       xmm1, [r8 + %1 + 8]
    vmovq       xmm2, [r8 + %1 + 16]
    vmovq       xmm3, [r8 + %1 + 24]
%if %3 = KIND_S2
    vaddps      xmm4, xmm2, xmm3
    vsubps      xmm5, xmm2, xmm3
    vmovq       xmm14, [rcx + LEAF_S2_LOW + LEAF_LEVEL(2)]
    vmovq       xmm15, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(2)]
    vmulps      xmm4, xmm4, xmm14
    vmulps      xmm5, xmm5, xmm15
    vaddps      xmm2, xmm0, xmm4
    vsubps      xmm3, xmm0, xmm4
    vpermilps   xmm5, xmm5, 0xb1
    vxorps      xmm6, xmm5, [sign_odd]
    vxorps      xmm7, xmm5, [sign_even]
    vaddps      xmm4, xmm1, xmm6
    vaddps      xmm5, xmm1, xmm7
%else
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
%if %3 = KIND_S4
    vmovq       xmm12, [rcx + LEAF_S4_0 + LEAF_LEVEL(2)]
    vmovq       xmm13, [rcx + LEAF_S4_1 + LEAF_LEVEL(2)]
    vmovq       xmm14, [rcx + LEAF_S4_2 + LEAF_LEVEL(2)]
    vmovq       xmm15, [rcx + LEAF_S4_3 + LEAF_LEVEL(2)]
    vmulps      xmm2, xmm2, xmm12
    vmulps      xmm4, xmm4, xmm13
    vmulps      xmm3, xmm3, xmm14
    vmulps      xmm5, xmm5, xmm15
%endif
%endif
    vmovq       [r8 + %1], xmm2
    vmovq       [r8 + %1 + 8], xmm4
    vmovq       [r8 + %1 + 16], xmm3
    vmovq       [r8 + %1 + 24], xmm5
%elif %2 = 3
    vmovups     xmm0, [r8 + %1]
    vmovups     xmm1, [r8 + %1 + 16]
    vmovups     xmm2, [r8 + %1 + 32]
    vmovups     xmm3, [r8 + %1 + 48]
%if %3 = KIND_NORMAL
    vmovups     xmm12, [rcx + LEAF_SCALED + LEAF_LEVEL(3)]
%else
    vmovups     xmm12, [rcx + LEAF_TANGENT + LEAF_LEVEL(3)]
%endif
    vmovsldup   xmm13, xmm12
    vmovshdup   xmm12, xmm12
    APPLY_FACTOR xmm2, xmm3, xmm13, xmm12, xmm6, xmm7
%if %3 = KIND_S2
    vaddps      xmm4, xmm2, xmm3
    vsubps      xmm5, xmm2, xmm3
    vmovups     xmm14, [rcx + LEAF_S2_LOW + LEAF_LEVEL(3)]
    vmovups     xmm15, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(3)]
    vmulps      xmm4, xmm4, xmm14
    vmulps      xmm5, xmm5, xmm15
    vaddps      xmm2, xmm0, xmm4
    vsubps      xmm3, xmm0, xmm4
    vpermilps   xmm5, xmm5, 0xb1
    vxorps      xmm6, xmm5, [sign_odd]
    vxorps      xmm7, xmm5, [sign_even]
    vaddps      xmm4, xmm1, xmm6
    vaddps      xmm5, xmm1, xmm7
%else
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
%if %3 = KIND_S4
    vmovups     xmm12, [rcx + LEAF_S4_0 + LEAF_LEVEL(3)]
    vmovups     xmm13, [rcx + LEAF_S4_1 + LEAF_LEVEL(3)]
    vmovups     xmm14, [rcx + LEAF_S4_2 + LEAF_LEVEL(3)]
    vmovups     xmm15, [rcx + LEAF_S4_3 + LEAF_LEVEL(3)]
    vmulps      xmm2, xmm2, xmm12
    vmulps      xmm4, xmm4, xmm13
    vmulps      xmm3, xmm3, xmm14
    vmulps      xmm5, xmm5, xmm15
%endif
%endif
    vmovups     [r8 + %1], xmm2
    vmovups     [r8 + %1 + 16], xmm4
    vmovups     [r8 + %1 + 32], xmm3
    vmovups     [r8 + %1 + 48], xmm5
%elif %2 = 4
    vmovups     ymm0, [r8 + %1]
    vmovups     ymm1, [r8 + %1 + 32]
    vmovups     ymm2, [r8 + %1 + 64]
    vmovups     ymm3, [r8 + %1 + 96]
%if %3 = KIND_NORMAL
    vmovups     ymm12, [rcx + LEAF_SCALED + LEAF_LEVEL(4)]
%else
    vmovups     ymm12, [rcx + LEAF_TANGENT + LEAF_LEVEL(4)]
%endif
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    APPLY_FACTOR ymm2, ymm3, ymm13, ymm12, ymm6, ymm7
%if %3 = KIND_S2
    vaddps      ymm4, ymm2, ymm3
    vsubps      ymm5, ymm2, ymm3
    vmovups     ymm14, [rcx + LEAF_S2_LOW + LEAF_LEVEL(4)]
    vmovups     ymm15, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(4)]
    vmulps      ymm4, ymm4, ymm14
    vmulps      ymm5, ymm5, ymm15
    vaddps      ymm2, ymm0, ymm4
    vsubps      ymm3, ymm0, ymm4
    vpermilps   ymm5, ymm5, 0xb1
    vxorps      ymm6, ymm5, [sign_odd]
    vxorps      ymm7, ymm5, [sign_even]
    vaddps      ymm4, ymm1, ymm6
    vaddps      ymm5, ymm1, ymm7
%else
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
%if %3 = KIND_S4
    vmovups     ymm12, [rcx + LEAF_S4_0 + LEAF_LEVEL(4)]
    vmovups     ymm13, [rcx + LEAF_S4_1 + LEAF_LEVEL(4)]
    vmovups     ymm14, [rcx + LEAF_S4_2 + LEAF_LEVEL(4)]
    vmovups     ymm15, [rcx + LEAF_S4_3 + LEAF_LEVEL(4)]
    vmulps      ymm2, ymm2, ymm12
    vmulps      ymm4, ymm4, ymm13
    vmulps      ymm3, ymm3, ymm14
    vmulps      ymm5, ymm5, ymm15
%endif
%endif
    vmovups     [r8 + %1], ymm2
    vmovups     [r8 + %1 + 32], ymm4
    vmovups     [r8 + %1 + 64], ymm3
    vmovups     [r8 + %1 + 96], ymm5
%endif
%endmacro

%macro LEAF2_BODY 1
%if %1 = KIND_S2
    LEAF_BASE 0, KIND_S4
%else
    LEAF_BASE 0, KIND_NORMAL
%endif
    LEAF_COMBINE 0, 2, %1
%endmacro

%macro LEAF3_BODY 1
%if %1 = KIND_S || %1 = KIND_S4
    LEAF_BASE 0, KIND_S4
%else
    LEAF_BASE 0, KIND_NORMAL
%endif
    LEAF_BASE 32, KIND_NORMAL
    LEAF_BASE 48, KIND_NORMAL
%if %1 = KIND_NORMAL
    LEAF_COMBINE 0, 2, KIND_NORMAL
%elif %1 = KIND_S
    LEAF_COMBINE 0, 2, KIND_S2
%elif %1 = KIND_S2
    LEAF_COMBINE 0, 2, KIND_S4
%else
    LEAF_COMBINE 0, 2, KIND_S2
%endif
    LEAF_COMBINE 0, 3, %1
%endmacro

%macro LEAF4_BODY 1
%if %1 = KIND_S2
    LEAF_BASE 0, KIND_S4
%else
    LEAF_BASE 0, KIND_NORMAL
%endif
    LEAF_BASE 32, KIND_NORMAL
    LEAF_BASE 48, KIND_NORMAL
    LEAF_BASE 64, KIND_NORMAL
    LEAF_BASE 96, KIND_NORMAL
%if %1 = KIND_NORMAL
    LEAF_COMBINE 0, 2, KIND_NORMAL
%elif %1 = KIND_S
    LEAF_COMBINE 0, 2, KIND_S4
%elif %1 = KIND_S2
    LEAF_COMBINE 0, 2, KIND_S2
%else
    LEAF_COMBINE 0, 2, KIND_S4
%endif
    LEAF_COMBINE 64, 2, KIND_S
    LEAF_COMBINE 96, 2, KIND_S
%if %1 = KIND_NORMAL
    LEAF_COMBINE 0, 3, KIND_NORMAL
%elif %1 = KIND_S
    LEAF_COMBINE 0, 3, KIND_S2
%elif %1 = KIND_S2
    LEAF_COMBINE 0, 3, KIND_S4
%else
    LEAF_COMBINE 0, 3, KIND_S2
%endif
    LEAF_COMBINE 0, 4, %1
%endmacro

; Register-resident fixed leaves.  Each XMM holds two adjacent complex
; samples.  Only the initial loads and final stores touch leaf data.
%macro REG_BASE 2
    vpermilps   xmm8, %1, 0x4e
    vaddps      xmm9, %1, xmm8
    vsubps      xmm8, xmm8, %1
    vblendps    %1, xmm9, xmm8, 0x0c
%if %2 = KIND_S4
    vmulps      %1, %1, [base_s4_scale]
%endif
%endmacro

; Inputs are [u0,u1] and [z,zp]. Outputs become [o0,o1] and [o2,o3].
%macro REG_Q1 3
    vpermilps   xmm8, %2, 0x4e
    vaddps      xmm9, %2, xmm8
    vsubps      xmm8, xmm8, %2
    vblendps    %2, xmm9, xmm8, 0x0c         ; [sum,difference]
%if %3 = KIND_S2
    vmovq       xmm8, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(2)]
    vmovq       xmm9, [rcx + LEAF_S2_LOW + LEAF_LEVEL(2)]
    vmovlhps    xmm9, xmm9, xmm8
    vmulps      %2, %2, xmm9
%endif
    vpermilps   xmm8, %2, 0xb4
    vxorps      xmm8, xmm8, [sign_lane3]
    vaddps      xmm9, %1, xmm8
    vsubps      %2, %1, xmm8
    vmovaps     %1, xmm9
%if %3 = KIND_S4
    vmovq       xmm8, [rcx + LEAF_S4_0 + LEAF_LEVEL(2)]
    vmovq       xmm9, [rcx + LEAF_S4_1 + LEAF_LEVEL(2)]
    vmovlhps    xmm8, xmm8, xmm9
    vmulps      %1, %1, xmm8
    vmovq       xmm8, [rcx + LEAF_S4_2 + LEAF_LEVEL(2)]
    vmovq       xmm9, [rcx + LEAF_S4_3 + LEAF_LEVEL(2)]
    vmovlhps    xmm8, xmm8, xmm9
    vmulps      %2, %2, xmm8
%endif
%endmacro

; Q2 consumes xmm0..xmm3 and returns the four output streams in the same
; registers. xmm4..xmm7, which hold the other FFT16 children, are preserved.
%macro REG_Q2 1
%if %1 = KIND_NORMAL
    vmovups     xmm12, [rcx + LEAF_SCALED + LEAF_LEVEL(3)]
%else
    vmovups     xmm12, [rcx + LEAF_TANGENT + LEAF_LEVEL(3)]
%endif
    vmovsldup   xmm13, xmm12
    vmovshdup   xmm12, xmm12
    APPLY_FACTOR xmm2, xmm3, xmm13, xmm12, xmm8, xmm9
%if %1 = KIND_S2
    vaddps      xmm10, xmm2, xmm3
    vsubps      xmm11, xmm2, xmm3
    vmulps      xmm10, xmm10, [rcx + LEAF_S2_LOW + LEAF_LEVEL(3)]
    vmulps      xmm11, xmm11, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(3)]
    vaddps      xmm12, xmm0, xmm10
    vsubps      xmm13, xmm0, xmm10
    vpermilps   xmm11, xmm11, 0xb1
    vxorps      xmm14, xmm11, [sign_odd]
    vxorps      xmm15, xmm11, [sign_even]
    vaddps      xmm10, xmm1, xmm14
    vaddps      xmm11, xmm1, xmm15
    vmovaps     xmm0, xmm12
    vmovaps     xmm1, xmm10
    vmovaps     xmm2, xmm13
    vmovaps     xmm3, xmm11
%else
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm10, xmm11, xmm8, xmm9
%if %1 = KIND_S4
    vmulps      xmm2, xmm2, [rcx + LEAF_S4_0 + LEAF_LEVEL(3)]
    vmulps      xmm10, xmm10, [rcx + LEAF_S4_1 + LEAF_LEVEL(3)]
    vmulps      xmm3, xmm3, [rcx + LEAF_S4_2 + LEAF_LEVEL(3)]
    vmulps      xmm11, xmm11, [rcx + LEAF_S4_3 + LEAF_LEVEL(3)]
%endif
    vmovaps     xmm0, xmm2
    vmovaps     xmm1, xmm10
    vmovaps     xmm2, xmm3
    vmovaps     xmm3, xmm11
%endif
%endmacro

; YMM0..YMM3 are the four FFT16 streams. Results remain in those registers.
%macro REG_Q4 1
%if %1 = KIND_NORMAL
    vmovups     ymm12, [rcx + LEAF_SCALED + LEAF_LEVEL(4)]
%else
    vmovups     ymm12, [rcx + LEAF_TANGENT + LEAF_LEVEL(4)]
%endif
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    APPLY_FACTOR ymm2, ymm3, ymm13, ymm12, ymm8, ymm9
%if %1 = KIND_S2
    vaddps      ymm10, ymm2, ymm3
    vsubps      ymm11, ymm2, ymm3
    vmulps      ymm10, ymm10, [rcx + LEAF_S2_LOW + LEAF_LEVEL(4)]
    vmulps      ymm11, ymm11, [rcx + LEAF_S2_HIGH + LEAF_LEVEL(4)]
    vaddps      ymm12, ymm0, ymm10
    vsubps      ymm13, ymm0, ymm10
    vpermilps   ymm11, ymm11, 0xb1
    vxorps      ymm14, ymm11, [sign_odd]
    vxorps      ymm15, ymm11, [sign_even]
    vaddps      ymm10, ymm1, ymm14
    vaddps      ymm11, ymm1, ymm15
    vmovaps     ymm0, ymm12
    vmovaps     ymm1, ymm10
    vmovaps     ymm2, ymm13
    vmovaps     ymm3, ymm11
%else
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm10, ymm11, ymm8, ymm9
%if %1 = KIND_S4
    vmulps      ymm2, ymm2, [rcx + LEAF_S4_0 + LEAF_LEVEL(4)]
    vmulps      ymm10, ymm10, [rcx + LEAF_S4_1 + LEAF_LEVEL(4)]
    vmulps      ymm3, ymm3, [rcx + LEAF_S4_2 + LEAF_LEVEL(4)]
    vmulps      ymm11, ymm11, [rcx + LEAF_S4_3 + LEAF_LEVEL(4)]
%endif
    vmovaps     ymm0, ymm2
    vmovaps     ymm1, ymm10
    vmovaps     ymm2, ymm3
    vmovaps     ymm3, ymm11
%endif
%endmacro

%macro REG_LEAF2_CORE 1
%if %1 = KIND_S2
    REG_BASE xmm0, KIND_S4
%else
    REG_BASE xmm0, KIND_NORMAL
%endif
    REG_Q1 xmm0, xmm1, %1
    vmovups     [r8], xmm0
    vmovups     [r8 + 16], xmm1
%endmacro

%macro REG_LEAF3_CORE 1
%if %1 = KIND_S || %1 = KIND_S4
    REG_BASE xmm0, KIND_S4
%else
    REG_BASE xmm0, KIND_NORMAL
%endif
    REG_BASE xmm2, KIND_NORMAL
    REG_BASE xmm3, KIND_NORMAL
%if %1 = KIND_NORMAL
    REG_Q1 xmm0, xmm1, KIND_NORMAL
%elif %1 = KIND_S
    REG_Q1 xmm0, xmm1, KIND_S2
%elif %1 = KIND_S2
    REG_Q1 xmm0, xmm1, KIND_S4
%else
    REG_Q1 xmm0, xmm1, KIND_S2
%endif
    REG_Q2 %1
    vmovups     [r8], xmm0
    vmovups     [r8 + 16], xmm1
    vmovups     [r8 + 32], xmm2
    vmovups     [r8 + 48], xmm3
%endmacro

%macro REG_LEAF4_CALC 1
%if %1 = KIND_S2
    REG_BASE xmm0, KIND_S4
%else
    REG_BASE xmm0, KIND_NORMAL
%endif
    REG_BASE xmm2, KIND_NORMAL
    REG_BASE xmm3, KIND_NORMAL
    REG_BASE xmm4, KIND_NORMAL
    REG_BASE xmm6, KIND_NORMAL
%if %1 = KIND_NORMAL
    REG_Q1 xmm0, xmm1, KIND_NORMAL
%elif %1 = KIND_S
    REG_Q1 xmm0, xmm1, KIND_S4
%elif %1 = KIND_S2
    REG_Q1 xmm0, xmm1, KIND_S2
%else
    REG_Q1 xmm0, xmm1, KIND_S4
%endif
    REG_Q1 xmm4, xmm5, KIND_S
    REG_Q1 xmm6, xmm7, KIND_S
%if %1 = KIND_NORMAL
    REG_Q2 KIND_NORMAL
%elif %1 = KIND_S
    REG_Q2 KIND_S2
%elif %1 = KIND_S2
    REG_Q2 KIND_S4
%else
    REG_Q2 KIND_S2
%endif
    vinsertf128 ymm0, ymm0, xmm1, 1
    vinsertf128 ymm1, ymm2, xmm3, 1
    vinsertf128 ymm2, ymm4, xmm5, 1
    vinsertf128 ymm3, ymm6, xmm7, 1
    REG_Q4 %1
%endmacro

%macro REG_LEAF4_CORE 1
    REG_LEAF4_CALC %1
    vmovups     [r8], ymm0
    vmovups     [r8 + 32], ymm1
    vmovups     [r8 + 64], ymm2
    vmovups     [r8 + 96], ymm3
%endmacro

%macro REG_LEAF2_BODY 1
    vmovups     xmm0, [r8]
    vmovups     xmm1, [r8 + 16]
    REG_LEAF2_CORE %1
%endmacro

%macro REG_LEAF3_BODY 1
    vmovups     xmm0, [r8]
    vmovups     xmm1, [r8 + 16]
    vmovups     xmm2, [r8 + 32]
    vmovups     xmm3, [r8 + 48]
    REG_LEAF3_CORE %1
%endmacro

%macro REG_LEAF4_BODY 1
    vmovups     xmm0, [r8]
    vmovups     xmm1, [r8 + 16]
    vmovups     xmm2, [r8 + 32]
    vmovups     xmm3, [r8 + 48]
    vmovups     xmm4, [r8 + 64]
    vmovups     xmm5, [r8 + 80]
    vmovups     xmm6, [r8 + 96]
    vmovups     xmm7, [r8 + 112]
    REG_LEAF4_CORE %1
%endmacro

; data, offsets, count, packed tables, kind.  Kind dispatch occurs once per
; homogeneous batch, leaving the inner leaf loop branch-free.
%macro MAKE_LEAF_BATCH 3
global %1
%1:
    cmp         r8d, KIND_S
    je          %%kind_s
    cmp         r8d, KIND_S2
    je          %%kind_s2
    cmp         r8d, KIND_S4
    je          %%kind_s4
    xor         rax, rax
%%loop_n:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 KIND_NORMAL
    inc         rax
    jmp         %%loop_n
%%kind_s:
    xor         rax, rax
%%loop_s:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 KIND_S
    inc         rax
    jmp         %%loop_s
%%kind_s2:
    xor         rax, rax
%%loop_s2:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 KIND_S2
    inc         rax
    jmp         %%loop_s2
%%kind_s4:
    xor         rax, rax
%%loop_s4:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 KIND_S4
    inc         rax
    jmp         %%loop_s4
%%done:
    vzeroupper
    ret
%endmacro

MAKE_LEAF_BATCH tangent_x86_batch_leaf2, REG_LEAF2_BODY, 2
MAKE_LEAF_BATCH tangent_x86_batch_leaf3, REG_LEAF3_BODY, 3
MAKE_LEAF_BATCH tangent_x86_batch_leaf4, REG_LEAF4_BODY, 4

; Load two permuted complex samples directly into one XMM. r10 addresses the
; permutation entries for the current output leaf and rdi is the original
; input. This folds the pre-shuffle into the fixed kernel.
%macro GATHER_PAIR 2
    mov         r11d, [r10 + (%2) * 4]
    vmovq       %1, [rdi + r11*8]
    mov         r11d, [r10 + (%2 + 1) * 4]
    vpinsrq     %1, %1, [rdi + r11*8], 1
%endmacro

%macro GATHER_LEAF2_BODY 1
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    REG_LEAF2_CORE %1
%endmacro

%macro GATHER_LEAF3_BODY 1
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    GATHER_PAIR xmm2, 4
    GATHER_PAIR xmm3, 6
    REG_LEAF3_CORE %1
%endmacro

%macro GATHER_LEAF4_BODY 1
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    GATHER_PAIR xmm2, 4
    GATHER_PAIR xmm3, 6
    GATHER_PAIR xmm4, 8
    GATHER_PAIR xmm5, 10
    GATHER_PAIR xmm6, 12
    GATHER_PAIR xmm7, 14
    REG_LEAF4_CORE %1
%endmacro

; Fixed 32-point top-level kernel.  Keep the complete 16-point even child in
; registers, retain only the two 8-point children in scratch, and combine
; directly. This mirrors FFmpeg's FFT32 register lifetime without importing
; its code.
%macro FFT32_NORMAL_CHUNK 8
    vmovups     ymm12, [r9 + %5]
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    APPLY_FACTOR %3, %4, ymm13, ymm12, ymm10, ymm11
    FINISH_UNSCALED %1, %2, %3, %4, ymm8, ymm9, ymm10, ymm11
    vmovups     [rsi + %6], %3
    vmovups     [rsi + %7], ymm8
    vmovups     [rsi + %8], %4
    vmovups     [rsi + %8 + 64], ymm9
%endmacro

; input, output, permutation, fixed-leaf tables, level-5 scaled factor.
global tangent_x86_gather_fft32_normal
tangent_x86_gather_fft32_normal:
    mov         r9, r8

    lea         r10, [rdx + 16*4]
    lea         r8, [rsi + 16*8]
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    GATHER_PAIR xmm2, 4
    GATHER_PAIR xmm3, 6
    REG_LEAF3_CORE KIND_S

    lea         r10, [rdx + 24*4]
    lea         r8, [rsi + 24*8]
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    GATHER_PAIR xmm2, 4
    GATHER_PAIR xmm3, 6
    REG_LEAF3_CORE KIND_S

    mov         r10, rdx
    mov         r8, rsi
    GATHER_PAIR xmm0, 0
    GATHER_PAIR xmm1, 2
    GATHER_PAIR xmm2, 4
    GATHER_PAIR xmm3, 6
    GATHER_PAIR xmm4, 8
    GATHER_PAIR xmm5, 10
    GATHER_PAIR xmm6, 12
    GATHER_PAIR xmm7, 14
    REG_LEAF4_CALC KIND_NORMAL

    vmovups     ymm4, [rsi + 16*8]
    vmovups     ymm5, [rsi + 20*8]
    vmovups     ymm6, [rsi + 24*8]
    vmovups     ymm7, [rsi + 28*8]

    FFT32_NORMAL_CHUNK ymm0, ymm2, ymm4, ymm6, 0, 0, 8*8, 16*8
    FFT32_NORMAL_CHUNK ymm1, ymm3, ymm5, ymm7, 32, 4*8, 12*8, 20*8
    vzeroupper
    ret

; input, output, permutation, offsets, count, tables, kind=[stack].
%macro MAKE_GATHER_LEAF_BATCH 2
global %1 %+ _n
global %1 %+ _s
global %1 %+ _s2
global %1 %+ _s4
%1 %+ _n:
    mov         r11d, KIND_NORMAL
    jmp         %%entry
%1 %+ _s:
    mov         r11d, KIND_S
    jmp         %%entry
%1 %+ _s2:
    mov         r11d, KIND_S2
    jmp         %%entry
%1 %+ _s4:
    mov         r11d, KIND_S4
%%entry:
    push        rbx
    mov         rbx, rcx                    ; leaf offsets
    mov         rcx, r9                     ; packed tables
    mov         r9, r8                      ; leaf count
    cmp         r11d, KIND_S
    je          %%kind_s
    cmp         r11d, KIND_S2
    je          %%kind_s2
    cmp         r11d, KIND_S4
    je          %%kind_s4
    xor         rax, rax
%%loop_n:
    cmp         rax, r9
    jae         %%done
    mov         r10d, [rbx + rax*4]
    lea         r8, [rsi + r10*8]
    lea         r10, [rdx + r10*4]
    %2 KIND_NORMAL
    inc         rax
    jmp         %%loop_n
%%kind_s:
    xor         rax, rax
%%loop_s:
    cmp         rax, r9
    jae         %%done
    mov         r10d, [rbx + rax*4]
    lea         r8, [rsi + r10*8]
    lea         r10, [rdx + r10*4]
    %2 KIND_S
    inc         rax
    jmp         %%loop_s
%%kind_s2:
    xor         rax, rax
%%loop_s2:
    cmp         rax, r9
    jae         %%done
    mov         r10d, [rbx + rax*4]
    lea         r8, [rsi + r10*8]
    lea         r10, [rdx + r10*4]
    %2 KIND_S2
    inc         rax
    jmp         %%loop_s2
%%kind_s4:
    xor         rax, rax
%%loop_s4:
    cmp         rax, r9
    jae         %%done
    mov         r10d, [rbx + rax*4]
    lea         r8, [rsi + r10*8]
    lea         r10, [rdx + r10*4]
    %2 KIND_S4
    inc         rax
    jmp         %%loop_s4
%%done:
    vzeroupper
    pop         rbx
    ret
%endmacro

MAKE_GATHER_LEAF_BATCH tangent_x86_gather_leaf2, GATHER_LEAF2_BODY
MAKE_GATHER_LEAF_BATCH tangent_x86_gather_leaf3, GATHER_LEAF3_BODY
MAKE_GATHER_LEAF_BATCH tangent_x86_gather_leaf4, GATHER_LEAF4_BODY

; Small unscaled batches. Normal and S nodes share the same scaled complex
; factor; the tangent form changes the operation count, not the result.
; q1: rdi=data, rsi=offsets, rdx=node_count
global tangent_x86_batch_unscaled_q1
tangent_x86_batch_unscaled_q1:
    xor         rax, rax
.batch_u1_loop:
    cmp         rax, rdx
    jae         .batch_u1_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovq       xmm0, [r8]
    vmovq       xmm1, [r8 + 8]
    vmovq       xmm2, [r8 + 16]
    vmovq       xmm3, [r8 + 24]
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
    vmovq       [r8], xmm2
    vmovq       [r8 + 8], xmm4
    vmovq       [r8 + 16], xmm3
    vmovq       [r8 + 24], xmm5
    inc         rax
    jmp         .batch_u1_loop
.batch_u1_done:
    vzeroupper
    ret

; q2/q4: rdi=data, rsi=offsets, rdx=node_count, rcx=factor
%macro BATCH_UNSCALED 3
global %1
%1:
    vmovups     %2, [rcx]
    vmovsldup   %3, %2
    vmovshdup   %2, %2
    xor         rax, rax
%%loop:
    cmp         rax, rdx
    jae         %%done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovups     %4, [r8]
    vmovups     %5, [r8 + %6]
    vmovups     %7, [r8 + 2*%6]
    vmovups     %8, [r8 + 3*%6]
    APPLY_FACTOR %7, %8, %3, %2, %9, %10
    FINISH_UNSCALED %4, %5, %7, %8, %9, %10, %11, %12
    vmovups     [r8], %7
    vmovups     [r8 + %6], %9
    vmovups     [r8 + 2*%6], %8
    vmovups     [r8 + 3*%6], %10
    inc         rax
    jmp         %%loop
%%done:
    vzeroupper
    ret
%endmacro

; name, factor, real, u0, u1, z, zp, quarter bytes, temporaries
%macro MAKE_BATCH_U2 0
global tangent_x86_batch_unscaled_q2
tangent_x86_batch_unscaled_q2:
    vmovups     xmm12, [rcx]
    vmovsldup   xmm13, xmm12
    vmovshdup   xmm12, xmm12
    xor         rax, rax
.batch_u2_loop:
    cmp         rax, rdx
    jae         .batch_u2_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovups     xmm0, [r8]
    vmovups     xmm1, [r8 + 16]
    vmovups     xmm2, [r8 + 32]
    vmovups     xmm3, [r8 + 48]
    APPLY_FACTOR xmm2, xmm3, xmm13, xmm12, xmm6, xmm7
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
    vmovups     [r8], xmm2
    vmovups     [r8 + 16], xmm4
    vmovups     [r8 + 32], xmm3
    vmovups     [r8 + 48], xmm5
    inc         rax
    jmp         .batch_u2_loop
.batch_u2_done:
    vzeroupper
    ret
%endmacro
MAKE_BATCH_U2

global tangent_x86_batch_unscaled_q4
tangent_x86_batch_unscaled_q4:
    vmovups     ymm12, [rcx]
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    xor         rax, rax
.batch_u4_loop:
    cmp         rax, rdx
    jae         .batch_u4_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    vmovups     ymm0, [r8]
    vmovups     ymm1, [r8 + 32]
    vmovups     ymm2, [r8 + 64]
    vmovups     ymm3, [r8 + 96]
    APPLY_FACTOR ymm2, ymm3, ymm13, ymm12, ymm6, ymm7
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmovups     [r8], ymm2
    vmovups     [r8 + 32], ymm4
    vmovups     [r8 + 64], ymm3
    vmovups     [r8 + 96], ymm5
    inc         rax
    jmp         .batch_u4_loop
.batch_u4_done:
    vzeroupper
    ret

; S2 q1: rdi=data, rsi=offsets, rdx=count, rcx=low, r8=high
global tangent_x86_batch_s2_q1
tangent_x86_batch_s2_q1:
    vbroadcastss xmm12, [rcx]
    vbroadcastss xmm13, [r8]
    xor         rax, rax
.batch_s21_loop:
    cmp         rax, rdx
    jae         .batch_s21_done
    mov         r9d, [rsi + rax*4]
    lea         r9, [rdi + r9*8]
    vmovq       xmm0, [r9]
    vmovq       xmm1, [r9 + 8]
    vmovq       xmm2, [r9 + 16]
    vmovq       xmm3, [r9 + 24]
    vaddps      xmm4, xmm2, xmm3
    vsubps      xmm5, xmm2, xmm3
    vmulps      xmm4, xmm4, xmm12
    vmulps      xmm5, xmm5, xmm13
    vaddps      xmm2, xmm0, xmm4
    vsubps      xmm3, xmm0, xmm4
    vpermilps   xmm5, xmm5, 0xb1
    vxorps      xmm6, xmm5, [sign_odd]
    vxorps      xmm7, xmm5, [sign_even]
    vaddps      xmm4, xmm1, xmm6
    vaddps      xmm5, xmm1, xmm7
    vmovq       [r9], xmm2
    vmovq       [r9 + 8], xmm4
    vmovq       [r9 + 16], xmm3
    vmovq       [r9 + 24], xmm5
    inc         rax
    jmp         .batch_s21_loop
.batch_s21_done:
    vzeroupper
    ret

; S2 q2/q4 generators.
; args: data, offsets, count, factor, low, high
%macro BATCH_S2_Q 3
global %1
%1:
    vmovups     %2, [rcx]
    vmovsldup   %3, %2
    vmovshdup   %2, %2
%if %3 = xmm13
    vmovq       xmm14, [r8]
    vpermilps   xmm14, xmm14, 0x50
    vmovq       xmm15, [r9]
    vpermilps   xmm15, xmm15, 0x50
%else
    DUP4        ymm14, [r8], 6, 7
    DUP4        ymm15, [r9], 6, 7
%endif
    xor         rax, rax
%%loop:
    cmp         rax, rdx
    jae         %%done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    vmovups     %4, [r10]
    vmovups     %5, [r10 + %8]
    vmovups     %6, [r10 + 2*%8]
    vmovups     %7, [r10 + 3*%8]
    APPLY_FACTOR %6, %7, %3, %2, %9, %10
    vaddps      %9, %6, %7
    vsubps      %10, %6, %7
    vmulps      %9, %9, %11
    vmulps      %10, %10, %12
    vaddps      %6, %4, %9
    vsubps      %7, %4, %9
    vpermilps   %10, %10, 0xb1
    vxorps      %9, %10, [sign_odd]
    vxorps      %4, %10, [sign_even]
    vaddps      %9, %5, %9
    vaddps      %10, %5, %4
    vmovups     [r10], %6
    vmovups     [r10 + %8], %9
    vmovups     [r10 + 2*%8], %7
    vmovups     [r10 + 3*%8], %10
    inc         rax
    jmp         %%loop
%%done:
    vzeroupper
    ret
%endmacro

; These are written explicitly because NASM register-size token comparisons
; are deliberately avoided in the hot code.
global tangent_x86_batch_s2_q2
tangent_x86_batch_s2_q2:
    vmovups     xmm12, [rcx]
    vmovsldup   xmm13, xmm12
    vmovshdup   xmm12, xmm12
    vmovq       xmm14, [r8]
    vpermilps   xmm14, xmm14, 0x50
    vmovq       xmm15, [r9]
    vpermilps   xmm15, xmm15, 0x50
    xor         rax, rax
.batch_s22_loop:
    cmp         rax, rdx
    jae         .batch_s22_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    vmovups     xmm0, [r10]
    vmovups     xmm1, [r10 + 16]
    vmovups     xmm2, [r10 + 32]
    vmovups     xmm3, [r10 + 48]
    APPLY_FACTOR xmm2, xmm3, xmm13, xmm12, xmm6, xmm7
    vaddps      xmm4, xmm2, xmm3
    vsubps      xmm5, xmm2, xmm3
    vmulps      xmm4, xmm4, xmm14
    vmulps      xmm5, xmm5, xmm15
    vaddps      xmm2, xmm0, xmm4
    vsubps      xmm3, xmm0, xmm4
    vpermilps   xmm5, xmm5, 0xb1
    vxorps      xmm6, xmm5, [sign_odd]
    vxorps      xmm7, xmm5, [sign_even]
    vaddps      xmm4, xmm1, xmm6
    vaddps      xmm5, xmm1, xmm7
    vmovups     [r10], xmm2
    vmovups     [r10 + 16], xmm4
    vmovups     [r10 + 32], xmm3
    vmovups     [r10 + 48], xmm5
    inc         rax
    jmp         .batch_s22_loop
.batch_s22_done:
    vzeroupper
    ret

global tangent_x86_batch_s2_q4
tangent_x86_batch_s2_q4:
    vmovups     ymm12, [rcx]
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    DUP4        ymm14, [r8], 6, 7
    DUP4        ymm15, [r9], 6, 7
    xor         rax, rax
.batch_s24_loop:
    cmp         rax, rdx
    jae         .batch_s24_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    vmovups     ymm0, [r10]
    vmovups     ymm1, [r10 + 32]
    vmovups     ymm2, [r10 + 64]
    vmovups     ymm3, [r10 + 96]
    APPLY_FACTOR ymm2, ymm3, ymm13, ymm12, ymm6, ymm7
    vaddps      ymm4, ymm2, ymm3
    vsubps      ymm5, ymm2, ymm3
    vmulps      ymm4, ymm4, ymm14
    vmulps      ymm5, ymm5, ymm15
    vaddps      ymm2, ymm0, ymm4
    vsubps      ymm3, ymm0, ymm4
    vpermilps   ymm5, ymm5, 0xb1
    vxorps      ymm6, ymm5, [sign_odd]
    vxorps      ymm7, ymm5, [sign_even]
    vaddps      ymm4, ymm1, ymm6
    vaddps      ymm5, ymm1, ymm7
    vmovups     [r10], ymm2
    vmovups     [r10 + 32], ymm4
    vmovups     [r10 + 64], ymm3
    vmovups     [r10 + 96], ymm5
    inc         rax
    jmp         .batch_s24_loop
.batch_s24_done:
    vzeroupper
    ret

; S4 q1: data, offsets, count, scale0, scale1, scale2, scale3
global tangent_x86_batch_s4_q1
tangent_x86_batch_s4_q1:
    vbroadcastss xmm12, [rcx]
    vbroadcastss xmm13, [r8]
    vbroadcastss xmm14, [r9]
    mov         r10, [rsp + 8]
    vbroadcastss xmm15, [r10]
    xor         rax, rax
.batch_s41_loop:
    cmp         rax, rdx
    jae         .batch_s41_done
    mov         r11d, [rsi + rax*4]
    lea         r11, [rdi + r11*8]
    vmovq       xmm0, [r11]
    vmovq       xmm1, [r11 + 8]
    vmovq       xmm2, [r11 + 16]
    vmovq       xmm3, [r11 + 24]
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
    vmulps      xmm2, xmm2, xmm12
    vmulps      xmm4, xmm4, xmm13
    vmulps      xmm3, xmm3, xmm14
    vmulps      xmm5, xmm5, xmm15
    vmovq       [r11], xmm2
    vmovq       [r11 + 8], xmm4
    vmovq       [r11 + 16], xmm3
    vmovq       [r11 + 24], xmm5
    inc         rax
    jmp         .batch_s41_loop
.batch_s41_done:
    vzeroupper
    ret

; S4 q2/q4: data, offsets, count, factor, scale0, scale1,
;           scale2=[rsp+8], scale3=[rsp+16]
global tangent_x86_batch_s4_q2
tangent_x86_batch_s4_q2:
    vmovups     xmm10, [rcx]
    vmovsldup   xmm11, xmm10
    vmovshdup   xmm10, xmm10
    vmovq       xmm12, [r8]
    vpermilps   xmm12, xmm12, 0x50
    vmovq       xmm13, [r9]
    vpermilps   xmm13, xmm13, 0x50
    mov         r10, [rsp + 8]
    vmovq       xmm14, [r10]
    vpermilps   xmm14, xmm14, 0x50
    mov         r10, [rsp + 16]
    vmovq       xmm15, [r10]
    vpermilps   xmm15, xmm15, 0x50
    xor         rax, rax
.batch_s42_loop:
    cmp         rax, rdx
    jae         .batch_s42_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    vmovups     xmm0, [r10]
    vmovups     xmm1, [r10 + 16]
    vmovups     xmm2, [r10 + 32]
    vmovups     xmm3, [r10 + 48]
    APPLY_FACTOR xmm2, xmm3, xmm11, xmm10, xmm6, xmm7
    FINISH_UNSCALED xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7
    vmulps      xmm2, xmm2, xmm12
    vmulps      xmm4, xmm4, xmm13
    vmulps      xmm3, xmm3, xmm14
    vmulps      xmm5, xmm5, xmm15
    vmovups     [r10], xmm2
    vmovups     [r10 + 16], xmm4
    vmovups     [r10 + 32], xmm3
    vmovups     [r10 + 48], xmm5
    inc         rax
    jmp         .batch_s42_loop
.batch_s42_done:
    vzeroupper
    ret

global tangent_x86_batch_s4_q4
tangent_x86_batch_s4_q4:
    vmovups     ymm10, [rcx]
    vmovsldup   ymm11, ymm10
    vmovshdup   ymm10, ymm10
    DUP4        ymm12, [r8], 6, 7
    DUP4        ymm13, [r9], 6, 7
    mov         r10, [rsp + 8]
    DUP4        ymm14, [r10], 6, 7
    mov         r10, [rsp + 16]
    DUP4        ymm15, [r10], 6, 7
    xor         rax, rax
.batch_s44_loop:
    cmp         rax, rdx
    jae         .batch_s44_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    vmovups     ymm0, [r10]
    vmovups     ymm1, [r10 + 32]
    vmovups     ymm2, [r10 + 64]
    vmovups     ymm3, [r10 + 96]
    APPLY_FACTOR ymm2, ymm3, ymm11, ymm10, ymm6, ymm7
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmulps      ymm2, ymm2, ymm12
    vmulps      ymm4, ymm4, ymm13
    vmulps      ymm3, ymm3, ymm14
    vmulps      ymm5, ymm5, ymm15
    vmovups     [r10], ymm2
    vmovups     [r10 + 32], ymm4
    vmovups     [r10 + 64], ymm3
    vmovups     [r10 + 96], ymm5
    inc         rax
    jmp         .batch_s44_loop
.batch_s44_done:
    vzeroupper
    ret

; Batched arbitrary-size unscaled nodes.
; data, offsets, node_count, quarter, factor
global tangent_x86_batch_unscaled_qn
tangent_x86_batch_unscaled_qn:
    push        r12
    push        r13
    mov         r12, rdx                    ; node count
    shl         rcx, 3
    mov         r13, rcx                    ; quarter bytes
    xor         rax, rax
.batch_un_loop_node:
    cmp         rax, r12
    jae         .batch_un_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]          ; stream 0
    lea         r11, [r10 + r13]            ; stream 1
    lea         rdx, [r11 + r13]            ; stream 2
    lea         rcx, [rdx + r13]            ; stream 3
    xor         r9, r9
.batch_un_loop_k:
    vmovups     ymm0, [r10 + r9]
    vmovups     ymm1, [r11 + r9]
    vmovups     ymm2, [rdx + r9]
    vmovups     ymm3, [rcx + r9]
    vmovups     ymm12, [r8 + r9]
    vmovsldup   ymm13, ymm12
    vmovshdup   ymm12, ymm12
    APPLY_FACTOR ymm2, ymm3, ymm13, ymm12, ymm6, ymm7
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmovups     [r10 + r9], ymm2
    vmovups     [r11 + r9], ymm4
    vmovups     [rdx + r9], ymm3
    vmovups     [rcx + r9], ymm5
    add         r9, 32
    cmp         r9, r13
    jb          .batch_un_loop_k
    inc         rax
    jmp         .batch_un_loop_node
.batch_un_done:
    vzeroupper
    pop         r13
    pop         r12
    ret

; Batched arbitrary-size tangent S nodes.
; data, offsets, node_count, quarter, tangent factor
global tangent_x86_batch_tangent_qn
tangent_x86_batch_tangent_qn:
    push        r12
    push        r13
    push        r14
    mov         r12, rdx
    shl         rcx, 3
    mov         r13, rcx
    mov         r14, rcx
    shr         r14, 1                       ; N/8 boundary in bytes
    xor         rax, rax
.batch_tn_loop_node:
    cmp         rax, r12
    jae         .batch_tn_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    lea         r11, [r10 + r13]
    lea         rdx, [r11 + r13]
    lea         rcx, [rdx + r13]
    xor         r9, r9
.batch_tn_loop_k:
    vmovups     ymm0, [r10 + r9]
    vmovups     ymm1, [r11 + r9]
    vmovups     ymm2, [rdx + r9]
    vmovups     ymm3, [rcx + r9]
    vmovups     ymm12, [r8 + r9]
    APPLY_TANGENT ymm2, ymm3, ymm12, r9, r14, ymm6, ymm7, ymm13, ymm15
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmovups     [r10 + r9], ymm2
    vmovups     [r11 + r9], ymm4
    vmovups     [rdx + r9], ymm3
    vmovups     [rcx + r9], ymm5
    add         r9, 32
    cmp         r9, r13
    jb          .batch_tn_loop_k
    inc         rax
    jmp         .batch_tn_loop_node
.batch_tn_done:
    vzeroupper
    pop         r14
    pop         r13
    pop         r12
    ret

; Batched arbitrary-size S2 nodes.
; data, offsets, node_count, quarter, factor, low, high=[rsp+8]
global tangent_x86_batch_s2_qn
tangent_x86_batch_s2_qn:
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdx
    shl         rcx, 3
    mov         r13, rcx                    ; quarter bytes
    mov         rbx, rcx
    shr         rbx, 1                      ; N/8 boundary in bytes
    mov         r14, r8                     ; factor
    mov         r15, r9                     ; low scales
    mov         r8, [rsp + 48]              ; high scales
    xor         rax, rax
.batch_s2n_loop_node:
    cmp         rax, r12
    jae         .batch_s2n_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    lea         r11, [r10 + r13]
    lea         rdx, [r11 + r13]
    lea         rcx, [rdx + r13]
    xor         r9, r9
.batch_s2n_loop_k:
    vmovups     ymm0, [r10 + r9]
    vmovups     ymm1, [r11 + r9]
    vmovups     ymm2, [rdx + r9]
    vmovups     ymm3, [rcx + r9]
    vmovups     ymm12, [r14 + r9]
    APPLY_TANGENT ymm2, ymm3, ymm12, r9, rbx, ymm6, ymm7, ymm13, ymm15
    vaddps      ymm4, ymm2, ymm3
    vsubps      ymm5, ymm2, ymm3
    vmovups     ymm14, [r15 + r9]
    vmovups     ymm15, [r8 + r9]
    vmulps      ymm4, ymm4, ymm14
    vmulps      ymm5, ymm5, ymm15
    vaddps      ymm2, ymm0, ymm4
    vsubps      ymm3, ymm0, ymm4
    vpermilps   ymm5, ymm5, 0xb1
    vxorps      ymm6, ymm5, [sign_odd]
    vxorps      ymm7, ymm5, [sign_even]
    vaddps      ymm4, ymm1, ymm6
    vaddps      ymm5, ymm1, ymm7
    vmovups     [r10 + r9], ymm2
    vmovups     [r11 + r9], ymm4
    vmovups     [rdx + r9], ymm3
    vmovups     [rcx + r9], ymm5
    add         r9, 32
    cmp         r9, r13
    jb          .batch_s2n_loop_k
    inc         rax
    jmp         .batch_s2n_loop_node
.batch_s2n_done:
    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret

; Batched arbitrary-size S4 nodes.
; data, offsets, node_count, quarter, factor, scale0,
; scale1=[rsp+8], scale2=[rsp+16], scale3=[rsp+24]
global tangent_x86_batch_s4_qn
tangent_x86_batch_s4_qn:
    push        rbx
    push        rbp
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdx
    shl         rcx, 3
    mov         r13, rcx                    ; quarter bytes
    mov         rbp, rcx
    shr         rbp, 1                      ; N/8 boundary in bytes
    mov         r14, r8                     ; factor
    mov         r15, r9                     ; scale0
    mov         r8, [rsp + 56]              ; scale1
    mov         r9, [rsp + 64]              ; scale2
    mov         rbx, [rsp + 72]             ; scale3
    xor         rax, rax
.batch_s4n_loop_node:
    cmp         rax, r12
    jae         .batch_s4n_done
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    lea         r11, [r10 + r13]
    lea         rdx, [r11 + r13]
    lea         rcx, [rdx + r13]
    push        rax
    xor         rax, rax                    ; byte offset within each stream
.batch_s4n_loop_k:
    vmovups     ymm0, [r10 + rax]
    vmovups     ymm1, [r11 + rax]
    vmovups     ymm2, [rdx + rax]
    vmovups     ymm3, [rcx + rax]
    vmovups     ymm10, [r14 + rax]
    APPLY_TANGENT ymm2, ymm3, ymm10, rax, rbp, ymm6, ymm7, ymm11, ymm8
    FINISH_UNSCALED ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    vmovups     ymm12, [r15 + rax]
    vmovups     ymm13, [r8 + rax]
    vmovups     ymm14, [r9 + rax]
    vmovups     ymm15, [rbx + rax]
    vmulps      ymm2, ymm2, ymm12
    vmulps      ymm4, ymm4, ymm13
    vmulps      ymm3, ymm3, ymm14
    vmulps      ymm5, ymm5, ymm15
    vmovups     [r10 + rax], ymm2
    vmovups     [r11 + rax], ymm4
    vmovups     [rdx + rax], ymm3
    vmovups     [rcx + rax], ymm5
    add         rax, 32
    cmp         rax, r13
    jb          .batch_s4n_loop_k
    pop         rax
    inc         rax
    jmp         .batch_s4n_loop_node
.batch_s4n_done:
    vzeroupper
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbp
    pop         rbx
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
