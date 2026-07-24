bits 64
default rel

section .rodata align=32
align 32
half_vector:
    times 8 dd 0x3f000000

section .text
global h16_paired_h8_avx2

%macro SCALED_H8 1
    vmovups ymm0,  [%1 + 0 * 32]       ; A
    vmovups ymm1,  [%1 + 1 * 32]       ; B
    vmovups ymm2,  [%1 + 2 * 32]       ; C
    vmovups ymm3,  [%1 + 3 * 32]       ; D
    vmovups ymm4,  [%1 + 4 * 32]       ; E
    vmovups ymm5,  [%1 + 5 * 32]       ; F
    vmovups ymm6,  [%1 + 6 * 32]       ; G
    vmovups ymm7,  [%1 + 7 * 32]       ; H

    vaddps  ymm8,  ymm1, ymm2          ; r1 = B + C
    vaddps  ymm9,  ymm3, ymm7          ; r2 = D + H
    vaddps  ymm10, ymm5, ymm6          ; r3 = F + G
    vaddps  ymm11, ymm8, ymm9          ; r4 = r1 + r2
    vaddps  ymm12, ymm10, ymm4         ; r5 = r3 + E
    vaddps  ymm13, ymm11, ymm12        ; r6 = r4 + r5
    vmulps  ymm13, ymm13, [half_vector] ; t = r6 / 2
    vsubps  ymm14, ymm0, ymm13         ; m = A - t
    vaddps  ymm11, ymm14, ymm3         ; mD = m + D
    vaddps  ymm12, ymm14, ymm4         ; mE = m + E
    vaddps  ymm15, ymm14, ymm7         ; mH = m + H

    vaddps  ymm0, ymm0, ymm13          ; y0 = A + t
    vmovups [%1 + 0 * 32], ymm0

    vaddps  ymm13, ymm12, ymm2
    vaddps  ymm13, ymm13, ymm6         ; y1 = mE + C + G
    vmovups [%1 + 1 * 32], ymm13

    vaddps  ymm13, ymm12, ymm1
    vaddps  ymm13, ymm13, ymm5         ; y2 = mE + B + F
    vmovups [%1 + 2 * 32], ymm13

    vaddps  ymm13, ymm12, ymm9         ; y3 = mE + r2
    vmovups [%1 + 3 * 32], ymm13

    vaddps  ymm13, ymm11, ymm8         ; y4 = mD + r1
    vmovups [%1 + 4 * 32], ymm13

    vaddps  ymm13, ymm15, ymm2
    vaddps  ymm13, ymm13, ymm5         ; y5 = mH + C + F
    vmovups [%1 + 5 * 32], ymm13

    vaddps  ymm13, ymm15, ymm1
    vaddps  ymm13, ymm13, ymm6         ; y6 = mH + B + G
    vmovups [%1 + 6 * 32], ymm13

    vaddps  ymm13, ymm11, ymm10        ; y7 = mD + r3
    vmovups [%1 + 7 * 32], ymm13
%endmacro

; void h16_paired_h8_avx2(float real_values[8][8],
;                         float imaginary_values[8][8]);
;
; Each coordinate vector contains four independent H16 transforms:
;   [P(t0), Q(t0), P(t1), Q(t1), P(t2), Q(t2), P(t3), Q(t3)].
; Applying one scaled H8 DAG therefore computes both H16 branches for
; all four transforms without lane-crossing instructions.
h16_paired_h8_avx2:
    SCALED_H8 rdi
    SCALED_H8 rsi
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
