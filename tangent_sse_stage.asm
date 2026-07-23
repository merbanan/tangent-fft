; 128-bit tangent FFT stage kernels for x86-64 System V.
;
; The C plan builder supplies homogeneous level/kind batches.  These kernels
; keep the node loop, butterfly loop, four split-radix stream pointers, and
; coefficient addressing in assembly.  SSE/SSE2 use the baseline expansion;
; SSE3 and later use duplicate-load and alternating-add instructions.

bits 64
default rel

section .rodata align=16
align 16
sse_sign_even: dd 0x80000000, 0, 0x80000000, 0
sse_sign_odd:  dd 0, 0x80000000, 0, 0x80000000
sse_sign_all:  times 4 dd 0x80000000
sse_sqrt_two:  times 4 dd 1.4142135623730951

section .text

; Load/store one or two complex values. q1 uses 64-bit accesses so the final
; node never reads beyond the work allocation.
%macro SSE_LOAD 3
%if %3
    movlps      %1, %2
%else
    movups      %1, %2
%endif
%endmacro

%macro SSE_STORE 3
%if %3
    movlps      %1, %2
%else
    movups      %1, %2
%endif
%endmacro

; z/zp in xmm2/xmm3, interleaved complex factor in xmm12.
%macro SSE_COMPLEX_PAIR 1
    movaps      xmm4, xmm2
    movaps      xmm5, xmm3
    shufps      xmm4, xmm4, 0xb1
    shufps      xmm5, xmm5, 0xb1
%if %1
    movsldup    xmm6, xmm12
    movshdup    xmm7, xmm12
%else
    movaps      xmm6, xmm12
    movaps      xmm7, xmm12
    shufps      xmm6, xmm6, 0xa0
    shufps      xmm7, xmm7, 0xf5
%endif
    mulps       xmm2, xmm6
    mulps       xmm3, xmm6
    mulps       xmm4, xmm7
    mulps       xmm5, xmm7
    xorps       xmm4, [sse_sign_even]
    xorps       xmm5, [sse_sign_odd]
    addps       xmm2, xmm4
    addps       xmm3, xmm5
%endmacro

; u0/u1 are xmm0/xmm1 and a/b are xmm2/xmm3.
; Results: o0=xmm2, o1=xmm4, o2=xmm3, o3=xmm5.
%macro SSE_FINISH 1
    movaps      xmm4, xmm2
    movaps      xmm5, xmm2
    addps       xmm4, xmm3
    subps       xmm5, xmm3
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm4
    subps       xmm3, xmm4
    shufps      xmm5, xmm5, 0xb1
%if %1
    movaps      xmm6, xmm5
    xorps       xmm6, [sse_sign_all]
    movaps      xmm4, xmm1
    addsubps    xmm4, xmm6
    movaps      xmm6, xmm1
    addsubps    xmm6, xmm5
    movaps      xmm5, xmm6
%else
    movaps      xmm4, xmm5
    movaps      xmm6, xmm5
    xorps       xmm4, [sse_sign_odd]
    xorps       xmm6, [sse_sign_even]
    addps       xmm4, xmm1
    addps       xmm6, xmm1
    movaps      xmm5, xmm6
%endif
%endmacro

; data, offsets, node_count, s4_kind
global tangent_sse_batch_base
tangent_sse_batch_base:
    xor         rax, rax
.base_loop:
    cmp         rax, rdx
    jae         .base_done
    mov         r8d, [rsi + rax*4]
    lea         r8, [rdi + r8*8]
    movlps      xmm0, [r8]
    movlps      xmm1, [r8 + 8]
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm1
    subps       xmm3, xmm1
    test        rcx, rcx
    jz          .base_store
    mulps       xmm3, [sse_sqrt_two]
.base_store:
    movlps      [r8], xmm2
    movlps      [r8 + 8], xmm3
    inc         rax
    jmp         .base_loop
.base_done:
    ret

%define SSE_LEAF_TABLE_STRIDE 160
%define SSE_LEAF_SCALED       (0 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_TANGENT      (1 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S2_LOW       (2 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S2_HIGH      (3 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S4_0         (4 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S4_1         (5 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S4_2         (6 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_S4_3         (7 * SSE_LEAF_TABLE_STRIDE)
%define SSE_LEAF_LEVEL(level) ((level) * 32)

%define SSE_KIND_NORMAL 0
%define SSE_KIND_S      1
%define SSE_KIND_S2     2
%define SSE_KIND_S4     3

; r8 points to a permuted fixed leaf and rcx to packed leaf tables.
%macro SSE_LEAF_BASE 2
    movlps      xmm0, [r8 + %1]
    movlps      xmm1, [r8 + %1 + 8]
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm1
    subps       xmm3, xmm1
%if %2 = SSE_KIND_S4
    mulps       xmm3, [sse_sqrt_two]
%endif
    movlps      [r8 + %1], xmm2
    movlps      [r8 + %1 + 8], xmm3
%endmacro

; One one- or two-complex leaf combine chunk.
; base bytes, quarter bytes, k bytes, level, kind, SSE3, q1.
%macro SSE_LEAF_COMBINE_CHUNK 7
    SSE_LOAD    xmm0, [r8 + %1 + %3], %7
    SSE_LOAD    xmm1, [r8 + %1 + %2 + %3], %7
    SSE_LOAD    xmm2, [r8 + %1 + 2*(%2) + %3], %7
    SSE_LOAD    xmm3, [r8 + %1 + 3*(%2) + %3], %7
%if %4 > 2
%if %5 = SSE_KIND_NORMAL
    SSE_LOAD    xmm12, [rcx + SSE_LEAF_SCALED + SSE_LEAF_LEVEL(%4) + %3], %7
%else
    SSE_LOAD    xmm12, [rcx + SSE_LEAF_TANGENT + SSE_LEAF_LEVEL(%4) + %3], %7
%endif
    SSE_COMPLEX_PAIR %6
%endif
%if %5 = SSE_KIND_S2
    movaps      xmm4, xmm2
    movaps      xmm5, xmm2
    addps       xmm4, xmm3
    subps       xmm5, xmm3
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S2_LOW + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm4, xmm6
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S2_HIGH + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm5, xmm6
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm4
    subps       xmm3, xmm4
    shufps      xmm5, xmm5, 0xb1
%if %6
    movaps      xmm6, xmm5
    xorps       xmm6, [sse_sign_all]
    movaps      xmm4, xmm1
    addsubps    xmm4, xmm6
    movaps      xmm6, xmm1
    addsubps    xmm6, xmm5
    movaps      xmm5, xmm6
%else
    movaps      xmm4, xmm5
    movaps      xmm6, xmm5
    xorps       xmm4, [sse_sign_odd]
    xorps       xmm6, [sse_sign_even]
    addps       xmm4, xmm1
    addps       xmm6, xmm1
    movaps      xmm5, xmm6
%endif
%else
    SSE_FINISH %6
%if %5 = SSE_KIND_S4
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S4_0 + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm2, xmm6
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S4_1 + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm4, xmm6
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S4_2 + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm3, xmm6
    SSE_LOAD    xmm6, [rcx + SSE_LEAF_S4_3 + SSE_LEAF_LEVEL(%4) + %3], %7
    mulps       xmm5, xmm6
%endif
%endif
    SSE_STORE   [r8 + %1 + %3], xmm2, %7
    SSE_STORE   [r8 + %1 + %2 + %3], xmm4, %7
    SSE_STORE   [r8 + %1 + 2*(%2) + %3], xmm3, %7
    SSE_STORE   [r8 + %1 + 3*(%2) + %3], xmm5, %7
%endmacro

%macro SSE_LEAF_COMBINE 4
%if %2 = 2
    SSE_LEAF_COMBINE_CHUNK %1, 8, 0, 2, %3, %4, 1
%elif %2 = 3
    SSE_LEAF_COMBINE_CHUNK %1, 16, 0, 3, %3, %4, 0
%else
    SSE_LEAF_COMBINE_CHUNK %1, 32, 0, 4, %3, %4, 0
    SSE_LEAF_COMBINE_CHUNK %1, 32, 16, 4, %3, %4, 0
%endif
%endmacro

%macro SSE_LEAF2_BODY 2
%if %1 = SSE_KIND_S2
    SSE_LEAF_BASE 0, SSE_KIND_S4
%else
    SSE_LEAF_BASE 0, SSE_KIND_NORMAL
%endif
    SSE_LEAF_COMBINE 0, 2, %1, %2
%endmacro

%macro SSE_LEAF3_BODY 2
%if %1 = SSE_KIND_S || %1 = SSE_KIND_S4
    SSE_LEAF_BASE 0, SSE_KIND_S4
%else
    SSE_LEAF_BASE 0, SSE_KIND_NORMAL
%endif
    SSE_LEAF_BASE 32, SSE_KIND_NORMAL
    SSE_LEAF_BASE 48, SSE_KIND_NORMAL
%if %1 = SSE_KIND_NORMAL
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_NORMAL, %2
%elif %1 = SSE_KIND_S
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S2, %2
%elif %1 = SSE_KIND_S2
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S4, %2
%else
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S2, %2
%endif
    SSE_LEAF_COMBINE 0, 3, %1, %2
%endmacro

%macro SSE_LEAF4_BODY 2
%if %1 = SSE_KIND_S2
    SSE_LEAF_BASE 0, SSE_KIND_S4
%else
    SSE_LEAF_BASE 0, SSE_KIND_NORMAL
%endif
    SSE_LEAF_BASE 32, SSE_KIND_NORMAL
    SSE_LEAF_BASE 48, SSE_KIND_NORMAL
    SSE_LEAF_BASE 64, SSE_KIND_NORMAL
    SSE_LEAF_BASE 96, SSE_KIND_NORMAL
%if %1 = SSE_KIND_NORMAL
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_NORMAL, %2
%elif %1 = SSE_KIND_S
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S4, %2
%elif %1 = SSE_KIND_S2
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S2, %2
%else
    SSE_LEAF_COMBINE 0, 2, SSE_KIND_S4, %2
%endif
    SSE_LEAF_COMBINE 64, 2, SSE_KIND_S, %2
    SSE_LEAF_COMBINE 96, 2, SSE_KIND_S, %2
%if %1 = SSE_KIND_NORMAL
    SSE_LEAF_COMBINE 0, 3, SSE_KIND_NORMAL, %2
%elif %1 = SSE_KIND_S
    SSE_LEAF_COMBINE 0, 3, SSE_KIND_S2, %2
%elif %1 = SSE_KIND_S2
    SSE_LEAF_COMBINE 0, 3, SSE_KIND_S4, %2
%else
    SSE_LEAF_COMBINE 0, 3, SSE_KIND_S2, %2
%endif
    SSE_LEAF_COMBINE 0, 4, %1, %2
%endmacro

; data, offsets, leaf_count, packed tables, kind
%macro DEFINE_SSE_LEAF_BATCH 3
global %1
%1:
    cmp         r8d, SSE_KIND_S
    je          %%kind_s
    cmp         r8d, SSE_KIND_S2
    je          %%kind_s2
    cmp         r8d, SSE_KIND_S4
    je          %%kind_s4
    xor         rax, rax
%%loop_n:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 SSE_KIND_NORMAL, %3
    inc         rax
    jmp         %%loop_n
%%kind_s:
    xor         rax, rax
%%loop_s:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 SSE_KIND_S, %3
    inc         rax
    jmp         %%loop_s
%%kind_s2:
    xor         rax, rax
%%loop_s2:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 SSE_KIND_S2, %3
    inc         rax
    jmp         %%loop_s2
%%kind_s4:
    xor         rax, rax
%%loop_s4:
    cmp         rax, rdx
    jae         %%done
    mov         r9d, [rsi + rax*4]
    lea         r8, [rdi + r9*8]
    %2 SSE_KIND_S4, %3
    inc         rax
    jmp         %%loop_s4
%%done:
    ret
%endmacro

DEFINE_SSE_LEAF_BATCH tangent_sse_batch_leaf2, SSE_LEAF2_BODY, 0
DEFINE_SSE_LEAF_BATCH tangent_sse_batch_leaf3, SSE_LEAF3_BODY, 0
DEFINE_SSE_LEAF_BATCH tangent_sse_batch_leaf4, SSE_LEAF4_BODY, 0
DEFINE_SSE_LEAF_BATCH tangent_sse3_batch_leaf2, SSE_LEAF2_BODY, 1
DEFINE_SSE_LEAF_BATCH tangent_sse3_batch_leaf3, SSE_LEAF3_BODY, 1
DEFINE_SSE_LEAF_BATCH tangent_sse3_batch_leaf4, SSE_LEAF4_BODY, 1

; Load one permuted pair into scratch immediately before its fixed leaf runs.
; This removes the separate whole-array permutation pass and keeps the newly
; written input block hot for the leaf body.
%macro SSE_GATHER_PAIR_TO_LEAF 1
    mov         r10d, [r11 + (%1)*4]
    movlps      xmm0, [r12 + r10*8]
    mov         r10d, [r11 + (%1 + 1)*4]
    movhps      xmm0, [r12 + r10*8]
    movups      [r8 + (%1)*8], xmm0
%endmacro

%macro SSE_GATHER_LEAF_INPUT 1
%assign %%sample 0
%rep (1 << (%1 - 1))
    SSE_GATHER_PAIR_TO_LEAF %%sample
%assign %%sample %%sample + 2
%endrep
%endmacro

; input, output, permutation, offsets, count, tables, kind=[stack]
%macro DEFINE_SSE_GATHER_LEAF_BATCH 4
global %1
%1:
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdi
    mov         r13, rsi
    mov         r14, rdx
    mov         r15, rcx
    mov         rbx, r8
    mov         rcx, r9
    mov         r10d, [rsp + 48]
    cmp         r10d, SSE_KIND_S
    je          %%kind_s
    cmp         r10d, SSE_KIND_S2
    je          %%kind_s2
    cmp         r10d, SSE_KIND_S4
    je          %%kind_s4
    xor         rax, rax
%%loop_n:
    cmp         rax, rbx
    jae         %%done
    mov         r10d, [r15 + rax*4]
    lea         r8, [r13 + r10*8]
    lea         r11, [r14 + r10*4]
    SSE_GATHER_LEAF_INPUT %3
    %2 SSE_KIND_NORMAL, %4
    inc         rax
    jmp         %%loop_n
%%kind_s:
    xor         rax, rax
%%loop_s:
    cmp         rax, rbx
    jae         %%done
    mov         r10d, [r15 + rax*4]
    lea         r8, [r13 + r10*8]
    lea         r11, [r14 + r10*4]
    SSE_GATHER_LEAF_INPUT %3
    %2 SSE_KIND_S, %4
    inc         rax
    jmp         %%loop_s
%%kind_s2:
    xor         rax, rax
%%loop_s2:
    cmp         rax, rbx
    jae         %%done
    mov         r10d, [r15 + rax*4]
    lea         r8, [r13 + r10*8]
    lea         r11, [r14 + r10*4]
    SSE_GATHER_LEAF_INPUT %3
    %2 SSE_KIND_S2, %4
    inc         rax
    jmp         %%loop_s2
%%kind_s4:
    xor         rax, rax
%%loop_s4:
    cmp         rax, rbx
    jae         %%done
    mov         r10d, [r15 + rax*4]
    lea         r8, [r13 + r10*8]
    lea         r11, [r14 + r10*4]
    SSE_GATHER_LEAF_INPUT %3
    %2 SSE_KIND_S4, %4
    inc         rax
    jmp         %%loop_s4
%%done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
%endmacro

DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse_gather_leaf2, SSE_LEAF2_BODY, 2, 0
DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse_gather_leaf3, SSE_LEAF3_BODY, 3, 0
DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse_gather_leaf4, SSE_LEAF4_BODY, 4, 0
DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse3_gather_leaf2, SSE_LEAF2_BODY, 2, 1
DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse3_gather_leaf3, SSE_LEAF3_BODY, 3, 1
DEFINE_SSE_GATHER_LEAF_BATCH tangent_sse3_gather_leaf4, SSE_LEAF4_BODY, 4, 1

%macro SSE_NODE_POINTERS 0
    mov         r10d, [rsi + rax*4]
    lea         r10, [rdi + r10*8]
    lea         r8, [r10 + r13]
    lea         r9, [r8 + r13]
    lea         rdx, [r9 + r13]
%endmacro

%macro SSE_LOAD_STREAMS 1
    SSE_LOAD    xmm0, [r10 + r11], %1
    SSE_LOAD    xmm1, [r8 + r11], %1
    SSE_LOAD    xmm2, [r9 + r11], %1
    SSE_LOAD    xmm3, [rdx + r11], %1
    SSE_LOAD    xmm12, [r14 + r11], %1
%endmacro

%macro SSE_STORE_STREAMS 1
    SSE_STORE   [r10 + r11], xmm2, %1
    SSE_STORE   [r8 + r11], xmm4, %1
    SSE_STORE   [r9 + r11], xmm3, %1
    SSE_STORE   [rdx + r11], xmm5, %1
%endmacro

; data, offsets, node_count, quarter, complex factor
%macro DEFINE_SSE_UNSCALED 2
global %1
%1:
    push        r12
    push        r13
    push        r14
    mov         r12, rdx
    lea         r13, [rcx*8]
    mov         r14, r8
    xor         rax, rax
    cmp         r13, 8
    je          %%q1_outer
%%q2_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
%%q2_inner:
    SSE_LOAD_STREAMS 0
    SSE_COMPLEX_PAIR %2
    SSE_FINISH %2
    SSE_STORE_STREAMS 0
    add         r11, 16
    cmp         r11, r13
    jb          %%q2_inner
    inc         rax
    jmp         %%q2_outer
%%q1_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
    SSE_LOAD_STREAMS 1
    SSE_COMPLEX_PAIR %2
    SSE_FINISH %2
    SSE_STORE_STREAMS 1
    inc         rax
    jmp         %%q1_outer
%%done:
    pop         r14
    pop         r13
    pop         r12
    ret
%endmacro

DEFINE_SSE_UNSCALED tangent_sse_batch_unscaled, 0
DEFINE_SSE_UNSCALED tangent_sse3_batch_unscaled, 1

; S nodes use the tangent FFT's reduced products away from the vector that
; straddles the N/8 form change. The boundary vector uses the general complex
; expansion, preserving the exact mixed low/high coefficient representation.
%macro SSE_TANGENT_LOW_PAIR 1
    movaps      xmm4, xmm2
    movaps      xmm5, xmm3
    shufps      xmm4, xmm4, 0xb1
    shufps      xmm5, xmm5, 0xb1
    xorps       xmm4, [sse_sign_even]
    xorps       xmm5, [sse_sign_odd]
%if %1
    movshdup    xmm6, xmm12
%else
    movaps      xmm6, xmm12
    shufps      xmm6, xmm6, 0xf5
%endif
    mulps       xmm4, xmm6
    mulps       xmm5, xmm6
    addps       xmm2, xmm4
    addps       xmm3, xmm5
%endmacro

%macro SSE_TANGENT_HIGH_PAIR 1
    movaps      xmm4, xmm2
    movaps      xmm5, xmm3
    shufps      xmm4, xmm4, 0xb1
    shufps      xmm5, xmm5, 0xb1
    xorps       xmm4, [sse_sign_odd]
    xorps       xmm5, [sse_sign_even]
%if %1
    movsldup    xmm6, xmm12
%else
    movaps      xmm6, xmm12
    shufps      xmm6, xmm6, 0xa0
%endif
    mulps       xmm2, xmm6
    mulps       xmm3, xmm6
    addps       xmm2, xmm4
    addps       xmm3, xmm5
%endmacro

; data, offsets, node_count, quarter, packed tangent factor
%macro DEFINE_SSE_TANGENT 2
global %1
%1:
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdx
    lea         r13, [rcx*8]
    mov         r14, r8
    mov         r15, r13
    shr         r15, 1
    xor         rax, rax
    cmp         r13, 8
    je          %%q1_outer
%%q2_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
%%q2_inner:
    SSE_LOAD_STREAMS 0
    cmp         r11, r15
    jb          %%low
    ja          %%high
    SSE_COMPLEX_PAIR %2
    jmp         %%finish
%%low:
    SSE_TANGENT_LOW_PAIR %2
    jmp         %%finish
%%high:
    SSE_TANGENT_HIGH_PAIR %2
%%finish:
    SSE_FINISH %2
    SSE_STORE_STREAMS 0
    add         r11, 16
    cmp         r11, r13
    jb          %%q2_inner
    inc         rax
    jmp         %%q2_outer
%%q1_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
    SSE_LOAD_STREAMS 1
    SSE_COMPLEX_PAIR %2
    SSE_FINISH %2
    SSE_STORE_STREAMS 1
    inc         rax
    jmp         %%q1_outer
%%done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    ret
%endmacro

DEFINE_SSE_TANGENT tangent_sse_batch_tangent, 0
DEFINE_SSE_TANGENT tangent_sse3_batch_tangent, 1

; data, offsets, node_count, quarter, factor, low_scale, high_scale
%macro DEFINE_SSE_S2 2
global %1
%1:
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdx
    lea         r13, [rcx*8]
    mov         r14, r8
    mov         r15, r9
    mov         rbx, [rsp + 48]
    xor         rax, rax
    cmp         r13, 8
    je          %%q1_outer
%%q2_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
%%q2_inner:
    SSE_LOAD_STREAMS 0
    SSE_COMPLEX_PAIR %2
    movaps      xmm4, xmm2
    movaps      xmm5, xmm2
    addps       xmm4, xmm3
    subps       xmm5, xmm3
    mulps       xmm4, [r15 + r11]
    mulps       xmm5, [rbx + r11]
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm4
    subps       xmm3, xmm4
    shufps      xmm5, xmm5, 0xb1
%if %2
    movaps      xmm6, xmm5
    xorps       xmm6, [sse_sign_all]
    movaps      xmm4, xmm1
    addsubps    xmm4, xmm6
    movaps      xmm6, xmm1
    addsubps    xmm6, xmm5
    movaps      xmm5, xmm6
%else
    movaps      xmm4, xmm5
    movaps      xmm6, xmm5
    xorps       xmm4, [sse_sign_odd]
    xorps       xmm6, [sse_sign_even]
    addps       xmm4, xmm1
    addps       xmm6, xmm1
    movaps      xmm5, xmm6
%endif
    SSE_STORE_STREAMS 0
    add         r11, 16
    cmp         r11, r13
    jb          %%q2_inner
    inc         rax
    jmp         %%q2_outer
%%q1_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
    SSE_LOAD_STREAMS 1
    SSE_COMPLEX_PAIR %2
    movaps      xmm4, xmm2
    movaps      xmm5, xmm2
    addps       xmm4, xmm3
    subps       xmm5, xmm3
    movlps      xmm6, [r15]
    mulps       xmm4, xmm6
    movlps      xmm6, [rbx]
    mulps       xmm5, xmm6
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0
    addps       xmm2, xmm4
    subps       xmm3, xmm4
    shufps      xmm5, xmm5, 0xb1
%if %2
    movaps      xmm6, xmm5
    xorps       xmm6, [sse_sign_all]
    movaps      xmm4, xmm1
    addsubps    xmm4, xmm6
    movaps      xmm6, xmm1
    addsubps    xmm6, xmm5
    movaps      xmm5, xmm6
%else
    movaps      xmm4, xmm5
    movaps      xmm6, xmm5
    xorps       xmm4, [sse_sign_odd]
    xorps       xmm6, [sse_sign_even]
    addps       xmm4, xmm1
    addps       xmm6, xmm1
    movaps      xmm5, xmm6
%endif
    SSE_STORE_STREAMS 1
    inc         rax
    jmp         %%q1_outer
%%done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    ret
%endmacro

DEFINE_SSE_S2 tangent_sse_batch_s2, 0
DEFINE_SSE_S2 tangent_sse3_batch_s2, 1

; data, offsets, node_count, quarter, factor, scale0,
; scale1=[stack], scale2=[stack], scale3=[stack]
%macro DEFINE_SSE_S4 2
global %1
%1:
    push        rbp
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    mov         r12, rdx
    lea         r13, [rcx*8]
    mov         r14, r8
    mov         r15, r9
    mov         rbx, [rsp + 56]
    mov         rbp, [rsp + 64]
    mov         rcx, [rsp + 72]
    xor         rax, rax
    cmp         r13, 8
    je          %%q1_outer
%%q2_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
%%q2_inner:
    SSE_LOAD_STREAMS 0
    SSE_COMPLEX_PAIR %2
    SSE_FINISH %2
    mulps       xmm2, [r15 + r11]
    mulps       xmm4, [rbx + r11]
    mulps       xmm3, [rbp + r11]
    mulps       xmm5, [rcx + r11]
    SSE_STORE_STREAMS 0
    add         r11, 16
    cmp         r11, r13
    jb          %%q2_inner
    inc         rax
    jmp         %%q2_outer
%%q1_outer:
    cmp         rax, r12
    jae         %%done
    SSE_NODE_POINTERS
    xor         r11, r11
    SSE_LOAD_STREAMS 1
    SSE_COMPLEX_PAIR %2
    SSE_FINISH %2
    movlps      xmm6, [r15]
    mulps       xmm2, xmm6
    movlps      xmm6, [rbx]
    mulps       xmm4, xmm6
    movlps      xmm6, [rbp]
    mulps       xmm3, xmm6
    movlps      xmm6, [rcx]
    mulps       xmm5, xmm6
    SSE_STORE_STREAMS 1
    inc         rax
    jmp         %%q1_outer
%%done:
    pop         r15
    pop         r14
    pop         r13
    pop         r12
    pop         rbx
    pop         rbp
    ret
%endmacro

DEFINE_SSE_S4 tangent_sse_batch_s4, 0
DEFINE_SSE_S4 tangent_sse3_batch_s4, 1
