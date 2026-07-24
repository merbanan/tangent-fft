bits 64
default rel

section .text

; input, output, uint32 permutation, count
global analysis_vgather_permute
analysis_vgather_permute:
    xor         rax, rax
.loop:
    vpcmpeqd    ymm1, ymm1, ymm1
    vmovdqu     xmm2, [rdx + rax*4]
    vgatherdpd  ymm0, [rdi + xmm2*8], ymm1
    vmovdqu     [rsi + rax*8], ymm0
    add         rax, 4
    cmp         rax, rcx
    jb          .loop
    vzeroupper
    ret

global analysis_scalar_permute
analysis_scalar_permute:
    xor         rax, rax
.loop:
    mov         r8d, [rdx + rax*4]
    vmovq       xmm0, [rdi + r8*8]
    mov         r8d, [rdx + rax*4 + 4]
    vpinsrq     xmm0, xmm0, [rdi + r8*8], 1
    mov         r8d, [rdx + rax*4 + 8]
    vmovq       xmm1, [rdi + r8*8]
    mov         r8d, [rdx + rax*4 + 12]
    vpinsrq     xmm1, xmm1, [rdi + r8*8], 1
    vinsertf128 ymm0, ymm0, xmm1, 1
    vmovdqu     [rsi + rax*8], ymm0
    add         rax, 4
    cmp         rax, rcx
    jb          .loop
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
