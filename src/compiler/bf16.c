/* Shared CPU/CUDA/HIP bit conversions. GPU loads widen only the values being
 * consumed; no FP32 checkpoint or device weight copy is allocated. */
#ifndef PUP_BF16_FN
#define PUP_BF16_FN static inline
#endif
PUP_BF16_FN float pup_bf16_load(unsigned short value) {
    union { unsigned int bits; float value; } x;
    /* Match BF16 widening semantics for signaling NaNs. Storage/view copies
     * bypass this conversion and preserve the original bits. */
    if ((value & 0x7fffU) > 0x7f80U) value |= 0x40U;
    x.bits = (unsigned int)value << 16;
    return x.value;
}
PUP_BF16_FN unsigned short pup_bf16_store(float value) {
    union { unsigned int bits; float value; } x;
    x.value = value;
    if ((x.bits & 0x7fffffffU) > 0x7f800000U)
        return (unsigned short)((x.bits >> 16) | 0x40U);
    return (unsigned short)((x.bits + 0x7fffU + ((x.bits >> 16) & 1U)) >> 16);
}
#undef PUP_BF16_FN
