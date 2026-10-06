#ifndef PUPPYGRAD_LLM_H
#define PUPPYGRAD_LLM_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define PUP_LLM_ABI_VERSION 1u
#define PUP_LLM_NO_EOS UINT32_MAX
#define PUP_LLM_DONE_LIMIT 0u
#define PUP_LLM_DONE_EOS 1u
#define PUP_LLM_DONE_ERROR 2u
/* All functions and callbacks are synchronous, on the calling thread in v1.
 * State is exclusively borrowed during calls; free_model releases it exactly once.
 * See docs/llm-runtime.md for ownership, callback ordering and error rules. */
typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t length;
} PupLlmError;
typedef struct {
    uint32_t vocab_size;
    uint32_t eos_token;
    uint64_t context_length;
} PupLlmInfo;
typedef struct {
    uint64_t max_new_tokens;
    float temperature;
    uint32_t reserved;
    uint64_t seed;
} PupLlmGeneration;
typedef struct {
    void (*on_tokens)(void *user, const uint32_t *tokens, size_t count);
    void (*on_done)(void *user, uint32_t reason);
    void *user;
} PupLlmCallbacks;
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    void *(*build_model)(const uint8_t *config_json, size_t config_len, PupLlmInfo *info, PupLlmError *error);
    int32_t (*infer)(void *state, const uint32_t *tokens, size_t count, const PupLlmGeneration *generation, const PupLlmCallbacks *callbacks, PupLlmError *error);
    int32_t (*read_output)(void *state, uint32_t *destination, size_t capacity, size_t *count, PupLlmError *error);
    void (*free_model)(void *state);
} PupLlmApi;
/* The table and function pointers remain valid until the library is unloaded. */
const PupLlmApi *get_llm_api(void);
#ifdef __cplusplus
}
#endif
#endif
