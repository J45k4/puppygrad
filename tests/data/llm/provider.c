#include "puppygrad_llm.h"
#include <stdlib.h>
#include <string.h>
/* Independent C provider exercises the ABI, without Rust model implementation. */
typedef struct { uint32_t tokens[32]; size_t count; int mode; } State;
static unsigned live_count, free_count, callback_count;
static float temperature_seen;
static uint64_t seed_seen;
unsigned test_live_count(void) { return live_count; }
unsigned test_free_count(void) { return free_count; }
unsigned test_callback_count(void) { return callback_count; }
float test_temperature(void) { return temperature_seen; }
uint64_t test_seed(void) { return seed_seen; }
static void error_message(PupLlmError *error, const char *message) {
    size_t len=strlen(message); error->length=len<error->capacity?len:error->capacity;
    if(error->length) memcpy(error->data,message,error->length);
}
static void *build_model(const uint8_t *config,size_t len,PupLlmInfo *info,PupLlmError *error) {
    char json[4096]; if(len>=sizeof(json)) {error_message(error,"config too long");return NULL;}
    memcpy(json,config,len);json[len]=0;
    if(strstr(json,"build_fail")) {error_message(error,"deliberate build failure");return NULL;}
#ifdef TEST_EXPECT_THREADS
    const char *threads=strstr(json,"\"threads\":");
    if(!threads || atoi(threads+10)<1 || atoi(threads+10)>2) {error_message(error,"missing or invalid thread configuration");return NULL;}
#endif
    State *s=calloc(1,sizeof(State));if(!s)return NULL;
    if(strstr(json,"double_done")) s->mode=1;
    if(strstr(json,"no_done")) s->mode=2;
    if(strstr(json,"after_done")) s->mode=3;
    if(strstr(json,"infer_fail")) s->mode=4;
    if(strstr(json,"eos")) s->mode=5;
    if(strstr(json,"wrong_output")) s->mode=6;
    if(strstr(json,"unicode")) s->mode=7;
    *info=(PupLlmInfo){6,5,32};live_count++;return s;
}
static int32_t infer(void *state,const uint32_t *input,size_t count,const PupLlmGeneration *generation,const PupLlmCallbacks *callbacks,PupLlmError *error) {
    State *s=state;s->count=0;
    if(!input || !count || generation->max_new_tokens>32) {error_message(error,"bad request");if(callbacks->on_done)callbacks->on_done(callbacks->user,PUP_LLM_DONE_ERROR);return 1;}
    temperature_seen=generation->temperature;seed_seen=generation->seed;
    if(s->mode==4) {error_message(error,"deliberate infer failure");if(callbacks->on_done)callbacks->on_done(callbacks->user,PUP_LLM_DONE_ERROR);return 1;}
    size_t total=generation->max_new_tokens;
    if(s->mode==5 && total>0)total=1;
    for(size_t i=0;i<total;i++)s->tokens[s->count++]=s->mode==5?5:s->mode==7?1+(i%3):(generation->temperature>0?3+(generation->seed%2):1+(i%2));
    if(s->mode==3 && callbacks->on_done)callbacks->on_done(callbacks->user,PUP_LLM_DONE_LIMIT);
    if(callbacks->on_tokens && s->count) {callback_count++;callbacks->on_tokens(callbacks->user,s->tokens,s->count);}
    if(s->mode!=2 && s->mode!=3 && callbacks->on_done)callbacks->on_done(callbacks->user,s->mode==5&&total>0?PUP_LLM_DONE_EOS:PUP_LLM_DONE_LIMIT);
    if(s->mode==1 && callbacks->on_done)callbacks->on_done(callbacks->user,PUP_LLM_DONE_LIMIT);
    if(s->mode==6 && s->count)s->tokens[0]=4;
    return 0;
}
static int32_t read_output(void *state,uint32_t *out,size_t capacity,size_t *count,PupLlmError *error) {
    State *s=state;*count=s->count;if(!out && !capacity)return 0;
    if(capacity<s->count) {error_message(error,"output too small");return 1;}
    if(s->count)memcpy(out,s->tokens,s->count*sizeof(uint32_t));return 0;
}
static void free_model(void *state) {if(state){live_count--;free_count++;free(state);}}
#ifndef TEST_ABI_VERSION
#define TEST_ABI_VERSION PUP_LLM_ABI_VERSION
#endif
#ifndef TEST_TABLE_SIZE
#define TEST_TABLE_SIZE sizeof(PupLlmApi)
#endif
static const PupLlmApi api={TEST_ABI_VERSION,TEST_TABLE_SIZE,build_model,infer,read_output,free_model};
const PupLlmApi *get_llm_api(void) {return &api;}
