/* Included verbatim in emitted C. Only libc, libm and POSIX threads are needed.
 * Each invocation owns its pool; no global state or host compute callbacks. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

/* Explicit vector values let generated microkernels retain accumulators across
 * the reduction loop. Generic builds use 128-bit vectors; AVX2 builds use 256.
 * memcpy handles panel loads without imposing an alignment requirement. */
#if defined(__AVX2__)
#define PUP_VEC_LANES 8
#else
#define PUP_VEC_LANES 4
#endif
typedef float pup_vec __attribute__((vector_size(PUP_VEC_LANES * sizeof(float))));

#define PUP_TILE_M 4
#define PUP_TILE_N (2 * PUP_VEC_LANES)

typedef struct pup_pool pup_pool;
typedef struct { pup_pool *pool; size_t id; } pup_worker;
struct pup_pool {
    size_t threads, created, epoch, pending;
    int stop;
    pthread_mutex_t mutex;
    pthread_cond_t work, done;
    pthread_t *handles;
    pup_worker *workers;
    void (*job)(const pup_pool *, size_t);
    const void *context;
};

static void *pup_worker_main(void *arg) {
    pup_worker *worker=arg;
    pup_pool *pool=worker->pool;
    size_t seen=0;
    pthread_mutex_lock(&pool->mutex);
    for(;;) {
        while(!pool->stop && seen==pool->epoch)
            pthread_cond_wait(&pool->work, &pool->mutex);
        if(pool->stop) break;
        seen=pool->epoch;
        pthread_mutex_unlock(&pool->mutex);
        pool->job(pool, worker->id);
        pthread_mutex_lock(&pool->mutex);
        if(--pool->pending==0) pthread_cond_signal(&pool->done);
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static void pup_pool_destroy(pup_pool *pool) {
    if(pool->threads<=1) return;
    pthread_mutex_lock(&pool->mutex);
    pool->stop=1;
    pthread_cond_broadcast(&pool->work);
    pthread_mutex_unlock(&pool->mutex);
    for(size_t i=0; i<pool->created; i++) pthread_join(pool->handles[i], NULL);
    free(pool->handles);
    free(pool->workers);
    pthread_cond_destroy(&pool->done);
    pthread_cond_destroy(&pool->work);
    pthread_mutex_destroy(&pool->mutex);
}

static int pup_pool_init(pup_pool *pool, size_t threads) {
    *pool=(pup_pool){.threads=threads};
    if(threads<=1) return 0;
    if(pthread_mutex_init(&pool->mutex, NULL)) return 3;
    if(pthread_cond_init(&pool->work, NULL)) {
        pthread_mutex_destroy(&pool->mutex); return 3;
    }
    if(pthread_cond_init(&pool->done, NULL)) {
        pthread_cond_destroy(&pool->work);
        pthread_mutex_destroy(&pool->mutex); return 3;
    }
    pool->handles=calloc(threads-1, sizeof(*pool->handles));
    pool->workers=calloc(threads-1, sizeof(*pool->workers));
    if(!pool->handles || !pool->workers) { pup_pool_destroy(pool); return 3; }
    for(size_t i=0; i<threads-1; i++) {
        pool->workers[i]=(pup_worker){pool, i+1};
        if(pthread_create(&pool->handles[i], NULL, pup_worker_main, &pool->workers[i])) {
            pup_pool_destroy(pool); return 3;
        }
        pool->created++;
    }
    return 0;
}

/* Each generated contraction owns its index expressions and epilogue.
 * The pool only distributes independent output tiles and waits for completion. */
static void pup_dispatch(pup_pool *pool, void (*job)(const pup_pool *,size_t), const void *context) {
    if(pool->threads>1) pthread_mutex_lock(&pool->mutex);
    pool->job=job; pool->context=context;
    if(pool->threads>1) {
        pool->pending=pool->threads-1;
        pool->epoch++;
        pthread_cond_broadcast(&pool->work);
        pthread_mutex_unlock(&pool->mutex);
    }
    job(pool, 0);
    if(pool->threads>1) {
        pthread_mutex_lock(&pool->mutex);
        while(pool->pending) pthread_cond_wait(&pool->done, &pool->mutex);
        pthread_mutex_unlock(&pool->mutex);
    }
}
