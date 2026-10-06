#ifndef PUPPYGRAD_IMAGE_H
#define PUPPYGRAD_IMAGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1 is synchronous. Config/request are borrowed UTF-8 JSON bytes.
 * infer completes its callbacks before returning; state calls are exclusive.
 * read_output(NULL,0,...) queries metadata. Copy requires capacity >= bytes.
 * Output is retained until the next infer or free_model. Format 1 = packed RGB8.
 * on_done fires once with the same status returned by infer. */
typedef struct { uint8_t *data; size_t capacity, length; } PupImageError;
typedef struct { uint32_t width,height,format,reserved; size_t stride,bytes; } PupImageInfo;
typedef struct {
    void *user;
    void (*on_progress)(void *,uint32_t,uint32_t);
    void (*on_done)(void *,int32_t);
} PupImageCallbacks;
typedef struct {
    uint32_t abi_version,struct_size;
    void *(*build_model)(const uint8_t *,size_t,PupImageError *);
    int32_t (*infer)(void *,const uint8_t *,size_t,const PupImageCallbacks *,PupImageError *);
    int32_t (*read_output)(void *,uint8_t *,size_t,PupImageInfo *,PupImageError *);
    void (*free_model)(void *);
} PupImageApi;
const PupImageApi *get_image_api(void);
#ifdef __cplusplus
}
#endif
#endif
