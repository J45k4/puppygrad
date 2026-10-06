#ifndef PUPPYGRAD_PROFILE_H
#define PUPPYGRAD_PROFILE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Present in C emitted with --profile. Read-only JSON, valid for library lifetime.
 * version=1 describes input slots, output shapes, kernel IDs and stats offsets. */
const char *pup_profile_metadata(void);
/* Caller owns every buffer; each output and stats must be disjoint from all
 * other buffers (read-only inputs may alias). stats must have
 * at least metadata.stats_length uint64_t entries. It is reset on each call.
 * Header: invocation_ns, setup_ns, output_copy_ns, cleanup_ns.
 * Per kernel: calls, elapsed_ns, packing_ns, compute_ns.
 * Packing/compute are nested within elapsed; do not sum them again.
 * CPU wall time is CLOCK_MONOTONIC and includes worker synchronization.
 * 0=success, 1=allocation failure, 2=index bounds, 3=thread/pool failure,
 * 4=missing or undersized stats buffer. Failed calls may contain partial stats and partially written outputs.
 * Concurrent calls need disjoint outputs and stats; no mutable global counters. */
int pup_run_profiled(const void **inputs, void **outputs, size_t threads,
                     uint64_t *stats, size_t stats_length);
#ifdef __cplusplus
}
#endif
#endif
