/*
 * ring_buffer.c — accumulate captured samples into fixed-length clips.
 *
 * Implemented as a simple growable FIFO of int16 samples: rb_push appends,
 * rb_pop_clip emits a clip of clip_frames and drops hop_frames off the front.
 * Kept deliberately plain (no lock-free tricks) — the runtime is a single
 * capture-then-process loop, so one thread owns this buffer.
 */
#include "ring_buffer.h"

#include <stdlib.h>
#include <string.h>

struct RingBuffer {
    int16_t *buf;      /* pending samples, [head, head+count)                */
    size_t   cap;      /* allocated capacity in samples                      */
    size_t   head;     /* index of first pending sample                      */
    size_t   count;    /* number of pending samples                          */
    size_t   clip;     /* samples per emitted clip                           */
    size_t   hop;      /* advance after each emitted clip                    */
};

RingBuffer *rb_create(size_t clip_frames, size_t hop_frames) {
    if (clip_frames == 0 || hop_frames == 0 || hop_frames > clip_frames) return NULL;

    RingBuffer *rb = (RingBuffer *)calloc(1, sizeof(*rb));
    if (!rb) return NULL;

    /* start with room for two clips; grows if a push overshoots */
    rb->cap  = clip_frames * 2;
    rb->buf  = (int16_t *)malloc(sizeof(int16_t) * rb->cap);
    if (!rb->buf) { free(rb); return NULL; }
    rb->clip = clip_frames;
    rb->hop  = hop_frames;
    return rb;
}

void rb_destroy(RingBuffer *rb) {
    if (!rb) return;
    free(rb->buf);
    free(rb);
}

size_t rb_clip_frames(const RingBuffer *rb) { return rb ? rb->clip : 0; }

/* move pending samples back to index 0 so we can append contiguously */
static void rb_compact(RingBuffer *rb) {
    if (rb->head == 0) return;
    memmove(rb->buf, rb->buf + rb->head, rb->count * sizeof(int16_t));
    rb->head = 0;
}

int rb_push(RingBuffer *rb, const int16_t *samples, size_t n) {
    if (!rb || (!samples && n)) return 1;
    if (n == 0) return 0;

    /* ensure contiguous room for count + n at the front of the array */
    if (rb->head + rb->count + n > rb->cap) {
        rb_compact(rb);
        if (rb->count + n > rb->cap) {
            size_t new_cap = rb->cap;
            while (new_cap < rb->count + n) new_cap *= 2;
            int16_t *nb = (int16_t *)realloc(rb->buf, sizeof(int16_t) * new_cap);
            if (!nb) return 2;
            rb->buf = nb;
            rb->cap = new_cap;
        }
    }
    memcpy(rb->buf + rb->head + rb->count, samples, n * sizeof(int16_t));
    rb->count += n;
    return 0;
}

int rb_pop_clip(RingBuffer *rb, int16_t *out) {
    if (!rb || !out) return 0;
    if (rb->count < rb->clip) return 0;

    memcpy(out, rb->buf + rb->head, rb->clip * sizeof(int16_t));

    /* slide forward by hop */
    rb->head  += rb->hop;
    rb->count -= rb->hop;
    if (rb->count == 0) rb->head = 0;   /* reset when drained */
    return 1;
}
