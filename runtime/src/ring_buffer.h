/*
 * ring_buffer.h — gather a stream of audio frames into fixed-length clips.
 *
 * Single job: capture hands us small, arbitrary-sized chunks of samples; the
 * model wants one clip of an exact length (clip_len_s * capture_rate samples).
 * This buffer accumulates incoming samples and tells the caller each time a
 * full clip's worth is ready, then slides forward by a hop so clips can overlap
 * (or not). It stores int16 at the capture rate; resampling/float conversion
 * happen downstream.
 *
 * No audio processing here — just accumulation and windowing over time.
 */
#ifndef ELP_RING_BUFFER_H
#define ELP_RING_BUFFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RingBuffer RingBuffer;

/*
 * clip_frames : samples per emitted clip (e.g. capture_rate * clip_len_s,
 *               computed at the capture rate — NOT the model's 4 kHz)
 * hop_frames  : how far to advance after emitting a clip. Equal to clip_frames
 *               means back-to-back, non-overlapping clips; smaller means overlap.
 * Returns NULL on bad args or allocation failure.
 */
RingBuffer *rb_create(size_t clip_frames, size_t hop_frames);
void        rb_destroy(RingBuffer *rb);

size_t rb_clip_frames(const RingBuffer *rb);

/* Append `n` newly captured samples. Returns 0 on success, non-zero on error. */
int rb_push(RingBuffer *rb, const int16_t *samples, size_t n);

/*
 * If a full clip is ready, copy it into `out` (must hold rb_clip_frames values),
 * advance by hop_frames, and return 1. If not enough samples yet, return 0.
 * Call in a loop after each push to drain all clips a large push produced:
 *     rb_push(rb, buf, n);
 *     while (rb_pop_clip(rb, clip)) { ...process clip... }
 */
int rb_pop_clip(RingBuffer *rb, int16_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ELP_RING_BUFFER_H */
