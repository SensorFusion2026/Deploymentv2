/*
 * spectrogram.h — prepared input for the ELP RUMBLE detector (model3).
 *
 * Reproduces the recipe the model was TRAINED on, taken from the training repo
 * (ELP-Rumble-Detector, data_creation/create_tfrecords.py) plus the model3
 * normalization stats — NOT from a deployment reconstruction. Any deviation
 * degrades accuracy silently, with no error. This is a contract, not a default.
 *
 * NOTE: rumble differs from gunshot on purpose (infrasound needs a long window
 * and a low-frequency slice). This module is rumble-specific.
 *
 * Recipe — steps fixed; numbers come from pipeline.json:
 *   1. input       : mono float32 PCM in [-1, 1], at sample_rate, clip_len_s long
 *   2. audio norm  : a = (pcm - audio_mean) / audio_std   (GLOBAL training stats)
 *   3. STFT        : PERIODIC Hann window of length frame_length, hop frame_step,
 *                    each frame ZERO-PADDED to fft_length (2048 > 2000), no end pad
 *   4. slice       : keep frequency bins [bin_start, bin_end)  -> num_bins
 *   5. log         : v = log(|stft| + log_eps)   (natural log; NOT log1p, NOT dB)
 *   6. spec norm   : z = (v - spec_mean) / spec_std        (GLOBAL training stats)
 *
 *   output: row-major [frame][bin] floats, mapping one-to-one onto the TFLite
 *   input tensor [1, frames, bins, 1]. Values are z-scores (centered near 0).
 */
#ifndef ELP_SPECTROGRAM_H
#define ELP_SPECTROGRAM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Numeric parameters, loaded from pipeline.json. The four *_mean / *_std values
 * are the model3 train-split statistics; they are DATA (they change per trained
 * model), which is why they live in config and are passed in here. The
 * algorithmic choices (Hann periodic, zero-pad to fft_length, log+eps, z-score)
 * live in spectrogram.c.
 */
typedef struct {
    int   sample_rate;   /* target_sr,        e.g. 4000   */
    float clip_len_s;    /* clip length,      e.g. 5.0    */
    float audio_mean;    /* audio z-score mean, e.g. -0.0139959 */
    float audio_std;     /* audio z-score std,  e.g.  0.0173435 */
    int   frame_length;  /* STFT window,      e.g. 2000   */
    int   frame_step;    /* STFT hop,         e.g. 32     */
    int   fft_length;    /* FFT size,         e.g. 2048 (frames zero-padded up) */
    int   bin_start;     /* first kept bin (inclusive), e.g. 2   */
    int   bin_end;       /* last kept bin (exclusive),  e.g. 100 */
    float log_eps;       /* log(|x| + eps),   e.g. 1e-10  */
    float spec_mean;     /* spec z-score mean, e.g. 2.7122834 */
    float spec_std;      /* spec z-score std,  e.g. 1.0036184 */
} SpecParams;

/*
 * Opaque handle. Holds the precomputed Hann window and the FFT plan so they are
 * built once at create time, never per clip. Callers never see inside it.
 */
typedef struct SpecCtx SpecCtx;

/* Derived dimensions — always computed from params, never hand-entered.   */
int    spec_num_frames(const SpecParams *p);      /* (samples - frame_length)/frame_step + 1 */
int    spec_num_bins(const SpecParams *p);        /* bin_end - bin_start                      */
int    spec_expected_samples(const SpecParams *p);/* lround(sample_rate * clip_len_s)         */
size_t spec_output_size(const SpecParams *p);     /* num_frames * num_bins                    */

/* Lifecycle. spec_create returns NULL on bad params or allocation failure. */
SpecCtx *spec_create(const SpecParams *p);
void     spec_destroy(SpecCtx *ctx);

/*
 * Compute the prepared (normalized log-magnitude, sliced) spectrogram for one clip.
 *   in         : in_samples float32 PCM samples, range [-1, 1]
 *   in_samples : MUST equal spec_expected_samples(p)
 *   out        : caller-allocated buffer of out_size floats, written
 *                row-major as out[frame * num_bins + bin]
 *   out_size   : MUST equal spec_output_size(p)
 *
 * Returns 0 on success, non-zero on a size mismatch or internal error.
 * Performs no heap allocation — safe to call in the capture/inference loop.
 */
int spec_process(SpecCtx *ctx,
                 const float *in, size_t in_samples,
                 float *out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* ELP_SPECTROGRAM_H */
