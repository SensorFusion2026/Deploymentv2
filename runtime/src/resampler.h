/*
 * resampler.h — convert a captured clip to the model's input format.
 *
 * Single job: take one clip of int16 samples at the capture rate and produce
 * float32 samples at the model's target rate (4 kHz), by LINEAR interpolation.
 *
 * "linear" is chosen to match the documented pipeline (pipeline.json
 * "resample": "linear", mirroring the reference deployment's np.interp). It does
 * NOT low-pass/anti-alias before downsampling — that matches the reference, and
 * is a known simplification, not an oversight. If the training preprocessing is
 * ever confirmed to resample differently, this module is the single place to
 * change it.
 *
 * Also does the int16 -> float32 conversion (divide by 32768) in the same pass,
 * since the spectrogram wants floats in [-1, 1].
 */
#ifndef ELP_RESAMPLER_H
#define ELP_RESAMPLER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of output samples produced from in_frames at in_rate -> out_rate. */
size_t resampler_out_len(size_t in_frames, unsigned int in_rate, unsigned int out_rate);

/*
 * Linearly resample + convert one clip.
 *   in        : in_frames int16 samples at in_rate (mono)
 *   out       : caller buffer, must hold resampler_out_len(...) float32 samples
 *   out_cap   : capacity of out (guards against a short buffer)
 * Returns the number of samples written (> 0), or 0 on error.
 */
size_t resampler_process(const int16_t *in, size_t in_frames, unsigned int in_rate,
                         float *out, size_t out_cap, unsigned int out_rate);

#ifdef __cplusplus
}
#endif

#endif /* ELP_RESAMPLER_H */
