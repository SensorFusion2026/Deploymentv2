/*
 * resampler.c — linear-interpolation resample + int16->float32 conversion.
 *
 * Mirrors the reference deployment's np.interp downsample: build the output
 * sample positions on the input timeline and linearly interpolate between the
 * two nearest input samples. No anti-alias filter (matches the reference).
 */
#include "resampler.h"

#define INT16_TO_FLOAT (1.0f / 32768.0f)

size_t resampler_out_len(size_t in_frames, unsigned int in_rate, unsigned int out_rate) {
    if (in_frames == 0 || in_rate == 0 || out_rate == 0) return 0;
    /* same count np.interp would produce for a linspace over the clip duration */
    return (size_t)((double)in_frames * (double)out_rate / (double)in_rate);
}

size_t resampler_process(const int16_t *in, size_t in_frames, unsigned int in_rate,
                         float *out, size_t out_cap, unsigned int out_rate) {
    if (!in || !out || in_frames == 0 || in_rate == 0 || out_rate == 0) return 0;

    size_t out_len = resampler_out_len(in_frames, in_rate, out_rate);
    if (out_len == 0 || out_len > out_cap) return 0;

    /* Same-rate fast path: just convert int16 -> float. */
    if (in_rate == out_rate) {
        size_t n = (in_frames < out_cap) ? in_frames : out_cap;
        for (size_t i = 0; i < n; i++) out[i] = (float)in[i] * INT16_TO_FLOAT;
        return n;
    }

    /* Map each output index to a fractional position on the input timeline.
       Position of output i: i * (in_rate/out_rate). Interpolate in[k], in[k+1]. */
    const double step = (double)in_rate / (double)out_rate;
    for (size_t i = 0; i < out_len; i++) {
        double pos = (double)i * step;
        size_t k   = (size_t)pos;             /* left input index            */
        double frac = pos - (double)k;        /* distance to the right sample */

        double s0 = (double)in[k];
        double s1 = (k + 1 < in_frames) ? (double)in[k + 1] : s0;  /* clamp at end */
        double s  = s0 + (s1 - s0) * frac;    /* linear interpolation         */

        out[i] = (float)(s * INT16_TO_FLOAT);
    }
    return out_len;
}
