/*
 * spectrogram.c — rumble (model3) prepared-input implementation.
 *
 * Reproduces, in C, the exact recipe spectrogram.h documents (which is the
 * recipe ELP-Rumble-Detector/create_tfrecords.py used to train the model):
 *
 *   audio z-score -> periodic-Hann STFT (zero-pad frame to fft_length)
 *                 -> slice bins [bin_start, bin_end)
 *                 -> log(|stft| + log_eps) -> spec z-score
 *
 * PARITY is verified by runtime/parity/ against the TensorFlow reference.
 *
 * --------------------------------------------------------------------------
 * FFT SWAP POINT: the only place this file does frequency-domain math is the
 * single function fft_radix2() below. Everything else (windowing, padding,
 * magnitude, slicing, the two log/normalize steps) is recipe and must NOT be
 * delegated to a library, because library window/scaling conventions differ
 * from tf.signal.stft and would silently break parity. To use Essentia / FFTW /
 * KissFFT, replace ONLY fft_radix2 with that library's complex FFT of the
 * already-prepared 2048-point buffer. Nothing else changes.
 * --------------------------------------------------------------------------
 */
#include "spectrogram.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- derived dimensions (pure functions of the params) ------------------ */

int spec_expected_samples(const SpecParams *p) {
    return (int)lround((double)p->sample_rate * (double)p->clip_len_s);
}
int spec_num_frames(const SpecParams *p) {
    return (spec_expected_samples(p) - p->frame_length) / p->frame_step + 1;
}
int spec_num_bins(const SpecParams *p) {
    return p->bin_end - p->bin_start;
}
size_t spec_output_size(const SpecParams *p) {
    return (size_t)spec_num_frames(p) * (size_t)spec_num_bins(p);
}

/* ---- opaque context: precomputed window, FFT tables, scratch ------------ */

struct SpecCtx {
    SpecParams p;
    int    n_fft;          /* fft_length, power of two                       */
    int    n_samples;      /* expected input samples                         */
    int    n_frames;
    int    n_bins;
    double *hann;          /* [frame_length] periodic Hann window            */
    double *anorm;         /* [n_samples] audio-normalized scratch           */
    double *re, *im;       /* [n_fft] FFT scratch                            */
    int    *brev;          /* [n_fft] bit-reversal permutation               */
    double *wre, *wim;     /* [n_fft/2] forward-FFT twiddles                 */
};

static int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

static int ilog2(int x) { int b = 0; while (x > 1) { x >>= 1; b++; } return b; }

/* in-place iterative radix-2 DIT FFT (forward, e^{-i2pi kn/N}). SWAP POINT. */
static void fft_radix2(const SpecCtx *c, double *re, double *im) {
    const int n = c->n_fft;
    for (int i = 0; i < n; i++) {           /* bit-reversal permutation      */
        int j = c->brev[i];
        if (j > i) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;           /* twiddle stride                */
        for (int base = 0; base < n; base += len) {
            int k = 0;
            for (int j = 0; j < half; j++, k += step) {
                const double wr = c->wre[k], wi = c->wim[k];
                const double ur = re[base + j],        ui = im[base + j];
                const double vr = re[base + j + half], vi = im[base + j + half];
                const double tr = wr * vr - wi * vi;   /* t = w * v           */
                const double ti = wr * vi + wi * vr;
                re[base + j]        = ur + tr; im[base + j]        = ui + ti;
                re[base + j + half] = ur - tr; im[base + j + half] = ui - ti;
            }
        }
    }
}

/* ---- lifecycle ---------------------------------------------------------- */

SpecCtx *spec_create(const SpecParams *p) {
    if (!p) return NULL;
    if (p->frame_length <= 0 || p->frame_step <= 0) return NULL;
    if (!is_pow2(p->fft_length) || p->fft_length < p->frame_length) return NULL;
    if (p->bin_start < 0 || p->bin_end <= p->bin_start) return NULL;
    if (p->bin_end > p->fft_length / 2 + 1) return NULL;
    if (p->spec_std == 0.0f || p->audio_std == 0.0f) return NULL;

    SpecCtx *c = (SpecCtx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->p         = *p;
    c->n_fft     = p->fft_length;
    c->n_samples = spec_expected_samples(p);
    c->n_frames  = spec_num_frames(p);
    c->n_bins    = spec_num_bins(p);

    c->hann  = (double *)malloc(sizeof(double) * (size_t)p->frame_length);
    c->anorm = (double *)malloc(sizeof(double) * (size_t)c->n_samples);
    c->re    = (double *)malloc(sizeof(double) * (size_t)c->n_fft);
    c->im    = (double *)malloc(sizeof(double) * (size_t)c->n_fft);
    c->brev  = (int *)   malloc(sizeof(int)    * (size_t)c->n_fft);
    c->wre   = (double *)malloc(sizeof(double) * (size_t)(c->n_fft / 2));
    c->wim   = (double *)malloc(sizeof(double) * (size_t)(c->n_fft / 2));
    if (!c->hann || !c->anorm || !c->re || !c->im || !c->brev || !c->wre || !c->wim) {
        spec_destroy(c);
        return NULL;
    }

    /* periodic Hann (divisor = frame_length, matching tf.signal.hann_window) */
    for (int n = 0; n < p->frame_length; n++)
        c->hann[n] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)n / (double)p->frame_length);

    /* bit-reversal table */
    const int bits = ilog2(c->n_fft);
    for (int i = 0; i < c->n_fft; i++) {
        int x = i, r = 0;
        for (int b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
        c->brev[i] = r;
    }
    /* forward twiddles: e^{-i 2pi k / N} */
    for (int k = 0; k < c->n_fft / 2; k++) {
        c->wre[k] = cos(-2.0 * M_PI * (double)k / (double)c->n_fft);
        c->wim[k] = sin(-2.0 * M_PI * (double)k / (double)c->n_fft);
    }
    return c;
}

void spec_destroy(SpecCtx *c) {
    if (!c) return;
    free(c->hann); free(c->anorm); free(c->re); free(c->im);
    free(c->brev); free(c->wre);  free(c->wim);
    free(c);
}

/* ---- the hot path ------------------------------------------------------- */

int spec_process(SpecCtx *c, const float *in, size_t in_samples,
                 float *out, size_t out_size) {
    if (!c || !in || !out) return 1;
    if ((int)in_samples != c->n_samples) return 2;
    if (out_size != spec_output_size(&c->p)) return 3;

    const SpecParams *p = &c->p;

    /* step 1: audio z-score (global training stats) */
    for (int i = 0; i < c->n_samples; i++)
        c->anorm[i] = ((double)in[i] - p->audio_mean) / p->audio_std;

    /* steps 2-6, per frame */
    for (int f = 0; f < c->n_frames; f++) {
        const int start = f * p->frame_step;

        /* window the frame, zero-pad up to fft_length */
        for (int k = 0; k < p->frame_length; k++) {
            c->re[k] = c->anorm[start + k] * c->hann[k];
            c->im[k] = 0.0;
        }
        for (int k = p->frame_length; k < c->n_fft; k++) {
            c->re[k] = 0.0;
            c->im[k] = 0.0;
        }

        fft_radix2(c, c->re, c->im);   /* <- the only frequency-domain step  */

        /* slice bins, magnitude, log+eps, spec z-score */
        float *row = out + (size_t)f * (size_t)c->n_bins;
        for (int b = p->bin_start; b < p->bin_end; b++) {
            const double mag = sqrt(c->re[b] * c->re[b] + c->im[b] * c->im[b]);
            const double v   = log(mag + p->log_eps);
            row[b - p->bin_start] = (float)((v - p->spec_mean) / p->spec_std);
        }
    }
    return 0;
}
