/*
 * test_spectrogram_parity.c — build-time parity gate (workstation only).
 *
 * Proves spectrogram.c reproduces the training recipe. Reads the three files
 * emitted by reference_spectrogram.py --emit:
 *     <dir>/spec_params.txt   recipe params  (key value lines)
 *     <dir>/input_audio.f32   the clip        (expected_samples float32)
 *     <dir>/golden_spec.f32   ground truth    (num_frames * num_bins float32)
 * Runs spec_process on the same input and checks the result against golden.
 * Exits 0 on PASS, 1 on FAIL, 2 on a setup error.
 *
 * This is NOT compiled into the deployed binary — it runs on your workstation
 * before a build, never on the Jetson.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -o test_spectrogram_parity \
 *       test_spectrogram_parity.c ../src/spectrogram.c -I../src -lm
 * Run (after generating the files):
 *   python reference_spectrogram.py --synth --emit out/
 *   ./test_spectrogram_parity out/ [tolerance]      # default tolerance 1e-3
 */
#include "spectrogram.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* all 12 SpecParams fields must appear in spec_params.txt */
#define ALL_FIELDS 0xFFF

static int load_params(const char *path, SpecParams *p) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char key[64];
    double val;
    int seen = 0;
    while (fscanf(f, "%63s %lf", key, &val) == 2) {
        if      (!strcmp(key, "sample_rate"))  { p->sample_rate  = (int)val;   seen |= 1 << 0; }
        else if (!strcmp(key, "clip_len_s"))   { p->clip_len_s   = (float)val; seen |= 1 << 1; }
        else if (!strcmp(key, "audio_mean"))   { p->audio_mean   = (float)val; seen |= 1 << 2; }
        else if (!strcmp(key, "audio_std"))    { p->audio_std    = (float)val; seen |= 1 << 3; }
        else if (!strcmp(key, "frame_length")) { p->frame_length = (int)val;   seen |= 1 << 4; }
        else if (!strcmp(key, "frame_step"))   { p->frame_step   = (int)val;   seen |= 1 << 5; }
        else if (!strcmp(key, "fft_length"))   { p->fft_length   = (int)val;   seen |= 1 << 6; }
        else if (!strcmp(key, "bin_start"))    { p->bin_start    = (int)val;   seen |= 1 << 7; }
        else if (!strcmp(key, "bin_end"))      { p->bin_end      = (int)val;   seen |= 1 << 8; }
        else if (!strcmp(key, "log_eps"))      { p->log_eps      = (float)val; seen |= 1 << 9; }
        else if (!strcmp(key, "spec_mean"))    { p->spec_mean    = (float)val; seen |= 1 << 10; }
        else if (!strcmp(key, "spec_std"))     { p->spec_std     = (float)val; seen |= 1 << 11; }
    }
    fclose(f);
    if (seen != ALL_FIELDS) {
        fprintf(stderr, "%s: missing fields (saw mask 0x%x, want 0x%x)\n", path, seen, ALL_FIELDS);
        return -1;
    }
    return 0;
}

/* read a raw float32 file; if expect != 0, require exactly that many floats */
static float *read_f32(const char *path, size_t expect) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < 0) { fclose(f); return NULL; }
    size_t n = (size_t)bytes / sizeof(float);
    if (expect && n != expect) {
        fprintf(stderr, "%s: %zu floats, expected %zu\n", path, n, expect);
        fclose(f);
        return NULL;
    }
    float *buf = (float *)malloc((size_t)bytes);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "%s: short read\n", path);
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <emit_dir> [tolerance]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    double tol = (argc >= 3) ? atof(argv[2]) : 1e-3;

    char pp[1024], ip[1024], gp[1024];
    snprintf(pp, sizeof pp, "%s/spec_params.txt", dir);
    snprintf(ip, sizeof ip, "%s/input_audio.f32", dir);
    snprintf(gp, sizeof gp, "%s/golden_spec.f32", dir);

    SpecParams p;
    memset(&p, 0, sizeof p);
    if (load_params(pp, &p) != 0) return 2;

    SpecCtx *ctx = spec_create(&p);
    if (!ctx) { fprintf(stderr, "spec_create failed (bad params)\n"); return 2; }

    size_t n_in  = (size_t)spec_expected_samples(&p);
    size_t n_out = spec_output_size(&p);

    float *in     = read_f32(ip, n_in);
    float *golden = read_f32(gp, n_out);
    float *out    = (float *)malloc(sizeof(float) * n_out);
    int   ret     = 2;

    if (in && golden && out) {
        int rc = spec_process(ctx, in, n_in, out, n_out);
        if (rc != 0) {
            fprintf(stderr, "spec_process returned %d\n", rc);
        } else {
            const int nb = spec_num_bins(&p);
            double maxd = 0.0, sumd = 0.0;
            size_t over = 0, argmax = 0;
            for (size_t i = 0; i < n_out; i++) {
                double d = fabs((double)out[i] - (double)golden[i]);
                sumd += d;
                if (d > maxd) { maxd = d; argmax = i; }
                if (d > tol) over++;
            }
            printf("parity: %zu values  (%d frames x %d bins)\n",
                   n_out, spec_num_frames(&p), nb);
            printf("  max abs diff  : %.3e  (frame %zu, bin %zu)\n",
                   maxd, argmax / (size_t)nb, argmax % (size_t)nb);
            printf("  mean abs diff : %.3e\n", sumd / (double)n_out);
            printf("  over tol %.1e : %zu / %zu\n", tol, over, n_out);
            int pass = (maxd <= tol);
            printf("%s\n", pass ? "PASS" : "FAIL");
            ret = pass ? 0 : 1;
        }
    }

    spec_destroy(ctx);
    free(in);
    free(golden);
    free(out);
    return ret;
}
