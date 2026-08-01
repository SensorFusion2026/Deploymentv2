#!/usr/bin/env python3
"""
reference_spectrogram.py — the known-correct "prepare" answer for the parity test.

Reproduces the EXACT recipe the rumble model3 was trained on, using the same
TensorFlow ops as the training repo (ELP-Rumble-Detector, create_tfrecords.py):

    audio z-score  ->  tf.signal.stft (periodic Hann, zero-pad to fft_length)
                   ->  slice bins [bin_start:bin_end]
                   ->  log(|stft| + log_eps)
                   ->  spec z-score

This is GROUND TRUTH. spectrogram.c must match THIS file's output, not the
deployment code. All numbers are read from config/pipeline.json so the reference
and the C implementation share one source.

Usage:
    python reference_spectrogram.py --synth --emit out/         # reproducible synthetic clip
    python reference_spectrogram.py --wav clip.wav --emit out/  # a real 4 kHz mono clip

Emits (raw little-endian float32):
    out/input_audio.f32   -> expected_samples values   (the clip the C also reads)
    out/golden_spec.f32   -> num_frames * num_bins values, row-major [frame][bin]
"""
import argparse
import json
import os
import sys
import numpy as np


def load_cfg(path):
    cfg = json.load(open(path))
    a, s = cfg["audio"], cfg["spectrogram"]
    return dict(
        sr=a["target_sr"],
        clip_len_s=a["clip_len_s"],
        audio_mean=a["audio_norm_mean"],
        audio_std=a["audio_norm_std"],
        frame_length=s["frame_length"],
        frame_step=s["frame_step"],
        fft_length=s["fft_length"],
        bin_start=s["bin_start"],
        bin_end=s["bin_end"],
        log_eps=s["log_eps"],
        spec_mean=s["spec_norm_mean"],
        spec_std=s["spec_norm_std"],
    )


def expected_samples(c):
    return int(round(c["sr"] * c["clip_len_s"]))


def compute_logspec(audio, c):
    """Run the exact training recipe. `audio` is float32, length expected_samples."""
    import tensorflow as tf

    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    audio = (audio - c["audio_mean"]) / c["audio_std"]

    stft = tf.signal.stft(
        audio,
        frame_length=c["frame_length"],
        frame_step=c["frame_step"],
        fft_length=c["fft_length"],
        window_fn=tf.signal.hann_window,  # periodic=True by default — must match C
        pad_end=False,
    )
    stft = stft[:, c["bin_start"]:c["bin_end"]]
    spec = tf.math.log(tf.abs(stft) + c["log_eps"])
    spec = (spec - c["spec_mean"]) / c["spec_std"]
    return spec.numpy().astype(np.float32)  # [frames, bins]


def synth_clip(c, seed=42):
    """A reproducible test clip with low-frequency (rumble-band) content."""
    n = expected_samples(c)
    t = np.arange(n) / c["sr"]
    rng = np.random.default_rng(seed)
    sig = (
        0.6 * np.sin(2 * np.pi * 20.0 * t)
        + 0.3 * np.sin(2 * np.pi * 35.0 * t)
        + 0.15 * np.sin(2 * np.pi * 62.0 * t)
        + 0.05 * rng.standard_normal(n)
    )
    return (sig / np.max(np.abs(sig))).astype(np.float32)


def load_wav(path, c):
    """Load a mono clip and pad/trim to expected_samples. Assumes it is at target_sr."""
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if sr != c["sr"]:
        sys.exit(f"clip is {sr} Hz; expected {c['sr']} Hz (resample upstream)")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    need = expected_samples(c)
    if len(a) < need:
        a = np.pad(a, (0, need - len(a)))
    return a[:need].astype(np.float32)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.normpath(os.path.join(here, "..", "..", "config", "pipeline.json"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_cfg)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synth", action="store_true", help="use a reproducible synthetic clip")
    src.add_argument("--wav", help="path to a 4 kHz mono WAV clip")
    ap.add_argument("--emit", help="output directory for input_audio.f32 + golden_spec.f32")
    args = ap.parse_args()

    c = load_cfg(args.config)
    audio = synth_clip(c) if args.synth else load_wav(args.wav, c)

    spec = compute_logspec(audio, c)
    frames = (expected_samples(c) - c["frame_length"]) // c["frame_step"] + 1
    bins = c["bin_end"] - c["bin_start"]
    assert spec.shape == (frames, bins), f"got {spec.shape}, expected {(frames, bins)}"

    print(f"input samples : {len(audio)}")
    print(f"spec shape    : {spec.shape}  (frames x bins)")
    print(f"spec min/max  : {spec.min():.5f} / {spec.max():.5f}")
    print(f"spec mean/std : {spec.mean():.5f} / {spec.std():.5f}")

    if args.emit:
        os.makedirs(args.emit, exist_ok=True)
        audio.astype("<f4").tofile(os.path.join(args.emit, "input_audio.f32"))
        spec.astype("<f4").tofile(os.path.join(args.emit, "golden_spec.f32"))
        # Emit the exact recipe params (sourced from pipeline.json) so the C
        # parity test builds the SAME SpecParams without hardcoding constants
        # or parsing JSON. The params travel with the golden they produced.
        fields = [
            ("sample_rate", c["sr"]), ("clip_len_s", c["clip_len_s"]),
            ("audio_mean", c["audio_mean"]), ("audio_std", c["audio_std"]),
            ("frame_length", c["frame_length"]), ("frame_step", c["frame_step"]),
            ("fft_length", c["fft_length"]), ("bin_start", c["bin_start"]),
            ("bin_end", c["bin_end"]), ("log_eps", c["log_eps"]),
            ("spec_mean", c["spec_mean"]), ("spec_std", c["spec_std"]),
        ]
        with open(os.path.join(args.emit, "spec_params.txt"), "w") as fh:
            for k, v in fields:
                fh.write(f"{k} {v!r}\n")
        print(f"wrote {args.emit}/input_audio.f32, golden_spec.f32, spec_params.txt")


if __name__ == "__main__":
    main()
