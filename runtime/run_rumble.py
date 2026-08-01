#!/usr/bin/env python3
"""End-to-end rumble detector on the Jetson (silence gate + labeled header)."""
import argparse, json, subprocess, time
import numpy as np
from tflite_runtime.interpreter import Interpreter

SILENCE_RMS = 0.005

def load_cfg(path):
    c = json.load(open(path)); a, s = c["audio"], c["spectrogram"]
    return dict(target_sr=a["target_sr"], clip_len_s=a["clip_len_s"],
        audio_mean=a["audio_norm_mean"], audio_std=a["audio_norm_std"],
        frame_length=s["frame_length"], frame_step=s["frame_step"],
        fft_length=s["fft_length"], bin_start=s["bin_start"], bin_end=s["bin_end"],
        log_eps=s["log_eps"], spec_mean=s["spec_norm_mean"], spec_std=s["spec_norm_std"],
        threshold=c["detector"]["threshold"])

def resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out: return x.astype(np.float32)
    n_out = int(len(x) * sr_out / sr_in)
    pos = np.arange(n_out) * (sr_in / sr_out)
    k = np.floor(pos).astype(int); frac = pos - k
    k1 = np.minimum(k + 1, len(x) - 1)
    return (x[k] + (x[k1] - x[k]) * frac).astype(np.float32)

def logspec(audio, c):
    a = (audio - c["audio_mean"]) / c["audio_std"]
    fl, fs, nfft = c["frame_length"], c["frame_step"], c["fft_length"]
    win = 0.5 - 0.5*np.cos(2*np.pi*np.arange(fl)/fl)
    n_frames = (len(a) - fl)//fs + 1
    out = np.empty((n_frames, c["bin_end"]-c["bin_start"]), np.float32)
    for i in range(n_frames):
        fr = a[i*fs:i*fs+fl]*win
        buf = np.zeros(nfft); buf[:fl] = fr
        mag = np.abs(np.fft.rfft(buf)[c["bin_start"]:c["bin_end"]])
        v = np.log(mag + c["log_eps"])
        out[i] = (v - c["spec_mean"]) / c["spec_std"]
    return out[np.newaxis, :, :, np.newaxis].astype(np.float32)

def score(interp, spec):
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    interp.set_tensor(inp["index"], spec); interp.invoke()
    return float(interp.get_tensor(outp["index"])[0][0])

def wav_bytes_to_float(raw, nch=1):
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
    return x[::nch] if nch > 1 else x

def read_wav(path):
    import wave
    w = wave.open(path,"rb"); sr=w.getframerate(); nch=w.getnchannels()
    raw=w.readframes(w.getnframes()); w.close()
    return wav_bytes_to_float(raw,nch), sr

def header(title, lines):
    bar = "=" * 52
    print(bar); print(f"  {title}")
    for ln in lines: print(f"  {ln}")
    print(bar)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline.json")
    ap.add_argument("--model",  default="rumble_model3_fp16.tflite")
    ap.add_argument("--wav"); ap.add_argument("--mic")
    ap.add_argument("--cap-sr", type=int, default=44100)
    ap.add_argument("--gate", type=float, default=SILENCE_RMS)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    c = load_cfg(args.config)
    interp = Interpreter(model_path=args.model); interp.allocate_tensors()
    need = int(round(c["target_sr"]*c["clip_len_s"]))
    def run_clip(audio, sr):
        rms = float(np.sqrt((audio**2).mean())) if len(audio) else 0.0
        if rms < args.gate:
            print(f"  rms={rms:.4f}  --silent-- (below gate, skipped)"); return 0.0
        au = resample_linear(audio, sr, c["target_sr"])
        if len(au) < need: au = np.pad(au, (0, need-len(au)))
        p = score(interp, logspec(au[:need], c))
        hit = p >= c["threshold"]
        print(f"  rms={rms:.4f}  P(rumble)={p:.3f}  {'>>> RUMBLE <<<' if hit else 'no rumble'}")
        return p
    if args.wav:
        a, sr = read_wav(args.wav)
        print(f"  {args.wav} ({sr} Hz, {len(a)} samples)"); run_clip(a, sr); return
    if args.mic:
        src = args.label or f"mic {args.mic}"
        header("RUMBLE DETECTOR - LIVE", [
            f"source    : {src}",
            f"device    : {args.mic} @ {args.cap_sr} Hz",
            f"window    : {c['clip_len_s']}s   threshold: {c['threshold']}   gate(rms): {args.gate}",
            f"model     : {args.model}",
            f"started   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Ctrl+C to stop",
        ])
        while True:
            raw = subprocess.run(["arecord","-D",args.mic,"-f","S16_LE","-r",str(args.cap_sr),
                 "-c","1","-d",str(int(c["clip_len_s"])),"-t","raw","-q"],
                capture_output=True).stdout
            a = wav_bytes_to_float(raw); t = time.strftime("%H:%M:%S")
            print(f"[{t}]", end=""); run_clip(a, args.cap_sr)
    print("give --wav <file> or --mic plughw:0,0")

if __name__ == "__main__":
    main()
