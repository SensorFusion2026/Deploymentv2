# Rumble Detection Pipeline

Listens to field audio, recognizes elephant infrasound **rumbles**, and sends an
alert. Read the tree top to bottom: each line names the file, its type, and why
it exists in the chain.

```
Deploymentv2/
├── README.md                        .md   notes — this file: the chain, start to end
├── requirements.txt                 .txt  build-world Python deps (workstation only)
├── models/
│   ├── rumble_model3.tflite         .tflite  the trained MODEL (model3) — everything exists to run it
│   └── README.md                    .md   notes — provenance: where it came from + its accuracy
├── runtime/         ── RUN the model (forever, on the field device) ──
│   │                only src/ ships to the Jetson; everything else here is build-time
│   ├── src/         ◀ DEPLOYS to the device — running's single-job parts, in flow order:
│   │   ├── config.{h,c}             interface + code — load the one settings file
│   │   ├── capture.{h,c}            interface + code — listen to the microphone
│   │   ├── ring_buffer.{h,c}        interface + code — gather audio into fixed 5 s clips
│   │   ├── resampler.{h,c}          interface + code — adjust the audio rate (to 4 kHz)
│   │   ├── spectrogram.{h,c}        interface + code — prepare each clip for the model  ★
│   │   ├── classifier.{h,cpp}       interface + code — run the model
│   │   ├── detector.{h,c}           interface + code — decide if it counts as a rumble
│   │   ├── event_sink.{h,c}         interface + code — announce the hit (event shape lives here)
│   │   └── main.c                   .c    code — the one conductor wiring the parts in order
│   ├── spectrogram_parity/  ◀ BUILD-TIME ONLY — workstation, NOT on the Jetson, NOT shipped
│   │   ├── reference_spectrogram.py .py   script — the known-correct "prepare" answer (TensorFlow)
│   │   └── test_spectrogram_parity.c .c   code — checks src/spectrogram.c matches it   ★
│   └── CMakeLists.txt               build — assembles src/ into one runnable program
├── config/
│   └── pipeline.json                .json settings — ONE source every part reads (no drift)
└── client/
    └── README.md                    .md   notes — a separate display that reads the hit notice
```

`.h` is the **interface** (the "menu" a part offers); `.c` / `.cpp` is the
**code** (the "kitchen" behind it). `.py` scripts run only in the build world;
`.json` files are shared data every part agrees on.

**Scope** — this repo only **runs** the rumble detector (model3). The model is
trained and quantized *outside* this repo (in ELP-Rumble-Detector, on the
cluster). Only two things cross that boundary into here: the **model artifact**
(`models/`) and the **recipe + normalization constants** it was trained with
(`config/pipeline.json`). Everything else about training stays in the training
repo. The gunshot detector is a separate deployment with a deliberately
different recipe; it is not part of this repo.

**The hit notice (event shape)** lives as a documented struct in `event_sink.h` —
the code that emits it is its own contract, so it can't drift. No separate
`events/` folder yet: a formal schema only earns its keep once a second person
is building the display in parallel against it.

**What deploys to the Jetson:** `runtime/src/` (compiled) + `config/pipeline.json`
+ `models/`. Everything else — `runtime/spectrogram_parity/`, `client/`,
`requirements.txt` — is build-time or off-device, and never runs on the device.

★ **parity** — the model learned from audio prepared one exact way. If
`spectrogram` prepares it even slightly differently, the model guesses worse with
no error to warn you. The parity check proves `src/spectrogram.c` matches the
reference (built from the rumble *training* code, not deployment code); it runs
**once on your workstation before a build**, and is never deployed.

**Status** — built & verified: `pipeline.json` (rumble model3 recipe), the
`spectrogram` interface + implementation, and the parity check (PASS vs
TensorFlow, max diff ~3e-4). Next: `config.{h,c}` or `capture.{h,c}`. Open item:
confirm whether the `.tflite` is fp16 or int8.