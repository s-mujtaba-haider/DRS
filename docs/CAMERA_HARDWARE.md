# Camera hardware for ball tracking and decision review

This page covers what each class of camera setup can and cannot support, with the numbers behind each judgement.

Specifications are taken from the vendor pages linked here (checked 18 Oct 2026). Prices are list prices shown on those pages and change often. Nothing is purchased automatically. The software works today with pre-recorded single-camera footage. Its camera model (`CameraModel.time_offset_ms` / `sync_uncertainty_ms`) and the multi-view triangulation API are already designed for synchronised streams.

## 1. The numbers that matter

A fast delivery travels at about 40 m/s (144 km/h). The ball's diameter is about 7.2 cm (Law 4, Size 1 ball).

**Distance travelled between frames.** This sets how precisely the time of an event (impact, bounce, edge, bails off) can be known:

| Frame rate | Ball travel per frame | ± half a frame |
|---|---|---|
| 25 fps (PAL broadcast) | 160 cm | ±80 cm |
| 50 fps | 80 cm | ±40 cm |
| 60 fps | 67 cm | ±33 cm |
| 120 fps | 33 cm | ±17 cm |
| 160 fps (Basler a2A1920-160) | 25 cm | ±12.5 cm |
| 240 fps | 17 cm | ±8 cm |
| 340 fps (reported Hawk-Eye rate) | 11.8 cm | ±5.9 cm |
| 522 fps (FLIR BFS-U3-04S2 at 720×540) | 7.7 cm | ±3.8 cm |

Interpolating between frames with a fitted flight model reduces this for *positions*. It does not help for *events* such as contact or bails-off, where the frame either shows the event or does not.

**Motion blur.** The streak length is speed × exposure:

| Exposure | Blur at 40 m/s |
|---|---|
| 1/50 s (open broadcast shutter) | 80 cm |
| 1 ms | 4 cm (more than half a ball) |
| 0.5 ms | 2 cm |
| 0.25 ms | 1 cm |
| 0.1 ms | 4 mm |

Target an exposure of ≤ 0.25 ms. Short exposure needs light: sunlight is fine, while floodlit matches need fast lenses, mono sensors or gain (with more noise).

**Spatial resolution.** A camera 25 m from the striker with a 3.45 µm pixel (Sony IMX392):

| Lens | mm per pixel | Field width (1920 px) | Ball diameter in pixels |
|---|---|---|---|
| 25 mm | 3.45 mm | 6.6 m | 21 px |
| 35 mm at 22 m | 2.2 mm | 4.2 m | 33 px |
| 50 mm | 1.7 mm | 3.3 m | 42 px |

Formula: mm per pixel = pixel pitch × distance / focal length.

**Synchronisation.** A timing error δt between two cameras displaces the triangulated ball by up to v·δt along its path:

| Timing error | Position error at 40 m/s |
|---|---|
| 1 ms | 4 cm |
| 0.1 ms | 4 mm |
| 1 frame at 160 fps (6.25 ms) | 25 cm |

Frame-level software alignment is therefore not enough for 3D tracking. Use a hardware trigger or PTP-scheduled acquisition.

**Data rate.** Uncompressed 8-bit mono is width × height × fps bytes per second:

| Stream | Data rate |
|---|---|
| 1920×1200 at 160 fps | 369 MB/s per camera |
| 720×540 at 522 fps | 203 MB/s per camera |
| 4 cameras at 1920×1200, 160 fps | 1.47 GB/s |

That exceeds what a single consumer NVMe sustains once its cache fills.

**RAM.** A 5 s pre-trigger ring buffer at 1920×1200 and 160 fps needs 1.84 GB per camera. Implications:
- With 16 GB of system RAM, two such buffers alongside the detector are workable.
- Four cameras need 32 GB.
- Six or more need 64 GB, or recording at reduced ROI.

## 2. Four classes of setup

| Class | Example | What it supports | What it cannot support |
|---|---|---|---|
| **A. Broadcast or phone footage** (25–60 fps, rolling shutter, compressed, cuts and zoom) | Existing match video; phone slow-motion (often 240 fps at reduced resolution, rolling shutter; check each phone) | Replay review with a human deciding. Image-space ball tracking. Approximate single-camera LBW, which the engine refuses to make definite | Reliable edge timing; bat-pad order; metric 3D; frame-accurate run-outs unless the camera is square-on and the event happens to fall between frames |
| **B. Research high-speed, single camera** | 1× global-shutter machine-vision camera at 160–520 fps | Much better event timing and less blur. Monocular flight fit with physics priors | Validated 3D, because depth comes from priors |
| **C. Multi-camera research rig** | 2–4 synchronised global-shutter cameras, calibrated against surveyed pitch points | Triangulated 3D ball position with measured uncertainty. LBW projection with a validated calibration. Run-out and no-ball side cameras | Officiating use without independent validation (see `INTERNATIONAL_DRS_REQUIREMENTS.md` §9) |
| **D. Professional officiating installation** | Vendor systems such as Hawk-Eye: reported 6 cameras at 340 fps (Wikipedia DRS article; vendor figures are not independently published), plus approved edge detection, broadcast integration, redundancy and ICC approval | All ICC DRS reviews | Not reproducible by this project; the approval process and accuracy tests are not public |

## 3. Example cameras with verified specifications

| Camera | Sensor / shutter | Resolution and rate | Interface and sync | List price | Source |
|---|---|---|---|---|---|
| Basler ace 2 R **a2A1920-160umBAS** (mono) / ucBAS (colour) | Sony IMX392, 1/2.3", 3.45 µm, **global shutter** | 1920×1200 at 160 fps | USB 3.0; C-mount; about 3.1 W | €329 / US$359 | https://www.baslerweb.com/en/shop/a2A1920-160umBAS/ , https://baslerweb.com/en/shop/a2A1920-160ucBAS |
| Basler ace 2 R **a2A1920-165g5mBAS** | IMX392 global shutter | 1920×1200 at 168 fps default, 193 fps with 8-bit sensor depth (Basler how-to) | 5GigE; Basler GigE cameras support PTP (IEEE 1588-2008) with 1 ns timestamp ticks | (see shop) | https://www.baslerweb.com/en/shop/a2a1920-165g5mbas/ , https://docs.baslerweb.com/knowledge/how-to-achieve-a-higher-frame-rate-on-a2a1920-165g5mg5c-cameras , https://docs.baslerweb.com/precision-time-protocol |
| Teledyne FLIR **Blackfly S BFS-U3-04S2M/C** | Sony IMX287, 1/2.9", 6.9 µm, global shutter | 720×540 at **522 fps** (8-bit ADC, ISP off); 437 fps at 10-bit; 320×240 at 997 fps; minimum exposure 4 µs | USB 3.1; opto-isolated trigger input; mount C or CS depending on suffix | about US$361 (Edmund Optics) | https://softwareservices.flir.com/BFS-U3-04S2/latest/Model/spec.html |

Notes:
- **Measure the frame rate yourself.** The headline rate depends on bit depth, ROI, ISP settings and bandwidth. The FLIR table drops from 522 to 290 fps with the ISP on. Basler quotes 168 fps by default and 193 fps in 8-bit for the 5GigE model.
- **USB 3 cameras have no PTP.** Synchronise them with a shared hardware trigger signal into each camera's opto-isolated input, such as a function generator or a microcontroller square wave, and record the trigger count.
- **PTP precision depends on the network.** Basler states that "the precision of PTP synchronisation depends to a large extent on your network hardware". Use a switch with IEEE 1588 hardware support, and measure the result (§5).
- **Not verified here:** lens prices, and whether specific Basler 5GigE models support "scheduled action commands". Check the camera's documentation page before buying.

## 4. Lenses and placement

- **Lens mount and format.** Use C-mount lenses rated for 1/2.3" (IMX392) or 1/2.9" (IMX287) sensors and at least the sensor's resolution. Prefer fixed focal length with locking screws, because a lens that moves invalidates the calibration.
- **Behind-the-bowler view (LBW).** Mount it elevated, roughly 20–30 m from the striker, with a 25–50 mm lens (table in §1). It sees the line well but depth poorly.
- **Side-on or oblique view.** At least one camera needs a large angle to the behind-the-bowler view (ideally 60–90°) for good depth along the pitch. Triangulation conditioning is reported by `ml/reconstruction`.
- **Crease cameras (run-out, stumping, no ball).** Place one square-on to each popping crease, level with the line. Use a high frame rate and a short exposure. Off-square views create parallax between the bat tip and the line.
- **Boundary.** Each camera covers a rope segment. Rope geometry comes from calibration, or from a line the operator marks.
- **Calibration targets.** Survey the stumps, the crease corners and extra ground markers with a tape measure or total station. See `calibrate_cameras.py`; calibration is never assumed from the image alone.

## 5. Synchronisation options, best first

1. **Hardware trigger.** A single TTL or opto signal drives every camera's exposure. Jitter is in the microsecond range (camera-specific, so measure it), and frame indices align by construction.
2. **PTP (IEEE 1588) with scheduled acquisition** on GigE or 5GigE cameras. Measure the offset with an LED strobe visible to all cameras.
3. **Free-running cameras with timestamp recording.** Measure the offset per session with an LED flash, clap or bat-ball sound. Record it as `time_offset_ms` with an honest `sync_uncertainty_ms`; the reconstruction marks results as degraded.
4. **Broadcast feeds.** Synchronisation is unknown. Do not triangulate.

**Validation test.** Run an LED strobe at a known rate (e.g. 1 kHz, square wave) in view of all cameras. Each camera's on/off phase gives the offset and jitter distribution, which should be recorded with each session.

## 6. Storage and compute

- **Recording.** Record to RAM ring buffers and write only the delivery window (about 3–5 s) per ball. One delivery is about 1.1–1.8 GB per 1920×1200 camera uncompressed. Either use a lossless or high-bit-rate compression stage (NVENC on the RTX 3080 Ti supports H.264/HEVC, but lossy compression must be validated so it does not hurt detection), or store raw clips on a second NVMe.
- **Interfaces.** Each USB 3 camera should have its own host controller; check how the motherboard shares lanes. For 5GigE, use one NIC port per camera or a switch with sufficient uplink.
- **Inference.** At 160 fps × 2 cameras = 320 fps to process, the current YOLO11s detector needs measuring on the actual pipeline (`scripts/benchmark_inference.py`). Crop-based inference around the predicted ball position is the main lever.

## 7. Staged recommendation for this workstation

| Stage | Hardware | Approx. cost | Unlocks |
|---|---|---|---|
| 0 (now) | Existing RTX 3080 Ti PC; existing footage; phone slow-motion for experiments | 0 | Annotation, detector training, approximate single-view analysis, replay-review workflow |
| 1 | **RAM upgrade to 32 GB** | Modest; check local prices | Two to four camera ring buffers alongside model inference without swapping |
| 2 | 2× Basler a2A1920-160umBAS (US$359 each) + 2 C-mount lenses + USB 3 cables + a microcontroller trigger | About US$1–1.5k (lenses unverified) | Synchronised two-view 3D at 160 fps, validated calibration, real LBW projection experiments |
| 3 | 1–2× FLIR BFS-U3-04S2M (about US$361) for crease views at 522 fps (720×540) | About US$0.8k with lenses | Frame-accurate run-out and no-ball experiments |
| 4 | Move to 5GigE + PTP: 4–6 cameras, a PTP-capable switch, a second NVMe, 64 GB RAM | Several thousand US$ | A multi-camera research installation; measured sync |
| 5 | Officiating grade | Vendor or partner territory | Redundant cameras, approved edge detection, independent evaluation, competition acceptance |

Stages 0–1 need no new cameras. Stage 2 is the smallest purchase that turns approximate single-camera depth into measured 3D. That is the most important step towards credible LBW results.
