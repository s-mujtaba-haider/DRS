# Match-day operation

This guide covers running the system during a live match. At club or research level it works as a **third-umpire replay and evidence console**: it records the cameras and puts the right replay in front of the official within seconds. It also helps the official mark and measure what matters, keeps the review count, and stores an auditable evidence package for every decision.

> **Who decides.** The system never decides a review. It shows an *advisory* recommendation under the selected DRS protocol, and a human official records the final decision. Ball-tracking (LBW) outputs are experimental and their real-world accuracy is unknown (see `LIMITATIONS.md`). Using this system in a sanctioned competition requires the organiser's and the governing body's acceptance (`INTERNATIONAL_DRS_REQUIREMENTS.md` §8–9).

## What it can and cannot support today

| Decision | What the system provides | How far to trust it |
|---|---|---|
| **Run out / stumping** | Frame-accurate replay, crease line (clicked or from calibration), wicket-break candidate detection, per-frame in/out-of-ground measurement with uncertainty, Law 29/30/38/39 checklist, and a timeline of grounding against the break window | The geometry is exact for what the operator marks. The limit is frame rate: at 50 fps a 20 ms gap can hide the moment of grounding, and the engine then says `INCONCLUSIVE` (the batter gets the benefit of the doubt only if the official decides so). Use 100–240 fps side-on cameras on the creases. |
| **Boundary / catch near the rope** | Replay, rope tracing, an event sequence (ball grounds, fielder touches, airborne contacts, rope contact), and Law 19 / 19.5.2 (2026) and Law 33 interpretation | Marked events are interpreted exactly. Image-space rope geometry is only valid for points **on the ground**. Needs a camera along the rope. |
| **LBW** | Ball tracking, calibration, monocular 3D flight fit, projected path with uncertainty, Law 36 checks, and the protocol's umpire's-call logic | **Experimental.** Synthetic benchmark only; often `INCONCLUSIVE` by design. Treat it as supporting evidence, not as a Hawk-Eye equivalent. |
| **Caught (edge)** | Replay, deflection candidates (≥ 100 fps), catch-near-boundary conditions | No audio edge detection (no validated synchronised audio). The official judges the edge from the replays. |

## Equipment

The minimum useful setup is the PC below plus two cameras:

- **Side-on at the striker's popping crease** (square-leg line): run-outs and stumpings at that end.
- **Behind the bowler's arm** (high, on the pitch axis): LBW replays, ball tracking and line.

Better: add a side-on camera at the non-striker's end and one camera per boundary arc. Camera options, frame-rate and exposure arithmetic, and budgets are in `CAMERA_HARDWARE.md`.

| Item | Requirement |
|---|---|
| Cameras | USB3 or GigE machine-vision cameras, or IP cameras over RTSP. Short exposure (≤ 1/2000 s) matters more than resolution. |
| PC | Measured on an RTX 3080 Ti, Ryzen 7 7700 and 16 GB RAM: 3 cameras at 1080p100 with NVENC in process mode is near the limit (`EVALUATION.md` §7). |
| Disk | About 35 MB/s per 1080p100 camera (measured, noisy content). A 300 s rolling buffer takes about 10.5 GB per camera. |
| Power | A UPS for the PC and the camera switch. |

## Before play

1. **Start the services** (`INSTALLATION.md`), then open http://localhost:3100.
   - **Roles.** Set `DRS_OPERATOR_PIN` and `DRS_UMPIRE_PIN` (in `.env` or the environment) before starting the backend. Every change then needs a PIN, entered in the sidebar. Only the umpire PIN can record or withdraw a decision. Viewing stays open.
   - **A second screen or PC for the third umpire.** Start the backend with `--host 0.0.0.0` and add the console's address to `DRS_CORS_ORIGINS` (for example `["http://192.168.1.20:3100"]`). Build the frontend with `NEXT_PUBLIC_API_BASE=http://<backend-ip>:8100`. Use a wired, isolated network: the API has no TLS.
2. **Live capture → add cameras.** Use a device index (`0`, `1`, …) or an `rtsp://` URL, and set the resolution and fps the camera actually supports. Start recording.
   - Check each panel: *FPS measured* should match *declared*, *Dropped frames* should be 0, and *Backlog* should stay near 0.
   - Rate mismatches are flagged; many webcams deliver half their declared rate in low light.
   - For 2+ high-rate cameras set `DRS_CAPTURE_PROCESS_PER_CAMERA=true` before starting the backend.
3. **Calibrate each fixed camera once.** Clip a few seconds with the stumps visible, run *Auto-detect* or click the landmarks, and compute.
   - Re-calibrate if a camera is bumped.
   - A calibration is only marked validated if you supply checked intrinsics.
4. **Match day → New match.** Enter the teams, the DRS protocol (ICC Test/ODI/T20I 2025, or the club protocol without umpire's call), the venue and the third umpire's name.
5. **Rehearse.**
   - In *Live capture*, choose **Rehearsal: replay a video** to feed a recorded match clip in as a looping live camera.
   - Clip it into a review, mark it, record a decision, then withdraw the review.
   - Check the timings on your hardware.

## During play: a review in about a minute

1. **Open the review.** The on-field umpire refers, or a captain signals a review.
   - Match day → the match → *New review*: requested by (team or umpire), decision type, on-field call, over.
   - Team reviews are refused when the team has none left, or already has an open review.
2. **Clip the replay.** In the review, *Clip N camera(s)* exports the last N seconds from every recording camera (default 8 s) and attaches them.
   - Each clip keeps per-frame capture times and reports its sync uncertainty.
   - Clip again to get more context.
   - **Synchronised replay** plays every camera on the shared capture clock: space to play or pause, `,`/`.` to step one frame, speeds from 0.05× to 1×.
   - *Open umpire monitor* opens the same view full screen for a second screen. It picks up new clips automatically.
3. **Mark and analyse:**
   - **Run out / stumping:**
     1. Draw the crease: two clicks on the **back edge** of the popping-crease marking, plus one point on the stump side (or pick a calibration).
     2. Draw the stumps box and run *Auto-detect break*, or step frames with `←`/`→` and mark *last intact* and *first broken*.
        - If the camera pans or zooms, tick *camera moves* and draw the box on a frame where the wicket is intact. The scan starts there and follows the camera.
     3. Tick the Law 29.1 confirmation only after seeing a bail completely removed.
     4. Mark the bat tip or foot in the frames around the break as grounded or in air.
        - *Propose feet (model)* runs a pose model over ±25 frames and draws magenta diamonds on the batter's ankles and feet: solid means likely grounded, dashed means unsure.
        - Clicking near a diamond snaps to it, but you still decide grounded or in air.
        - It never proposes the bat tip.
     5. Set the known facts and run *Analyse and attach*.
   - **Boundary:** trace the rope's inner edge and click a field-side point. Step through the frames and add events in order, ending with *ball dead* or *catch completed*. Set the facts and analyse.
   - **LBW:** run ball tracking on the end-on clip, open the session, check the track, mark the impact frame, run LBW, then attach the analysis.
4. **Compute the recommendation.** It applies the match protocol: umpire's call, review retention, and inconclusive evidence referred back to the official. Read the reasons and warnings.
5. **Record the decision.** The official chooses OUT / NOT OUT (or SIX / FOUR / NO BOUNDARY) and confirms the outcome (upheld, overturned or umpire's call), then signs with their name. Review counts update immediately.

`←`/`→` steps one frame and Shift+`←`/`→` steps ten. The frame number and the capture timestamp are always shown.

## After play

- Each review's **PDF report** contains:
  - the review, the protocol, the advisory recommendation and the official's decision;
  - video provenance, rule checks and timelines;
  - **annotated evidence frames**: the frames around the break with the crease line and marked points, or the boundary frames with the rope.
- Each review's **Evidence package (JSON)** contains:
  - the match, the protocol version, every analysis request and result, and video provenance (SHA-256 plus capture timing);
  - the append-only audit log (who did what, when).
- Archive the packages and the `storage/` folder after the match. The rolling capture buffer deletes old segments automatically; clipped videos are kept.

## Known operational limits

- **Software synchronisation.** Cameras are timestamped on arrival at the PC, not at exposure. Cross-camera timing is uncertain by about one frame interval plus transport jitter.
  - This is fine for single-camera run-out and stumping decisions. For multi-camera 3D it costs about 3.5 cm per ms of error (`EVALUATION.md` §7), so it is flagged as degraded.
  - Hardware-triggered or PTP cameras are the fix (`CAMERA_HARDWARE.md`).
- **No broadcast-style graphics.** The console is for the official, not for the TV feed.
- **One operator, one review at a time per team.**
- **Video only.** No audio snickometer.
