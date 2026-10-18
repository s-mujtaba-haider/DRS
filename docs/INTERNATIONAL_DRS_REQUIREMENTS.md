# International DRS requirements

What an international-standard Decision Review System has to do, which rules govern it, and what this project can and cannot claim.

Compiled 18 Oct 2026 from public sources. Every rule fact gives its source. Items marked **(unverified)** could not be confirmed from a primary source during this review. The machine-readable version of the review rules lives in `configs/drs_protocol/*.yaml` and is applied by `ml/protocol`.

## Sources

| Ref | Source | What it establishes |
|---|---|---|
| [L26] | MCC, *Laws of Cricket, changes for 2026* (Feb 2026) — https://www.lords.org/getmedia/72990a68-98cd-4c6e-89ac-a2dce54533ad/Ten-major-changes-to-Laws-for-2026-edition_5.pdf | New edition of the Laws in force from October 2026. Includes Law 19.5.2 (boundary catches), Law 27.3.1 (keeper position after release), Law 35 (hit wicket) and Appendix A2.7/A2.8 (ball "held in hand" with "complete control", needed for run-out and stumping) |
| [LAWS] | MCC Laws online — https://lawsofcricket.lords.org/laws-of-cricket/ (e.g. `/appeals-dismissals/leg-before-wicket`, `/appeals-dismissals/run-out`) | Full text of Laws 19, 29, 30, 31, 33, 36, 38 and 39. The pages render with JavaScript and could not be machine-read here, so the summaries below are paraphrased and must be checked against the official text |
| [ICC25] | ICC, *Several changes made to ICC playing conditions* (27 Jun 2025) — https://www.icc-cricket.com/news/several-changes-made-to-icc-playing-conditions | Wicket zone becomes "the actual outline of the stumps and bails". Effective for Tests from 17 Jun 2025, ODIs from 2 Jul 2025 and T20Is from 10 Jul 2025. Boundary-catch change adopted ahead of the Laws |
| [ICC-D3] | ICC men's Test Appendix D, Protocol 3 Player Review (page dated 28 Sep 2017) — https://www.icc-cricket.com/news/mens-test-match-appendix-d-decision-review-system-drs-and-third-umpire-protocol-3-player-review ; ODI equivalent: https://www.icc-cricket.com/news/mens-odi-match-appendix-d-decision-review-system-drs-and-third-umpire-protocol-3-player-review | 15 s request limit. Who may review. Centre-of-ball definitions for pitching, impact and wicket. Umpire's call. 300 cm and 250 cm + 40 cm distance rules. Reviews that count as unsuccessful. Handling of technology failure |
| [ICC-D1] | ICC men's Test Appendix D, Protocol 1 General — https://www.icc-cricket.com/news/mens-test-match-appendix-d-decision-review-system-drs-and-third-umpire-protocol-1-general | DRS requires the minimum camera specification plus **approved** ball-tracking and sound-based edge detection, a neutral ICC-appointed third umpire and an ICC Technical Officer |
| [ICC23] | ICC, *changes to playing conditions* (May 2023, effective 1 Jun 2023) — https://www.icc-cricket.com/news/icc-announces-changes-to-playing-conditions-x9146 | Soft signal abolished |
| [CA22] | cricket.com.au, *Umpire's call explained* (16 Mar 2022) — https://www.cricket.com.au/news/3319658 | No umpire's call for pitching. Wicket zone raised to the top of the bails in April 2021. Distance rules still applied in 2022 |
| [WIKI] | Wikipedia, *Decision Review System* — https://en.wikipedia.org/wiki/Decision_Review_System | Secondary. Review counts (3 per Test innings, 2 per ODI/T20I innings). 80-over top-up ended in 2017. Hawk-Eye 6 cameras at 340 fps, Virtual Eye 4 at 230 fps |

The **2025 edition of the ICC Appendix D text was not retrieved.** The ICC website's Appendix D pages are dated 2017. Every protocol config therefore lists its unverified items, and `ml/protocol` warns about them in every decision.

## 1. Decisions that are reviewed

Players may review any on-field decision about whether a batter is dismissed, except *Timed out* [ICC-D3 §3.1.1]. Only the dismissed batter may review an OUT, and only the fielding captain may review a NOT OUT [ICC-D3 §3.1.3]. Umpires may refer other decisions to the third umpire, and since 1 Jun 2023 they do so without a soft signal [ICC23].

| Decision | Typical path | Main evidence |
|---|---|---|
| LBW (Law 36) | Player review | Ball tracking (pitching, impact, wicket), edge detection for bat-first, no-ball check |
| Caught, incl. caught behind (Law 33) | Player review (edge); umpire referral (clean catch) | Edge detection (audio/visual), replays of the catch, ball-ground contact |
| Run out (Law 38, with Laws 29 and 30) | Umpire referral | Side-on frames at the popping crease, wicket-break frame, bat/foot grounding |
| Stumped (Law 39) | Umpire referral | Same as run out, plus keeper possession and whether the batter was attempting a run |
| Boundary 4/6 and boundary catches (Law 19) | Umpire referral | Rope position, ball-ground contact, fielder grounding and airborne contacts |
| No ball (front foot), and checks within other reviews | Third umpire / technology | Front-foot landing frame against the popping crease |
| Hit wicket, obstruction, etc. | Umpire referral | Replays |

## 2. The rules and how they shape the software

The software keeps three layers separate:

- **evidence engines** answer geometric and temporal questions with uncertainty, in `ml/lbw` and the other decision modules;
- **the Laws** decide which conditions matter, in `ml/lbw/rules.py`;
- **the competition protocol** sets umpire's-call bands, distance rules and review counts, in `ml/protocol` with `configs/drs_protocol`.

### LBW: Law 36 plus the ICC protocol
- **Law 36 conditions** [LAWS], paraphrased:
  - a fair delivery (not a no ball);
  - the ball did not pitch outside leg;
  - the ball did not touch the bat first;
  - the first interception is by the striker's person;
  - the impact is in line, or outside off when no genuine shot is offered;
  - the ball would have hit the wicket.
- **ICC ball-tracking categories** [ICC-D3 §3.4]:
  - *Pitching* is judged by the **centre of the ball**: In Line, Outside Off or Outside Leg. There is no umpire's call for pitching [ICC-D3, CA22].
  - *Impact* is judged against a zone between the wickets of indefinite height:
    - In Line: the centre is inside the zone.
    - Umpire's Call: some part of the ball is inside but the centre is outside.
    - Outside: no part of the ball is inside.
  - *Wicket*:
    - Hitting: the centre is inside the wicket zone.
    - Umpire's Call: the ball hits the wicket but the centre is outside the zone.
    - Missing: the ball misses.
- **Wicket zone definition:**
  - 2017: outside of the outer stumps, the base of the stumps and the **bottom of the bails** [ICC-D3].
  - April 2021: raised to the **top of the bails** [CA22].
  - June/July 2025: "**the actual outline of the stumps and bails**" [ICC25]. This project models it as the rectangle up to the top of the bails between the outer stump edges **(unverified approximation)**.
- **Overturning:**
  - An on-field NOT OUT needs Hitting, plus impact In Line. If no shot is offered, impact In Line, umpire's call on the off side or Outside Off also qualifies.
  - An on-field OUT needs Missing, impact Outside (Outside Leg when no shot is offered), or pitching Outside Leg [ICC-D3 §3.4.5–3.4.6].
- **Distance rules.** An on-field NOT OUT stands if:
  - the impact is ≥ 300 cm from the stumps; or
  - the impact is > 250 cm from the stumps and the ball travelled < 40 cm between pitching and impact.

  Source: [ICC-D3 §3.4.6.4, CA22]. That these rules remain in force after the 2025 changes is **(unverified)**. They are configurable per protocol.
- **Review allowance:**
  - Unsuccessful reviews per innings: 3 in Tests, 2 in ODIs and T20Is [WIKI] **(secondary)**.
  - Reviews that are successful, or unchanged on umpire's call, do not count against the allowance [ICC-D3 §3.6].
  - The 80-over top-up ended in 2017 [WIKI].
  - Requests must come within 15 s of the ball becoming dead [ICC-D3 §3.2.2].
- **Software impact.** The engine has to produce *probabilities of each category* (centre vs any part, by side) and the impact distances, not just "hit / miss". These are `LbwResult.zone_probabilities` and `WicketProjection.p_centre_in_zone_by_height`. When the uncertainty straddles a band, the review is `INCONCLUSIVE`; the system does not force a category.

### Run out and stumped: Laws 29, 30, 38 and 39
- **Wicket put down (Law 29):** a bail completely removed or a stump struck out of the ground, by the ball or by a fielder's hand or arm holding the ball.
- **Ball held in the hand (2026 edition, A2.7/A2.8):** the ball must be *held with complete control*, not merely touching the hand [L26].
- **Out of ground (Law 30):** a batter is out of ground unless some part of the person or the bat in hand is grounded behind the popping crease. A batter who is running or diving and has grounded beyond the crease is not out of ground if contact is then lost (bat-bounce protection).
- **Stumped (Law 39):** wicket-keeper, the striker out of ground and not attempting a run; a no ball protects from stumping.

**Software impact:** the decision is a **temporal ordering** of two events: the first frame the wicket is down and the first frame of grounding behind the line. With frame period Δt, each event is only known to lie within an interval. The software must report the intervals and return INCONCLUSIVE when they overlap. It must also never infer grounding from a frame in which the bat is airborne or hidden.

### Caught and boundaries: Laws 33 and 19
- A catch requires the ball not to have touched the ground before it is held with complete control.
- **2026 Law 19.5.2 [L26, ICC25]** (first adopted by the ICC in 2025):
  - A fielder who has gone beyond the boundary may touch the ball while airborne **only once**.
  - After that touch, the fielder must be wholly grounded inside the boundary for the rest of the delivery.
  - Landing or stepping outside after that touch means a boundary is scored.
- **Software impact:** boundary review is a sequence of contact and grounding events relative to the rope (who touched what, when, and where the fielder was grounded), not a single ball position.

### Edges
- Bat contact is evidence for caught decisions and for LBW (bat first).
- ICC DRS requires *approved* sound-based edge detection [ICC-D1].
- A trajectory deflection or a sound spike is a **candidate**, never proof (see `ml/contact_analysis`).

## 3. Computer-vision tasks per decision

| Decision | Tasks |
|---|---|
| LBW | Ball detection and tracking; calibrated multi-view triangulation (or monocular with physics priors, flagged); bounce detection; impact time; flight-model fit with uncertainty; projection to the stump plane; zone probabilities; batter handedness; edge and no-ball checks |
| Caught behind / edge | Ball and bat localisation in the contact window; deflection analysis; synchronised audio at a high sample rate; frame selection |
| Bat-pad order | Temporal ordering of two contacts; requires a frame period shorter than the gap between the contacts |
| Run out / stumping | Crease line in the image (from calibration or marked); bails and stumps state change; bat-tip and foot localisation; grounding; event intervals; keeper possession |
| Boundary | Rope geometry (calibrated or marked); ball-ground contact; fielder foot positions and airborne state; ball-fielder contact sequence |
| Front-foot no ball | Popping-crease line; landing frame of the front foot; whether some part of the foot is behind the line |

## 4. Evidence available and its limits

- **Broadcast video (25/50/60 fps, interlaced or compressed):**
  - At 40 m/s (144 km/h) the ball moves 0.67–1.6 m between frames.
  - Impact and bounce times are quantised to that spacing.
  - Cuts, zoom and graphics break tracking.
  - It is adequate for a replay review of run-outs only when the camera is square to the crease.
- **Hawk-Eye–class tracking:** 340 fps per camera, about 0.12 m between frames at 40 m/s [WIKI]. That is still more than one ball diameter (7.2 cm), so the impact frame remains quantised.
- **Audio:** stump microphones at broadcast sample rates (48 kHz is typical). Synchronisation with video is not guaranteed, and the audio path latency is unknown unless measured.
- **Without calibration:** image coordinates only. Metric statements, such as distances and zone tests, need a validated calibration.

## 5. Camera and synchronisation requirements

See `CAMERA_HARDWARE.md` for the numbers. In summary:
- at least two calibrated, hardware-synchronised views of the pitch for 3D ball tracking;
- a frame rate of at least 150–340 fps;
- exposure short enough that blur is under about 1 cm (≤ 0.25 ms at 40 m/s);
- square-on high-frame-rate views of each popping crease for run-outs and no balls;
- coverage of the boundary for boundary decisions;
- synchronisation error well under one frame. 1 ms at 40 m/s is a 4 cm along-track error, so the target is ≤ 0.1 ms, which needs a hardware trigger or PTP.

## 6. Outputs the software produces

- **LBW:** `LbwResult` contains:
  - the verdict: PROJECTED_TO_HIT, PROJECTED_TO_MISS, INCONCLUSIVE or INSUFFICIENT_DATA;
  - pitching, impact and stump-plane estimates with 90% intervals;
  - zone probabilities;
  - rule checks;
  - warnings and the model version.
- **DRS protocol:** `ProtocolDecision` contains:
  - the outcome: UPHELD, OVERTURNED, UMPIRES_CALL or INCONCLUSIVE;
  - the decision displayed: OUT, NOT_OUT or REFER_TO_THIRD_UMPIRE;
  - whether the review is retained;
  - per-component status (pitching, impact, wicket) with the probabilities used;
  - the protocol id and version.
- **Review counts:** per team and innings, the unsuccessful, successful and retained reviews and the reviews remaining.
- **Every output** carries a disclaimer that it is experimental and that INCONCLUSIVE means a human third umpire decides.

## 7. Technical risks

| Risk | Effect | Mitigation in this project |
|---|---|---|
| Detector misses or false positives on tiny or blurred balls | Wrong or broken track | Track-quality gates; reliability flags; manual correction |
| Calibration error | Biased zone tests | Calibration validation flag; definite verdicts refused on unvalidated calibration |
| Synchronisation error | 3D error along the flight path | Sync-uncertainty metadata; reconstruction degraded when sync is unknown |
| Low frame rate | Event-time quantisation | Event intervals; INCONCLUSIVE when intervals overlap |
| Flight-model inadequacy (swing, seam, spin after bounce) | Projection bias | Alternative-model probe; covariance inflation; distance limit |
| Outdated playing conditions | Wrong review semantics | Versioned protocols with sources and `unverified` lists |
| Over-trust by users | Misuse as an official system | Disclaimers; INCONCLUSIVE preferred to forced answers |

## 8. Research prototype vs officially accepted system

ICC DRS requires:
- *approved* ball-tracking and edge-detection technology;
- a neutral third umpire;
- an ICC Technical Officer;
- the minimum broadcast camera specification [ICC-D1].

Approval criteria and test results for commercial providers are not public. This project:
- is **not** approved by any cricket authority;
- has **unknown** real-world accuracy (see `EVALUATION.md` and `LIMITATIONS.md`);
- must not be used to make or overturn official decisions.

It may serve as an analysis and training aid, or as a replay-review tool in which a human makes the call. INCONCLUSIVE outputs are designed for that role.

## 9. Validation required before operational use

1. **Static geometry:**
   - Surveyed targets on the pitch.
   - 3D error and reprojection error per camera.
   - Repeatability across days.
2. **Synchronisation:**
   - A measured offset distribution per camera pair, using an LED strobe or hardware trigger test.
3. **Ball localisation:**
   - Error against an independent reference, such as a ball launcher with known release, a radar gun, or a second independent tracking system, across lighting conditions.
4. **Trajectory and prediction:**
   - Physical tests in which the ball actually reaches the stumps, for example deliveries with no batter and a stump-hit sensor or high-speed side camera.
   - Stump-plane error distribution, and interval coverage of the 90% intervals.
5. **Decision level:**
   - Agreement with expert panels on a held-out set, with sample sizes reported.
   - Rates of INCONCLUSIVE outcomes and of confident errors.
6. **Run-out and boundary:**
   - Frame-interval tests with a known event timing, such as an electronically triggered bail.
7. **Operations:**
   - Turnaround time.
   - Failure modes and redundancy.
   - Operator training.
   - Audit logs.
8. **Independent technical evaluation**, then acceptance by the competition authority.

## 10. What cannot be verified about commercial systems

The following are not published, so no comparison is possible:
- internal algorithms;
- calibration procedures;
- flight models, and how spin or seam is handled after the bounce;
- uncertainty treatment;
- test methodology and results.

Public figures are marketing or press claims:
- 340 fps and 6 cameras [WIKI];
- a claimed average error of about 2.2 mm (theconversation.com, 2013).

The ICC commissioned accuracy testing in 2012 (ESPNcricinfo, https://www.espncricinfo.com/print/story/553449), but the results were not found. Nothing in this project copies or reverse-engineers a commercial system.
