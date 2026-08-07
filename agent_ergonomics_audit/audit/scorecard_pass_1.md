# Pass 1 scorecard

Scores use the 0-1000 anchors from the 11-dimension skill rubric.

| Surface | Pre median | Main gap | Post target |
|---|---:|---|---:|
| CLI root | 400 | no version/orientation/discovery contract | 850 |
| Robot run | 550 | syntax stderr and coarse terminal errors | 900 |
| Model discovery | 250 | readiness/execution contradiction | 850 |
| Installer | 300 | typo mutation, raw errors, silent fallback | 875 |
| Release packaging | 150 | unpinned siblings and advisory shipping | 850 |

Pre-pass median across the five critical surfaces: **300**.

| Surface | Post median | Uplift | Evidence boundary |
|---|---:|---:|---|
| CLI root | 900 | +500 | live debug and packaged-release commands |
| Robot run | 900 | +350 | syntax/help/empty-history process probes |
| Model discovery | 850 | +600 | local resolver plus path-free status probes |
| Installer | 900 | +600 | authenticated success and fail-closed archive probes |
| Release packaging | 850 | +700 | one native DSR target; not a five-target release |

Post-pass median across the five critical surfaces: **900**, for a median uplift of **+600**. Quality-gate status is tracked separately and cannot be inferred from this ergonomics score.
