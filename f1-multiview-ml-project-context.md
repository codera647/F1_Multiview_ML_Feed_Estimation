# F1 multiview watch tool with an ML feed recommender

Project context document. Written to capture the full shape of the idea, the reasoning behind each decision, and the concrete build plan, so it can be picked back up at any point without losing the thread.

## 1. What this project actually is

A tool that displays several F1 video feeds at once (onboard cameras, broadcast, timing tower), similar to F1TV's Multi View feature, with one addition: a small trained model watches the live timing data (gaps, sectors, flags) and recommends which feed is worth featuring right now, rather than leaving the viewer to guess.

The project has two genuinely separate systems inside it:

- A data pipeline that reads live timing numbers and produces a prediction. This is the ML component.
- A video pipeline that plays camera streams into a grid of tiles. This is a systems/software engineering component, not ML.

These two pipelines never share code. The only place they meet is a small piece of app logic that reads the model's output and highlights whichever video tile corresponds to the recommended driver. Keeping this separation explicit matters, both technically and for the story this project tells in an interview: the model does the modeling, the app does the display.

## 2. Why this project, and how it's scoped

The original motivation was career focused: building real domain experience in F1 plus a portfolio project, aimed either at F1 teams directly, their technology partner companies (AWS, Oracle, Cognizant, Neural Concept, and similar), or more realistically as a strong differentiator in a broader AI/ML engineer job search. As originally framed, "build a multiview app" is mostly a software/systems project and would not, on its own, demonstrate modeling work. The project was deliberately reshaped to include a genuine ML decision (the feed recommender) sitting inside the app, so the finished result shows both real systems engineering and real modeling work, not just one or the other.

The project is scoped in three phases, in priority order:

1. The ML model itself (the actual modeling work, and the part worth doing first, since it needs no video plumbing at all)
2. A lightweight results dashboard (no video, just numbers and a replay), used to prove the model works and to demo it convincingly
3. The full video multiview app (optional, later, once phases 1 and 2 are solid)

## 3. The core problem the model solves

With ten or more feeds running at once, nobody can watch all of them, and manually guessing which one matters is exhausting. A TV director does this job for broadcast, picking one feed at a time. The model automates a simplified version of that judgment: given the current state of the race, which driver's feed is about to be worth watching.

## 4. Where the raw data comes from

The primary data source is FastF1, a Python library that exposes official F1 timing and telemetry data.

- `session.laps`, per lap position for every driver, used to detect position changes lap to lap
- Car telemetry and track position (X, Y coordinates) at high frequency within a lap, used for finer grained, mid lap position detection
- `session.race_control_messages`, structured flag and safety car events with timestamps
- Live gap and interval data between cars

No video, no images, no camera data feed into this at any point. Every column the model ever sees is a number or a short category.

## 5. What a raw data row looks like

One row per driver per moment in time (a snapshot taken every few seconds during a session):

| time | driver | gap_ahead | gap_change_5s | in_drs_zone | lap | tyre | tyre_age | flag |
|---|---|---|---|---|---|---|---|---|
| 14:32:00 | VER | 0.9 | -0.4 | 1 | 45 | SOFT | 12 | GREEN |
| 14:32:00 | HAM | 23.1 | 0.0 | 0 | 45 | MED | 20 | GREEN |
| 14:32:00 | LEC | 1.6 | -0.7 | 1 | 45 | SOFT | 8 | GREEN |
| 14:32:05 | VER | 0.3 | -0.6 | 1 | 45 | SOFT | 12 | GREEN |
| 14:32:05 | LEC | 1.1 | -0.5 | 1 | 45 | SOFT | 8 | GREEN |

This is the same X, y shape as any ordinary tabular ML project. Nothing about the "video streams and multiple camera views" framing changes this table.

## 6. How the label is defined and built

There is no existing labeled dataset for "worth watching." It has to be constructed from historical, already finished races, using hindsight: since the outcome is already known, the actual future can be read and stamped backward onto the situation that preceded it. This is a form of weak or heuristic supervision, a standard technique for bootstrapping training data without manual labeling.

Precise label definition used for the first version of the model:

**label = 1 if the driver completes an on-track pass (overtaking or being overtaken) within the next 10 seconds. label = 0 otherwise.**

Both the overtaking driver and the overtaken driver get label = 1 on their rows in that window, since either driver's feed would show the same moment. Every other driver on track at that timestamp, running with a stable gap, gets label = 0.

Worked example, sampling every 5 seconds, with a pass completing at 14:32:07:

| time | gap to car ahead | situation | label |
|---|---|---|---|
| 14:31:45 | 2.4s | closing slowly | 0 |
| 14:31:50 | 2.1s | closing slowly | 0 |
| 14:31:55 | 1.8s | closing faster | 0 |
| 14:32:00 | 0.9s | DRS active | 1 |
| 14:32:05 | 0.3s | DRS active | 1 |
| 14:32:07 | pass completes | | |
| 14:32:10 | now ahead | resolved | 0 |

The label flips to 1 ten seconds before the pass, not at the moment of the pass, which is what makes the model useful: it needs to flag the moment before the action so a viewer can switch and watch it happen.

The labeling itself is a script, not manual annotation. It walks through every row of a historical session, checks the already known result, and stamps 0 or 1 automatically. It runs across a whole season in seconds.

Important scoping decision: this is one label for one well defined question. Other candidate events (pit stop undercut threat, purple sector, flag events) are not merged into this same label. Each would be its own separate task with its own label, built later if wanted, not blended into "worth watching" as one fuzzy bucket.

## 7. Features

Adding a feature does not require touching the label. All features are additional columns computed for the same rows, from the same underlying FastF1 data already being pulled.

Starting feature set:
- Gap to car ahead, current value
- Gap change over the last 3, 5, and 10 seconds (closing rate, stronger signal than the raw gap)
- In DRS detection zone, yes or no
- Lap number and stint length (tyre age proxy)
- Tyre compound
- Sector time relative to personal best or session best
- Recent pit stop nearby (undercut signal)
- Current flag status
- Weather

Start with the strongest two or three (closing rate and DRS zone are the likeliest to carry the most signal), get a working model, then add the rest and retrain on the same labeled rows to see what actually helps.

## 8. Model choice

Primary choice: **XGBoost** (LightGBM is an equally valid, functionally interchangeable alternative).

Reasoning: this is structured, tabular, moderate sized data, and the real signal is in interactions between features (closing rate and DRS zone and late stint together matter more than any one alone). Gradient boosted trees are specifically strong at capturing that kind of interaction without hand specifying rules. Deep learning is deliberately avoided for the first version, it needs more data than this problem has, and it produces a much harder to explain result, which matters given the point of the project is to demonstrate real modeling judgment.

Baseline: logistic regression, trained first just to confirm the features carry signal at all before moving to XGBoost.

Future v2, only after the above works: a sequence model (small LSTM or similar) over the last N seconds of gap history, to capture momentum patterns a single snapshot of features cannot. Not a starting point.

## 9. Training and evaluation

- **Split by race, not by row.** Rows from the same race must not appear in both train and test, or the evaluation will leak information and look better than it is.
- **Do not use plain accuracy.** Positive labels (an overtake happening imminently) are rare relative to quiet racing, likely under 10 percent of rows. A model that always predicts "nothing happening" would still score high accuracy while being useless. Use precision and recall, or AUC-PR.
- **Chronological backtest.** Replay a full, real race through the trained model as if it were live, and check whether the model's top pick at each moment actually lines up with real, known overtakes. This backtest is also the best demo artifact this project produces, a side by side of the model's picks against what actually happened in a race people can recognize.

## 10. Data scope

Roughly 20 cars times 60 to 70 laps times about 20 races per season gives on the order of 20,000 to 30,000 lap level observations per season. A few seasons (2021 through the present is a reasonable window, more consistent regulations and data quality than older years) comfortably reaches the hundreds of thousands of rows, more if sampling at sub lap granularity. Plenty for XGBoost.

## 11. Phase 2: the lightweight results dashboard (MVP deliverable)

Purpose: a trained model with a precision number printed in a notebook does not convince anyone. This dashboard reuses the same chronological backtest from evaluation and turns it into something visible: a real historical race replaying, with the model's live score shown next to each driver, and a marker showing when real overtakes actually happened. This is close in spirit to F1's own Battle Forecast broadcast graphic.

No video, no F1TV subscription needed for this piece.

Suggested stack: Streamlit or Plotly Dash, pure Python, fastest path to something working and presentable given the model itself is already in Python. A small React frontend with an API serving the model's scores is a more polished option later, but is cosmetic effort best deferred until the model is proven.

Core features:
- Leaderboard of drivers with live gap to the car ahead, replaying a chosen historical race at adjustable speed
- The model's current score per driver, shown as a highlight, color, or bar
- A timeline marker for real overtakes, so the model's rising score just before each one is visible

## 12. Phase 3: the full video multiview app (optional, later)

This is the original "multiview watch tool" idea, revisited once phases 1 and 2 are working.

**Existing landscape, worth studying before building:**

- **MultiViewer for F1** (multiviewer.app), the main cross platform desktop client for playing official F1 TV, IndyCar, and other feeds side by side with live timing. Core is closed source, but exposes a documented local API (F1MV API, REST v1/v2 plus GraphQL, with local discovery).
- Companion tools built on that local API: `mvf1` (Python, controls players programmatically), `npm_f1mv_api` (Node/TypeScript client for the local API), a Stream Deck plugin, `UF1 Viewer with F1MV` (adds extra visualization windows), and several smart home integrations that sync lights to race events.
- Standalone clients that rebuild the multi feed experience from scratch instead of building on MultiViewer: `f1viewer` (Go, terminal UI, hands playback to mpv/VLC), `Race Control` (C#/.NET, Windows only), and `Formula 1 Multiviewer` (actively developed, built on zium.app, an alternative F1TV frontend, requires an active F1TV subscription).
- Pure live timing tools with no video at all: `f1gopher` (Go, library plus CLI plus web server), and FastF1's own SignalR client for capturing the live feed.

**Recommended approach:** build a companion app on top of MultiViewer's documented local API rather than reimplementing video streaming and DRM handling from scratch, which is the highest risk, highest effort, and lowest career value part of this space. The whole ecosystem assumes the user's own legitimate, paid F1TV login, orchestrated locally, never redistributed streams.

**Tech stack:**
- App shell: Tauri (Rust plus a web frontend, lighter weight) or Electron plus React (faster iteration, larger install size, used by several of the companion tools above)
- Frontend: React or Svelte for the resizable panel grid, a WebSocket client talking to the local API
- Data layer: Python with FastF1's SignalR client, or Node with `npm_f1mv_api`
- Video playback: mpv or libVLC, controlled programmatically, the same pattern `f1viewer` and `Race Control` use. Not a video player built from scratch.
- Local storage: SQLite or plain JSON for saved layouts

## 13. Resources required

- FastF1 (installed locally, free, used for both historical training data and live capture)
- MultiViewer for F1 installed, and its local API documentation, for phase 3 only
- An active F1TV subscription for phase 3 (Access tier covers timing, Pro or Premium needed for onboard cameras), a real budget line
- mpv or the libVLC SDK, for phase 3 only

## 14. Hardware and compatibility

- **Model training (phase 1):** any modern laptop CPU. XGBoost does not need a GPU at this data scale.
- **Dashboard (phase 2):** negligible requirements, runs comfortably on any machine that can run a browser.
- **Video multiview app (phase 3):** the real hardware bottleneck is video, not ML. Concurrent hardware accelerated decode (H.264/HEVC) of four to six simultaneous streams is comfortable on a modern laptop with a competent GPU; more than that strains most consumer hardware, which is also why F1TV's own Multi View caps out around six feeds. Bandwidth matters too, several concurrent HD streams plus a live data feed adds up quickly. Cross platform target (Windows, macOS, Linux) matches the existing ecosystem.

## 15. Reference research papers (2025 to 2026)

- Thomas et al., "Explainable Reinforcement Learning for Formula One Race Strategy," arXiv:2501.04068. RL agent trained on a Mercedes style Monte Carlo simulator, with SHAP based explainability.
- Todd et al., "Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy," arXiv:2501.04067. Deep learning plus XGBoost forecasting tyre energy from real Mercedes telemetry.
- Fieni et al., "Towards Learning-Based Formula 1 Race Strategies," arXiv:2512.21570. Joint optimization of energy, tire wear, and pit timing.
- Fieni, Wuthrich, Neumann, Onder (ETH Zurich), "Learning-based Multi-agent Race Strategies in Formula 1," arXiv:2602.23056. Multi agent RL with self play for competitor aware strategy.
- Cappello and Hoegh, "A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing," arXiv:2512.00640. Bayesian state space model on public FastF1 data.

None of these solve the exact "worth watching" problem this project defines, but they're the closest published work on FastF1 based modeling and are worth reading for technique and for how they frame evaluation.

## 16. Build order

1. **Weeks 1 to 3, domain immersion only, no code.** Current F1 technical regulations, race strategy content (Chain Bear and Driver61 on YouTube), the papers above, and a skim of existing public F1 data projects so nothing gets rebuilt by accident.
2. **Weeks 4 to 10, the model.** Pull two or three seasons with FastF1, build the raw per lap per driver table, write and sanity check the labeling script, engineer the starting feature set, train XGBoost, evaluate with precision/recall and the chronological backtest.
3. **Weeks 10 to 12, the dashboard.** Build the Streamlit replay dashboard described in section 11, using the same backtest data.
4. **Later, optional, the video app.** Phase 3 as described in section 12, only once the above is solid.

## 17. Open decisions still to make

- Snapshot interval for live rows: every 5 seconds is a reasonable starting point, worth testing against lap level granularity for simplicity in an early version.
- Whether to expand beyond the single overtake label into a second, separate task (an undercut alert) later, and if so, how it should be surfaced differently in the eventual app (a full screen alert rather than a single tile highlight, since it does not map to one driver's feed the way an overtake does).
- Whether phase 3 is worth building at all once phases 1 and 2 exist, or whether the portfolio value is already captured by the model and dashboard alone.
