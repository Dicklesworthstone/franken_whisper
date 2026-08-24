# FRANKENTUI_METHODOLOGY.md

> Planning methodology for the optional TUI feature in franken_whisper,
> covering the separation between robot mode and TUI mode, the integration
> with the `ftui` crate, and guidelines for extending the TUI.

---

## 1. Design Principles

franken_whisper is agent-first. The primary interface is robot mode: line-oriented
NDJSON output with a versioned schema, designed for machine consumption. The TUI is
a secondary, optional interface for human operators who want to inspect runs,
segments, and pipeline events interactively.

This ordering is not negotiable. Robot mode correctness and schema stability always
take priority. TUI features must never alter the pipeline's behavior, output
contracts, or persistence format.

---

## 2. Feature Gating

The TUI is gated behind the `tui` Cargo feature flag:

```toml
[features]
default = []
tui = ["dep:ftui"]
```

When `tui` is not enabled, the `run_tui()` function returns an error:

```rust
#[cfg(not(feature = "tui"))]
pub fn run_tui() -> FwResult<()> {
    Err(FwError::Unsupported(
        "tui feature is disabled; rebuild with `--features tui`".to_owned(),
    ))
}
```

When `tui` is enabled, the `ftui` crate (`../frankentui/crates/ftui`) provides the
rendering framework. The dependency uses the `crossterm` backend for terminal I/O:

```toml
ftui = { path = "../frankentui/crates/ftui", default-features = false, features = ["crossterm"], optional = true }
```

This means the TUI adds zero compile-time or runtime cost when the feature is off.

---

## 3. Robot Mode vs. TUI Mode

### Robot Mode (Agent-First)

- Entry point: `franken_whisper robot run <args>`
- Output: NDJSON to stdout, one line per event.
- Schema version: `ROBOT_SCHEMA_VERSION` (currently `"1.1.0"` — the additive listen-family bump), included in every line.
- Event types:
  `run_start`, `stage`, `run_complete`, `run_error`,
  diagnostic/introspection events such as `backends.discovery` and `health.report`,
  `routing_decision` output from `franken_whisper robot routing-history`,
  and speculative streaming events such as
  `transcript.partial`, `transcript.confirm`, `transcript.retract`,
  `transcript.correct`, and `transcript.speculation_stats`, plus the live
  listen family (`listen.session_start`, `speech_started`,
  `transcript.delta`, `utterance_end`, `listen.warning`,
  `listen.session_stats`) emitted by `fw robot listen`.
- Streaming: pipeline events arrive via `mpsc::channel` from `FrankenWhisperEngine::transcribe_with_stream` and are emitted as they arrive.
- Contract: deterministic where possible, explicit error codes, no human decoration mixed with machine output.
- Introspection: `franken_whisper robot schema` emits a self-describing JSON document with required fields and examples for every event type.

### TUI Mode (Human-First)

- Entry point: `franken_whisper tui`
- Output: interactive terminal UI rendered by `ftui`.
- Data source: reads from frankensqlite (`RunStore`) on startup and on periodic ticks (every 7.5 seconds, via 500ms tick with reload every 15 ticks).
- No pipeline execution: the TUI does not run transcriptions. It is a read-only viewer of persisted run data.
- DB path: `FRANKEN_WHISPER_DB` env var, defaulting to `.franken_whisper/storage.sqlite3`.

### Key Separation Rules

1. TUI code lives entirely in `src/tui.rs`, wrapped in `#[cfg(feature = "tui")] mod enabled`.
2. TUI never imports or calls robot mode functions. Robot mode never imports TUI.
3. The pipeline (`src/orchestrator.rs`) has no knowledge of TUI existence.
4. Both modes share the same data model (`src/model.rs`) and storage layer (`src/storage.rs`).
5. TUI reads persisted data; robot mode emits live data. They never compete for the same I/O channel.

---

## 4. Current TUI Architecture

### Application Structure

The TUI is built on the `ftui` Model-View-Update (MVU) architecture:

```
WhisperTuiApp (state)
    |
    +-- Model::update(Msg) -> Cmd   (state transitions)
    +-- Model::view(&self, Frame)   (rendering)
    +-- Model::subscriptions()      (timer ticks)
```

**State** (`WhisperTuiApp`):
- `db_path: PathBuf` -- frankensqlite database location.
- `runs: Vec<RunSummary>` -- recent runs (capped at `RUNS_LIMIT = 64`).
- `selected_run: usize` -- index of the highlighted run.
- `details: Option<StoredRunDetails>` -- full details for the selected run (segments, events, etc.).
- `focus: FocusPane` -- which pane has keyboard focus (`Runs`, `Timeline`, `Events`).
- `timeline_scroll: u16` / `events_scroll: u16` -- scroll offsets for detail panes.
- `show_help: bool` -- help overlay toggle.
- `status_line: String` -- footer status message.
- `tick_count: u64` -- monotonic tick counter for periodic refresh.

**Messages** (`Msg`):
- `Key(KeyEvent)` -- keyboard input.
- `Tick` -- 500ms timer tick from subscription.
- `Ignore` -- unhandled events (mouse, resize, etc.).

### Screen Layout

```
+------------------------------------------------------------------+
| Header: franken_whisper :: run status + transcript timeline       |
+--------------------+---------------------------------------------+
|                    |                                             |
|  Runs [FOCUS]      |  Transcript Timeline                       |
|  > 2026-02-22...   |  000 00:00:00 -> 00:01:02 segment text    |
|    2026-02-21...   |  001 00:01:02 -> 00:02:15 ...             |
|    ...             |                                             |
|                    +---------------------------------------------+
|                    |  Stage Events                               |
|                    |  001 [ingest] ingest.start | materializing  |
|                    |  002 [ingest] ingest.ok | input materialized|
+--------------------+---------------------------------------------+
| Status: Loaded N runs | focus=Runs | Tab/Shift+Tab | q quit     |
+------------------------------------------------------------------+
```

Three panes:
- **Runs** (left, 33%): list of recent runs with selection marker.
- **Transcript Timeline** (right-top, ~62%): segments for the selected run with timestamps, speaker labels, and confidence scores.
- **Stage Events** (right-bottom, ~38%): pipeline events for the selected run showing sequence, stage, code, and message.

Compact mode: when terminal is smaller than 70x16 characters, the layout collapses to a single-pane summary.

Help overlay: toggled with `h` or `?`, rendered as a centered bordered rectangle.

### Keyboard Controls

| Key              | Action                                    |
|-----------------|-------------------------------------------|
| Tab             | Cycle focus forward: Runs -> Timeline -> Events -> Runs |
| Shift+Tab       | Cycle focus backward                       |
| Up/Down         | In Runs: move selection. In Timeline/Events: scroll. |
| PageUp/PageDown | Move/scroll by 8 units                     |
| r               | Force reload from database                 |
| h / ?           | Toggle help overlay                        |
| q / Ctrl+C      | Quit                                       |

### Data Flow

```
frankensqlite (storage.sqlite3)
    |
    +-- RunStore::open(db_path)
    |
    +-- list_recent_runs(RUNS_LIMIT) -> Vec<RunSummary>
    |       used for the Runs pane
    |
    +-- load_run_details(run_id) -> Option<StoredRunDetails>
            used for Timeline and Events panes
            includes: segments, events, warnings, acceleration, replay
```

The TUI opens a fresh `RunStore` connection on each reload cycle. This avoids holding
a long-lived database handle and ensures the TUI always sees the latest committed data,
even if another process (robot mode run, sync import) has written to the database since
the last reload.

---

## 4.1 LiveTranscriptionView (bd-339.1)

The `LiveTranscriptionView` is a standalone TUI component designed for real-time
transcription display. Unlike the main TUI (which reads persisted data from the
database), this view is fed segments as they arrive from the `StreamingEngine`.

### Component Structure

```rust
pub(crate) struct LiveTranscriptionView {
    segments: Vec<TranscriptionSegment>,  // accumulated segments, arrival order
    diarization_active: bool,             // controls speaker label display
    backend: BackendKind,                 // backend producing the transcription
    started_epoch_secs: f64,              // wall-clock start time
    current_epoch_secs: f64,              // updated via tick
    scroll_offset: u16,                   // vertical scroll position
    auto_scroll: bool,                    // auto-follow latest segment
    max_segments: usize,                  // retention limit (default 10,000)
}
```

### Key Behaviors

- **Auto-scroll**: Enabled by default. The view automatically adjusts `scroll_offset`
  so the most recently arrived segment is always visible. Scrolling up manually
  disables auto-scroll; scrolling to the bottom re-enables it. `jump_to_latest()`
  force-re-enables auto-scroll.

- **Segment retention**: When `segments.len()` exceeds `max_segments` (default
  `DEFAULT_MAX_SEGMENTS = 10_000`), the oldest segments are drained from the front.
  The scroll offset is adjusted to compensate for removed entries.

- **Speaker labels**: When `diarization_active` is true, each segment line includes
  a `[SPEAKER_XX]` prefix extracted from `TranscriptionSegment::speaker`.

- **Status bar**: Rendered at the bottom of the view area. Shows elapsed time
  (`HH:MM:SS`), segment count, backend name, scroll state (`LIVE` or `PAUSED`),
  and diarization status.

- **Rendering**: `render(area, frame, focused)` splits the area into transcript +
  status bar. The transcript uses `Paragraph` with scroll; the border style changes
  based on focus state (cyan when focused, gray when not).

### API for Integration

| Method                     | Purpose                                    |
|---------------------------|--------------------------------------------|
| `new(backend, diarize)`   | Create view in initial empty state          |
| `push_segment(seg)`       | Add a single segment (auto-scroll aware)    |
| `push_segments(iter)`     | Batch add segments                          |
| `tick(epoch_secs)`        | Update wall-clock for elapsed display       |
| `scroll_up(n)`            | Manual scroll up, disables auto-scroll      |
| `scroll_down(n)`          | Manual scroll down, re-enables at bottom    |
| `jump_to_latest()`        | Force re-enable auto-scroll                 |
| `segment_count()`         | Number of accumulated segments              |
| `segment_lines()`         | Formatted lines for rendering               |
| `effective_scroll(height)`| Compute scroll offset for visible area      |
| `status_bar_text()`       | Formatted status bar string                 |
| `render(area, frame, focused)` | Full render into given area            |

### Segment Line Format

```
{index:03} {start_ts} -> {end_ts} [{speaker}] {text} ({confidence:.3})
```

Where `start_ts` and `end_ts` are formatted as `HH:MM:SS.mmm`, the speaker label
appears only when diarization is active, and confidence appears when present.

---

## 4.2 Planned: Speaker Color-Coding

When diarization is active, each unique speaker label (e.g. `SPEAKER_00`,
`SPEAKER_01`) will be assigned a distinct color from a predefined palette. This
provides visual differentiation between speakers in both the existing Timeline
pane and the LiveTranscriptionView.

Implementation plan:
- Maintain a `HashMap<String, PackedRgba>` mapping speaker labels to colors.
- Assign colors from a palette on first encounter (cycling on overflow).
- Apply per-line styling in `timeline_lines()`, `segment_lines()`, and `render()`.
- The color palette should be legible on both dark and light terminal backgrounds.

Status: **Planned** -- infrastructure is ready (segment lines include speaker labels;
`LiveTranscriptionView` tracks `diarization_active`), but color assignment and
per-line styled rendering are not yet implemented.

---

## 4.3 Planned: Waveform Visualization

A waveform display component showing audio amplitude over time, synchronized with
the transcript timeline. This provides visual context for segment boundaries and
silence regions.

Implementation plan:
- Read amplitude data from the normalized WAV file (already available at persist time).
- Render a sparkline or block-character waveform using ftui primitives.
- Synchronize horizontal scroll with the Timeline pane's time axis.
- Highlight the currently selected segment's time range.

Status: **Planned** -- requires reading PCM sample data and downsampling for display.
The `audio` module already produces 16kHz mono WAV files suitable for this purpose.

---

## 4.4 Planned: Search and Filter

Text search and filter capabilities across transcript segments and pipeline events.

Implementation plan:
- Add a search input mode (activated by `/` key, following vi convention).
- Filter segments and events by substring match on text, speaker, stage, or code.
- Highlight matching text in rendered lines.
- Navigate between matches with `n` (next) and `N` (previous).
- Filter mode: when active, non-matching lines are hidden rather than just highlighted.

Status: **Planned** -- the existing `timeline_lines()` and `event_lines()` methods
generate all display lines; filtering can be applied as a post-processing step before
rendering.

---

## 5. How to Add New TUI Screens/Views

### Step 1: Define the New Pane

Add a variant to `FocusPane`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusPane {
    Runs,
    Timeline,
    Events,
    NewPane,  // <-- add here
}
```

Update the `next()` and `prev()` methods to include the new variant in the focus cycle.

### Step 2: Add State Fields

Add any state the new pane needs to `WhisperTuiApp`:

```rust
struct WhisperTuiApp {
    // ... existing fields ...
    new_pane_scroll: u16,
    new_pane_data: Vec<String>,
}
```

Initialize the new fields in `WhisperTuiApp::new()`.

### Step 3: Add Data Loading

If the new pane needs data from storage, add a query method to `RunStore` in
`src/storage.rs`, then call it from `WhisperTuiApp::reload_data()`.

If the new pane derives its content from existing `StoredRunDetails` fields, add
a rendering helper method on `WhisperTuiApp` (following the pattern of
`timeline_lines()` and `event_lines()`).

### Step 4: Handle Input

In `Model::update`, add key handling for the new pane under the appropriate
`FocusPane` match arm:

```rust
KeyCode::Up => match self.focus {
    // ... existing arms ...
    FocusPane::NewPane => {
        self.new_pane_scroll = self.new_pane_scroll.saturating_sub(1);
    }
},
```

### Step 5: Render the Pane

In `Model::view`, adjust the layout constraints to accommodate the new pane. Use
`ftui::layout::Flex` to split the available area. Render the pane content using
`ftui::widgets::paragraph::Paragraph` (or another ftui widget) inside a
`Block::bordered()` with a title that reflects focus state.

Follow the existing pattern:

```rust
let title = if self.focus == FocusPane::NewPane {
    "New Pane [FOCUS]"
} else {
    "New Pane"
};
Paragraph::new(self.new_pane_content())
    .scroll((self.new_pane_scroll, 0))
    .block(Block::bordered().title(title).border_style(
        if self.focus == FocusPane::NewPane {
            Style::new().fg(PackedRgba::rgb(/* focus color */)).bold()
        } else {
            Style::new().fg(PackedRgba::rgb(120, 120, 120))
        },
    ))
    .render(pane_rect, frame);
```

### Step 6: Add Tests

Add tests in the `#[cfg(test)] mod tests` block inside `mod enabled`. Tests should:
- Verify the new pane's data loading (use `seed_runs` fixture helper).
- Verify input handling (construct `Msg::Key` events and call `app.update`).
- Verify the focus cycle includes the new pane.
- Verify edge cases (empty database, high-volume data, scroll boundary clamping).

Tests run without a terminal -- `ftui::Model` is pure state machine logic that can
be exercised without rendering.

---

## 6. Event Model: Pipeline to TUI Display

The pipeline and TUI are decoupled through the persistence layer:

```
Pipeline (orchestrator.rs)
    |
    +-- EventLog::push() emits RunEvent with monotonic seq
    |       |
    |       +-- [optional] Sender<StreamedRunEvent> for robot mode live streaming
    |
    +-- RunReport assembled at pipeline end
    |
    +-- RunStore::persist_report() writes to frankensqlite
            |
            +-- runs table (metadata, result, replay envelope)
            +-- segments table (per-segment records)
            +-- events table (stage events with payload)

TUI (tui.rs)
    |
    +-- RunStore::list_recent_runs() -> Runs pane
    +-- RunStore::load_run_details() -> Timeline pane (segments) + Events pane (events)
```

The TUI never receives live events from the pipeline. It polls the database on a
timer (every 15 ticks at 500ms = ~7.5s reload interval) and refreshes its state.

This design has several advantages:
- The pipeline has no runtime coupling to the TUI.
- The TUI can be started and stopped independently of pipeline execution.
- Multiple TUI instances can read the same database concurrently.
- The TUI always shows the persisted (committed) state, which matches what robot
  mode would have emitted.

For future real-time TUI updates during a live pipeline run, the `StreamedRunEvent`
mechanism (currently used only by robot mode) could be extended to also feed a TUI
event channel. This would require adding a second receiver in the TUI's subscription
model, but the pipeline-side infrastructure (`EventLog` with `event_tx`) already
supports it.

The `LiveTranscriptionView` component (section 4.1) is specifically designed to
consume streaming segments via its `push_segment()` / `push_segments()` API. When
the `StreamedRunEvent` channel is connected to the TUI, segments extracted from
stage events can be fed directly into `LiveTranscriptionView` for real-time display.

---

## 7. ftui Framework Integration Points

The `ftui` crate provides:

| Component                  | Usage in franken_whisper                        |
|---------------------------|-------------------------------------------------|
| `ftui::App`               | Application runner, manages terminal lifecycle  |
| `ftui::Model` trait       | Implemented by `WhisperTuiApp`                  |
| `ftui::Cmd`               | Command type returned by `update()` (`none()`, `quit()`) |
| `ftui::core::event::Event`| Terminal events (key, tick, resize)             |
| `ftui::layout::Flex`      | Flexbox-style layout splitting                  |
| `ftui::layout::Constraint`| Size constraints (Fixed, Fill, Percentage)      |
| `ftui::widgets::Paragraph`| Text rendering with scroll support              |
| `ftui::widgets::block::Block` | Bordered containers with titles             |
| `ftui::render::frame::Frame` | Render target for `view()`                   |
| `ftui::runtime::Every`    | Timer subscription (500ms tick)                 |
| `ftui::ScreenMode`        | `InlineAuto { min_height: 20, max_height: 44 }` |
| `ftui::Style`, `PackedRgba` | Styling (colors, bold)                        |

The TUI uses `ScreenMode::InlineAuto` which renders inline in the terminal (not
fullscreen alternate screen), with height clamped between 20 and 44 rows. This
allows the TUI to coexist with other terminal output above it.

---

## 8. Testing Strategy

TUI tests live inside `#[cfg(feature = "tui")] mod enabled` and exercise:

1. **State management**: selection, scrolling, focus cycling, reload.
2. **Data volume**: 120 runs (verifies RUNS_LIMIT cap), 2000 events per run (verifies scroll stability).
3. **Edge cases**: empty database, no events, extreme scroll positions.
4. **Fixture helpers**: `seed_runs(db_path, count, events_per_run)` creates realistic test data via `RunStore`.

Tests do not render to a terminal. They create `WhisperTuiApp` instances, call
methods directly, and assert on state. The `Model::update` method is called with
constructed `Msg` values to simulate keyboard input.

Run TUI tests with:

```bash
cargo test --features tui
```

---

## 9. Future Directions

### Recently Implemented

- **LiveTranscriptionView** (bd-339.1): Real-time segment display component with
  auto-scroll, speaker labels, timestamps, elapsed time, and backend status bar.
  Retention limit with oldest-first drain. Scroll-up pauses auto-follow. See
  section 4.1 for full details.

### Planned TUI Enhancements

All enhancements preserve the robot mode contract.

- **Speaker color-coding** (section 4.2): Per-speaker color assignment from a palette
  for visual differentiation in Timeline and LiveTranscriptionView.
- **Waveform visualization** (section 4.3): Sparkline/block-character audio amplitude
  display synchronized with the transcript timeline.
- **Search and filter** (section 4.4): Text search across segments and events with
  highlight, navigation, and filter modes.
- Live pipeline progress view via `StreamedRunEvent` channel, feeding the
  `LiveTranscriptionView` component.
- Evidence ledger explorer pane showing backend routing decisions from the
  `RoutingEvidenceLedger`.
- Replay pack inspector for forensic analysis.
- Latency decomposition visualization (stage timing sparklines), consuming the
  `stage_latency_decomposition_v1` artifact from the orchestrator.
- Acceleration/GPU status dashboard.
- Configuration editor for stage budgets and routing policy.

### Rules for Future TUI Work

All future TUI work must follow the same rules:
- Gated behind `#[cfg(feature = "tui")]`.
- Read-only access to pipeline data (via storage or event channel).
- No impact on robot mode output or pipeline behavior.
- Tested with the fixture-based approach (no terminal rendering in tests).
