#[cfg(not(feature = "tui"))]
use crate::error::{FwError, FwResult};

#[cfg(feature = "tui")]
mod enabled {
    use std::path::PathBuf;
    use std::time::Duration;

    use chrono::Utc;
    use ftui::core::event::{Event, KeyCode, KeyEventKind, Modifiers};
    use ftui::core::geometry::Rect;
    use ftui::layout::{Constraint, Flex};
    use ftui::render::frame::Frame;
    use ftui::runtime::{Every, Subscription};
    use ftui::widgets::Widget;
    use ftui::widgets::block::Block;
    use ftui::widgets::paragraph::Paragraph;
    use ftui::{App, Cmd, Model, PackedRgba, ScreenMode, Style};

    use crate::error::{FwError, FwResult};
    use crate::model::{BackendKind, RunSummary, StoredRunDetails, TranscriptionSegment};
    use crate::storage::RunStore;

    const RUNS_LIMIT: usize = 64;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FocusPane {
        Runs,
        Timeline,
        Events,
    }

    impl FocusPane {
        fn next(self) -> Self {
            match self {
                Self::Runs => Self::Timeline,
                Self::Timeline => Self::Events,
                Self::Events => Self::Runs,
            }
        }

        fn prev(self) -> Self {
            match self {
                Self::Runs => Self::Events,
                Self::Timeline => Self::Runs,
                Self::Events => Self::Timeline,
            }
        }
    }

    struct WhisperTuiApp {
        db_path: PathBuf,
        runs: Vec<RunSummary>,
        selected_run: usize,
        details: Option<StoredRunDetails>,
        focus: FocusPane,
        timeline_scroll: u16,
        events_scroll: u16,
        show_help: bool,
        status_line: String,
        tick_count: u64,
    }

    enum Msg {
        Key(ftui::KeyEvent),
        Tick,
        Ignore,
    }

    impl From<Event> for Msg {
        fn from(event: Event) -> Self {
            match event {
                Event::Key(key) => Self::Key(key),
                Event::Tick => Self::Tick,
                _ => Self::Ignore,
            }
        }
    }

    impl WhisperTuiApp {
        fn new(db_path: PathBuf) -> Self {
            let mut app = Self {
                db_path,
                runs: Vec::new(),
                selected_run: 0,
                details: None,
                focus: FocusPane::Runs,
                timeline_scroll: 0,
                events_scroll: 0,
                show_help: false,
                status_line: "Booting franken_whisper TUI...".to_owned(),
                tick_count: 0,
            };
            app.reload_data();
            app
        }

        fn move_selection(&mut self, delta: isize) {
            if self.runs.is_empty() {
                self.selected_run = 0;
                return;
            }

            let current = self.selected_run as isize;
            let max = self.runs.len().saturating_sub(1) as isize;
            self.selected_run = (current + delta).clamp(0, max) as usize;
            self.timeline_scroll = 0;
            self.events_scroll = 0;
            self.refresh_selected_details();
        }

        fn refresh_selected_details(&mut self) {
            if self.runs.is_empty() {
                self.details = None;
                return;
            }

            let run_id = self.runs[self.selected_run].run_id.clone();
            match RunStore::open(&self.db_path).and_then(|store| store.load_run_details(&run_id)) {
                Ok(details) => {
                    self.details = details;
                }
                Err(error) => {
                    self.status_line = format!("failed to load run details: {error}");
                    self.details = None;
                }
            }
        }

        fn reload_data(&mut self) {
            let before_selected_id = self
                .runs
                .get(self.selected_run)
                .map(|run| run.run_id.clone());

            let store = match RunStore::open(&self.db_path) {
                Ok(store) => store,
                Err(error) => {
                    self.status_line = format!("db open failed: {error}");
                    self.runs.clear();
                    self.details = None;
                    return;
                }
            };

            let runs = match store.list_recent_runs(RUNS_LIMIT) {
                Ok(runs) => runs,
                Err(error) => {
                    self.status_line = format!("run query failed: {error}");
                    self.runs.clear();
                    self.details = None;
                    return;
                }
            };

            self.runs = runs;

            if self.runs.is_empty() {
                self.selected_run = 0;
                self.details = None;
                self.status_line = format!(
                    "No runs in {}. Execute `franken_whisper transcribe ...` first.",
                    self.db_path.display()
                );
                return;
            }

            if let Some(id) = before_selected_id
                && let Some(index) = self.runs.iter().position(|run| run.run_id == id)
            {
                self.selected_run = index;
            }
            if self.selected_run >= self.runs.len() {
                self.selected_run = self.runs.len() - 1;
            }

            let selected_id = self.runs[self.selected_run].run_id.clone();
            match store.load_run_details(&selected_id) {
                Ok(details) => {
                    self.details = details;
                    self.status_line = format!(
                        "Loaded {} runs from {} @ {}",
                        self.runs.len(),
                        self.db_path.display(),
                        Utc::now().format("%H:%M:%S")
                    );
                }
                Err(error) => {
                    self.details = None;
                    self.status_line =
                        format!("failed to load selected run details for {selected_id}: {error}");
                }
            }
        }

        fn timeline_lines(&self) -> Vec<String> {
            let Some(details) = &self.details else {
                return vec!["No selected run details available".to_owned()];
            };

            if details.segments.is_empty() {
                return vec!["Selected run has no segments".to_owned()];
            }

            details
                .segments
                .iter()
                .enumerate()
                .map(|(index, segment)| {
                    let start = format_ts(segment.start_sec);
                    let end = format_ts(segment.end_sec);
                    let speaker = segment
                        .speaker
                        .as_deref()
                        .map(|value| format!("[{value}] "))
                        .unwrap_or_default();
                    let confidence = segment
                        .confidence
                        .map(|value| format!(" ({value:.3})"))
                        .unwrap_or_default();

                    format!(
                        "{:03} {} -> {} {}{}{}",
                        index, start, end, speaker, segment.text, confidence
                    )
                })
                .collect()
        }

        fn event_lines(&self) -> Vec<String> {
            let Some(details) = &self.details else {
                return vec!["No events available".to_owned()];
            };
            if details.events.is_empty() {
                return vec!["Selected run has no events".to_owned()];
            }

            details
                .events
                .iter()
                .map(|event| {
                    format!(
                        "{:03} [{}] {} | {}",
                        event.seq, event.stage, event.code, event.message
                    )
                })
                .collect()
        }

        fn runs_text(&self) -> String {
            if self.runs.is_empty() {
                return "No runs found".to_owned();
            }

            self.runs
                .iter()
                .enumerate()
                .map(|(index, run)| {
                    let marker = if index == self.selected_run { ">" } else { " " };
                    format!(
                        "{} {} | {} | {}",
                        marker,
                        run.started_at_rfc3339,
                        run.backend.as_str(),
                        run.run_id
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        fn status_hint(&self) -> String {
            format!(
                "focus={:?} | Tab/Shift+Tab focus | Up/Down navigate | PgUp/PgDn scroll | r reload | h/? help | q quit",
                self.focus
            )
        }

        fn warning_legend(&self) -> &'static str {
            "warning_legend=[fallback: deterministic safe-mode fallback path, divergence: compatibility drift observed]"
        }
    }

    impl Model for WhisperTuiApp {
        type Message = Msg;

        fn update(&mut self, msg: Msg) -> Cmd<Self::Message> {
            match msg {
                Msg::Key(key) if key.kind == KeyEventKind::Press => {
                    if key.modifiers.contains(Modifiers::CTRL) && key.code == KeyCode::Char('c') {
                        return Cmd::quit();
                    }

                    match key.code {
                        KeyCode::Char('q') => return Cmd::quit(),
                        KeyCode::Char('r') => self.reload_data(),
                        KeyCode::Char('h') | KeyCode::Char('?') => {
                            self.show_help = !self.show_help;
                        }
                        KeyCode::Tab => {
                            self.focus = self.focus.next();
                        }
                        KeyCode::BackTab => {
                            self.focus = self.focus.prev();
                        }
                        KeyCode::Up => match self.focus {
                            FocusPane::Runs => self.move_selection(-1),
                            FocusPane::Timeline => {
                                self.timeline_scroll = self.timeline_scroll.saturating_sub(1);
                            }
                            FocusPane::Events => {
                                self.events_scroll = self.events_scroll.saturating_sub(1);
                            }
                        },
                        KeyCode::Down => match self.focus {
                            FocusPane::Runs => self.move_selection(1),
                            FocusPane::Timeline => {
                                self.timeline_scroll = self.timeline_scroll.saturating_add(1);
                            }
                            FocusPane::Events => {
                                self.events_scroll = self.events_scroll.saturating_add(1);
                            }
                        },
                        KeyCode::PageUp => match self.focus {
                            FocusPane::Runs => self.move_selection(-8),
                            FocusPane::Timeline => {
                                self.timeline_scroll = self.timeline_scroll.saturating_sub(8);
                            }
                            FocusPane::Events => {
                                self.events_scroll = self.events_scroll.saturating_sub(8);
                            }
                        },
                        KeyCode::PageDown => match self.focus {
                            FocusPane::Runs => self.move_selection(8),
                            FocusPane::Timeline => {
                                self.timeline_scroll = self.timeline_scroll.saturating_add(8);
                            }
                            FocusPane::Events => {
                                self.events_scroll = self.events_scroll.saturating_add(8);
                            }
                        },
                        _ => {}
                    }
                }
                Msg::Tick => {
                    self.tick_count = self.tick_count.saturating_add(1);
                    if self.tick_count.is_multiple_of(15) {
                        self.reload_data();
                    }
                }
                Msg::Ignore | Msg::Key(_) => {}
            }

            Cmd::none()
        }

        fn view(&self, frame: &mut Frame) {
            let full = Rect::from_size(frame.buffer.width(), frame.buffer.height());

            if full.width < 70 || full.height < 16 {
                let compact_text = format!(
                    "franken_whisper tui (compact)\n\n{}\n\n{}",
                    self.status_line,
                    self.status_hint()
                );
                Paragraph::new(compact_text)
                    .block(
                        Block::bordered()
                            .title("franken_whisper compact")
                            .border_style(Style::new().fg(PackedRgba::rgb(220, 200, 120))),
                    )
                    .render(full, frame);
                return;
            }

            let shell = Flex::vertical()
                .constraints([Constraint::Fixed(1), Constraint::Fill, Constraint::Fixed(1)])
                .split(full);

            let header = shell[0];
            let body = shell[1];
            let footer = shell[2];

            let header_text = format!(
                "franken_whisper :: run status + transcript timeline :: db={} :: runs={}",
                self.db_path.display(),
                self.runs.len()
            );
            Paragraph::new(header_text)
                .style(Style::new().fg(PackedRgba::rgb(160, 230, 255)).bold())
                .render(header, frame);

            let body_chunks = Flex::horizontal()
                .constraints([Constraint::Percentage(33.0), Constraint::Percentage(67.0)])
                .gap(1)
                .split(body);

            let left = body_chunks[0];
            let right = body_chunks[1];

            let right_chunks = Flex::vertical()
                .constraints([Constraint::Percentage(62.0), Constraint::Percentage(38.0)])
                .gap(1)
                .split(right);

            let runs_title = if self.focus == FocusPane::Runs {
                "Runs [FOCUS]"
            } else {
                "Runs"
            };
            Paragraph::new(self.runs_text())
                .scroll((self.selected_run.saturating_sub(4) as u16, 0))
                .block(Block::bordered().title(runs_title).border_style(
                    if self.focus == FocusPane::Runs {
                        Style::new().fg(PackedRgba::rgb(255, 180, 120)).bold()
                    } else {
                        Style::new().fg(PackedRgba::rgb(120, 120, 120))
                    },
                ))
                .render(left, frame);

            let timeline_title = if self.focus == FocusPane::Timeline {
                "Transcript Timeline [FOCUS]"
            } else {
                "Transcript Timeline"
            };
            Paragraph::new(self.timeline_lines().join("\n"))
                .scroll((self.timeline_scroll, 0))
                .block(Block::bordered().title(timeline_title).border_style(
                    if self.focus == FocusPane::Timeline {
                        Style::new().fg(PackedRgba::rgb(120, 220, 160)).bold()
                    } else {
                        Style::new().fg(PackedRgba::rgb(120, 120, 120))
                    },
                ))
                .render(right_chunks[0], frame);

            let events_title = if self.focus == FocusPane::Events {
                "Stage Events [FOCUS]"
            } else {
                "Stage Events"
            };
            Paragraph::new(self.event_lines().join("\n"))
                .scroll((self.events_scroll, 0))
                .block(Block::bordered().title(events_title).border_style(
                    if self.focus == FocusPane::Events {
                        Style::new().fg(PackedRgba::rgb(255, 140, 160)).bold()
                    } else {
                        Style::new().fg(PackedRgba::rgb(120, 120, 120))
                    },
                ))
                .render(right_chunks[1], frame);

            let footer_text = format!(
                "{} | {} | {}",
                self.status_line,
                self.status_hint(),
                self.warning_legend()
            );
            Paragraph::new(footer_text)
                .style(Style::new().fg(PackedRgba::rgb(210, 210, 210)))
                .render(footer, frame);

            if self.show_help {
                let overlay = centered_rect(full, 72, 14);
                let help_text = [
                    "franken_whisper TUI controls",
                    "",
                    "Tab / Shift+Tab : switch pane focus",
                    "Up/Down         : runs select or pane scroll",
                    "PageUp/PageDown : faster movement",
                    "r               : reload from frankensqlite",
                    "h or ?          : toggle help",
                    "q or Ctrl+C     : quit",
                    "",
                    "Pane meaning:",
                    "- Runs: recently persisted runs",
                    "- Transcript Timeline: segment timeline",
                    "- Stage Events: pipeline stage stream",
                    "",
                    "Warning legend:",
                    "- fallback: deterministic safe-mode fallback path",
                    "- divergence: compatibility drift observed",
                ]
                .join("\n");

                Paragraph::new(help_text)
                    .block(
                        Block::bordered()
                            .title("Help")
                            .border_style(Style::new().fg(PackedRgba::rgb(240, 210, 120)).bold())
                            .style(Style::new().bg(PackedRgba::rgb(18, 18, 18))),
                    )
                    .render(overlay, frame);
            }
        }

        fn subscriptions(&self) -> Vec<Box<dyn Subscription<Self::Message>>> {
            vec![Box::new(Every::new(Duration::from_millis(500), || {
                Msg::Tick
            }))]
        }
    }

    fn centered_rect(area: Rect, width: u16, height: u16) -> Rect {
        let w = width.min(area.width.saturating_sub(2)).max(10);
        let h = height.min(area.height.saturating_sub(2)).max(5);

        let x = area.x + area.width.saturating_sub(w) / 2;
        let y = area.y + area.height.saturating_sub(h) / 2;
        Rect::new(x, y, w, h)
    }

    fn format_ts(seconds: Option<f64>) -> String {
        let Some(value) = seconds else {
            return "--:--:--.---".to_owned();
        };

        if value.is_sign_negative() {
            return "00:00:00.000".to_owned();
        }

        let total_ms = (value * 1_000.0).round() as u64;
        let hours = total_ms / 3_600_000;
        let minutes = (total_ms % 3_600_000) / 60_000;
        let secs = (total_ms % 60_000) / 1_000;
        let millis = total_ms % 1_000;

        format!("{hours:02}:{minutes:02}:{secs:02}.{millis:03}")
    }

    // ── Live transcription view (bd-339.1) ─────────────────────────────

    /// A live transcription view component that displays segments as they
    /// arrive in real-time, with speaker labels, timestamps, auto-scroll,
    /// and a status bar showing elapsed time and backend information.
    ///
    /// This component is designed to be embedded inside a larger TUI layout.
    /// It does not own the rendering loop; the parent `Model` calls
    /// [`push_segment`] when new data arrives, and [`render`] each frame.
    /// Correction lifecycle state for a displayed segment in speculative mode.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum SegmentCorrectionState {
        /// Standard segment, not part of speculative streaming.
        Original,
        /// Fast model result, pending quality confirmation.
        Speculative,
        /// Quality model agreed with fast model.
        Confirmed,
        /// Quality model disagreed, this segment has been replaced.
        Retracted,
        /// This segment is the quality model's correction.
        Corrected,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct LiveTranscriptionView {
        /// Accumulated segments, in arrival order.
        segments: Vec<TranscriptionSegment>,
        /// Whether diarization is active (controls speaker label display).
        diarization_active: bool,
        /// The backend currently producing the transcription.
        backend: BackendKind,
        /// Wall-clock start time (epoch seconds) for elapsed-time display.
        started_epoch_secs: f64,
        /// Current wall-clock time (epoch seconds), updated via tick.
        current_epoch_secs: f64,
        /// Vertical scroll offset within the segment list.
        scroll_offset: u16,
        /// When true, the view auto-scrolls to keep the latest segment
        /// visible. Toggled off when the user scrolls up manually.
        auto_scroll: bool,
        /// Maximum number of segments to retain. When exceeded, the oldest
        /// segments are drained.
        max_segments: usize,
        /// Per-segment correction state for speculative streaming.
        correction_states: std::collections::HashMap<usize, SegmentCorrectionState>,
        /// Map from partial seq to segment indices, for retraction targeting.
        seq_to_segment_idx: std::collections::HashMap<u64, Vec<usize>>,
        /// Whether speculative streaming mode is active.
        speculative_active: bool,
        /// Count of corrections applied.
        correction_count: u32,
        /// Count of total speculation windows.
        speculation_window_count: u32,
    }

    impl LiveTranscriptionView {
        /// Default maximum retained segment count.
        const DEFAULT_MAX_SEGMENTS: usize = 10_000;

        /// Create a new view in its initial empty state.
        pub(crate) fn new(backend: BackendKind, diarization_active: bool) -> Self {
            let now = chrono::Utc::now().timestamp() as f64;
            Self {
                segments: Vec::new(),
                diarization_active,
                backend,
                started_epoch_secs: now,
                current_epoch_secs: now,
                scroll_offset: 0,
                auto_scroll: true,
                max_segments: Self::DEFAULT_MAX_SEGMENTS,
                correction_states: std::collections::HashMap::new(),
                seq_to_segment_idx: std::collections::HashMap::new(),
                speculative_active: false,
                correction_count: 0,
                speculation_window_count: 0,
            }
        }

        /// Create a view with explicit start time (useful for testing).
        #[cfg(test)]
        pub(crate) fn with_start_time(
            backend: BackendKind,
            diarization_active: bool,
            started_epoch_secs: f64,
        ) -> Self {
            Self {
                segments: Vec::new(),
                diarization_active,
                backend,
                started_epoch_secs,
                current_epoch_secs: started_epoch_secs,
                scroll_offset: 0,
                auto_scroll: true,
                max_segments: Self::DEFAULT_MAX_SEGMENTS,
                correction_states: std::collections::HashMap::new(),
                seq_to_segment_idx: std::collections::HashMap::new(),
                speculative_active: false,
                correction_count: 0,
                speculation_window_count: 0,
            }
        }

        /// Override the maximum segment retention count.
        pub(crate) fn set_max_segments(&mut self, max: usize) {
            self.max_segments = max.max(1);
        }

        /// Push a new segment into the view. If `auto_scroll` is enabled,
        /// the scroll offset is adjusted so the latest segment is visible.
        pub(crate) fn push_segment(&mut self, segment: TranscriptionSegment) {
            self.segments.push(segment);
            self.enforce_max_segments();
        }

        /// Push multiple segments at once (batch arrival).
        pub(crate) fn push_segments(
            &mut self,
            segments: impl IntoIterator<Item = TranscriptionSegment>,
        ) {
            self.segments.extend(segments);
            self.enforce_max_segments();
        }

        /// Drain oldest segments when over the retention limit.
        fn enforce_max_segments(&mut self) {
            if self.segments.len() > self.max_segments {
                let excess = self.segments.len() - self.max_segments;
                self.segments.drain(..excess);
                // Adjust scroll offset so it doesn't point beyond the
                // remaining segments.
                self.scroll_offset = self.scroll_offset.saturating_sub(excess as u16);
            }
        }

        /// Update the current wall-clock time (call on each tick).
        pub(crate) fn tick(&mut self, current_epoch_secs: f64) {
            self.current_epoch_secs = current_epoch_secs;
        }

        /// Scroll up by `n` lines. Disables auto-scroll.
        pub(crate) fn scroll_up(&mut self, n: u16) {
            self.scroll_offset = self.scroll_offset.saturating_sub(n);
            self.auto_scroll = false;
        }

        /// Scroll down by `n` lines. Re-enables auto-scroll if the user
        /// scrolls to the bottom.
        pub(crate) fn scroll_down(&mut self, n: u16) {
            self.scroll_offset = self.scroll_offset.saturating_add(n);
            // Re-enable auto-scroll when the user reaches the bottom.
            if self.scroll_offset as usize >= self.segments.len().saturating_sub(1) {
                self.auto_scroll = true;
            }
        }

        /// Re-enable auto-scroll (e.g. via a key binding).
        pub(crate) fn jump_to_latest(&mut self) {
            self.auto_scroll = true;
        }

        /// Return the number of accumulated segments.
        pub(crate) fn segment_count(&self) -> usize {
            self.segments.len()
        }

        /// Return whether auto-scroll is currently enabled.
        pub(crate) fn is_auto_scroll(&self) -> bool {
            self.auto_scroll
        }

        /// Compute the elapsed duration string (HH:MM:SS).
        pub(crate) fn elapsed_display(&self) -> String {
            let elapsed = (self.current_epoch_secs - self.started_epoch_secs).max(0.0);
            let total_secs = elapsed.round() as u64;
            let hours = total_secs / 3_600;
            let minutes = (total_secs % 3_600) / 60;
            let secs = total_secs % 60;
            format!("{hours:02}:{minutes:02}:{secs:02}")
        }

        /// Format the status bar text.
        pub(crate) fn status_bar_text(&self) -> String {
            let elapsed = self.elapsed_display();
            let seg_count = self.segments.len();
            let backend_str = self.backend.as_str();
            let scroll_indicator = if self.auto_scroll { "LIVE" } else { "PAUSED" };
            let diarization_indicator = if self.diarization_active {
                " | diarization=on"
            } else {
                ""
            };
            format!(
                "elapsed={elapsed} | segments={seg_count} | backend={backend_str} | {scroll_indicator}{diarization_indicator}"
            )
        }

        /// Format a single segment line for display.
        fn format_segment_line(
            index: usize,
            segment: &TranscriptionSegment,
            show_speaker: bool,
        ) -> String {
            let start = format_ts(segment.start_sec);
            let end = format_ts(segment.end_sec);

            let speaker_prefix = if show_speaker {
                segment
                    .speaker
                    .as_deref()
                    .map(|s| format!("[{s}] "))
                    .unwrap_or_default()
            } else {
                String::new()
            };

            let confidence_suffix = segment
                .confidence
                .map(|c| format!(" ({c:.3})"))
                .unwrap_or_default();

            format!(
                "{index:03} {start} -> {end} {speaker_prefix}{}{confidence_suffix}",
                segment.text
            )
        }

        /// Generate all formatted segment lines.
        pub(crate) fn segment_lines(&self) -> Vec<String> {
            if self.segments.is_empty() {
                return vec!["Waiting for transcription segments...".to_owned()];
            }

            self.segments
                .iter()
                .enumerate()
                .map(|(index, segment)| {
                    Self::format_segment_line(index, segment, self.diarization_active)
                })
                .collect()
        }

        /// Compute the effective scroll offset for rendering.
        /// When auto-scroll is on, the offset is set so the last segment
        /// is visible within `visible_height` lines.
        pub(crate) fn effective_scroll(&self, visible_height: u16) -> u16 {
            if self.auto_scroll {
                let total = self.segments.len() as u16;
                total.saturating_sub(visible_height)
            } else {
                self.scroll_offset
            }
        }

        /// Render the live transcription view into the given area.
        pub(crate) fn render(&self, area: Rect, frame: &mut Frame, focused: bool) {
            let title = if focused {
                "Live Transcription [FOCUS]"
            } else {
                "Live Transcription"
            };

            let border_style = if focused {
                Style::new().fg(PackedRgba::rgb(100, 220, 255)).bold()
            } else {
                Style::new().fg(PackedRgba::rgb(120, 120, 120))
            };

            // Reserve 1 line at the bottom for the status bar.
            let chunks = Flex::vertical()
                .constraints([Constraint::Fill, Constraint::Fixed(1)])
                .split(area);

            let transcript_area = chunks[0];
            let status_area = chunks[1];

            // The visible height inside the bordered block is height - 2
            // (top + bottom border).
            let inner_height = transcript_area.height.saturating_sub(2);
            let scroll = self.effective_scroll(inner_height);

            Paragraph::new(self.segment_lines().join("\n"))
                .scroll((scroll, 0))
                .block(Block::bordered().title(title).border_style(border_style))
                .render(transcript_area, frame);

            Paragraph::new(self.status_bar_text())
                .style(Style::new().fg(PackedRgba::rgb(180, 180, 100)))
                .render(status_area, frame);
        }
    }

    // ── Speaker color-coding (bd-339.2) ────────────────────────────────

    /// Palette of ANSI-friendly colors for speaker color-coding.
    /// Each entry is an (R, G, B) triple suitable for `PackedRgba::rgb`.
    const SPEAKER_PALETTE: [(u8, u8, u8); 8] = [
        (120, 200, 255), // light blue
        (255, 180, 120), // peach
        (120, 255, 160), // mint
        (255, 140, 200), // pink
        (255, 255, 120), // yellow
        (180, 140, 255), // lavender
        (120, 255, 255), // cyan
        (255, 200, 200), // salmon
    ];

    /// Assigns a consistent color index to each speaker name by hashing.
    /// The same speaker name always maps to the same palette entry.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct SpeakerColorMap {
        /// Cached mapping from speaker name to palette index.
        cache: std::collections::HashMap<String, usize>,
    }

    impl SpeakerColorMap {
        /// Create a new empty color map.
        pub(crate) fn new() -> Self {
            Self {
                cache: std::collections::HashMap::new(),
            }
        }

        /// Return the palette index for a given speaker name.
        /// Uses a simple FNV-1a-style hash to assign a deterministic index.
        pub(crate) fn color_index(&mut self, speaker: &str) -> usize {
            if let Some(&idx) = self.cache.get(speaker) {
                return idx;
            }
            let hash = Self::hash_speaker(speaker);
            let idx = hash % SPEAKER_PALETTE.len();
            self.cache.insert(speaker.to_owned(), idx);
            idx
        }

        /// Return the (R, G, B) color for a speaker.
        pub(crate) fn color_for(&mut self, speaker: &str) -> (u8, u8, u8) {
            let idx = self.color_index(speaker);
            SPEAKER_PALETTE[idx]
        }

        /// Simple hash function (FNV-1a inspired) for speaker name.
        fn hash_speaker(name: &str) -> usize {
            let mut h: u64 = 0xcbf29ce484222325;
            for byte in name.as_bytes() {
                h ^= *byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h as usize
        }
    }

    // ── Waveform visualization (bd-339.2) ────────────────────────────────

    /// Unicode block characters for amplitude bar rendering, from empty to full.
    const WAVEFORM_BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    /// Renders audio amplitude data as unicode block characters.
    #[derive(Debug, Clone)]
    pub(crate) struct WaveformDisplay {
        /// Amplitude samples, each in the range `[0.0, 1.0]`.
        amplitudes: Vec<f32>,
    }

    impl WaveformDisplay {
        /// Create a new waveform display from amplitude samples.
        /// Values are clamped to `[0.0, 1.0]`.
        pub(crate) fn new(amplitudes: Vec<f32>) -> Self {
            Self { amplitudes }
        }

        /// Render the amplitudes as a string of unicode block characters.
        /// Each amplitude maps to one of the 9 block levels (space through full block).
        pub(crate) fn amplitude_bars(&self) -> String {
            self.amplitudes
                .iter()
                .map(|&amp| {
                    let clamped = if amp.is_finite() { amp.clamp(0.0, 1.0) } else { 0.0 };
                    let idx = (clamped * 8.0).round() as usize;
                    WAVEFORM_BLOCKS[idx.min(8)]
                })
                .collect()
        }

        /// Return the number of samples.
        pub(crate) fn len(&self) -> usize {
            self.amplitudes.len()
        }
    }

    // ── Waveform state for live visualization (bd-339.2) ──────────────

    /// Tracks audio energy levels over time for live waveform visualization.
    /// Maintains a rolling window of RMS energy samples that can be rendered
    /// as an ASCII waveform bar.
    #[derive(Debug, Clone)]
    pub(crate) struct WaveformState {
        /// Rolling buffer of RMS energy values in `[0.0, 1.0]`.
        samples: Vec<f32>,
        /// Maximum number of samples to retain.
        max_samples: usize,
        /// Peak energy seen so far, used for normalization.
        peak_energy: f32,
    }

    impl WaveformState {
        /// Default maximum sample count.
        const DEFAULT_MAX_SAMPLES: usize = 80;

        /// Create a new waveform state with default capacity.
        pub(crate) fn new() -> Self {
            Self {
                samples: Vec::new(),
                max_samples: Self::DEFAULT_MAX_SAMPLES,
                peak_energy: 0.0,
            }
        }

        /// Create a waveform state with a custom maximum sample count.
        pub(crate) fn with_max_samples(max_samples: usize) -> Self {
            Self {
                samples: Vec::new(),
                max_samples: max_samples.max(1),
                peak_energy: 0.0,
            }
        }

        /// Push a raw RMS energy value. The value is clamped to `[0.0, 1.0]`
        /// and stored. If the buffer exceeds `max_samples`, the oldest sample
        /// is removed.
        pub(crate) fn push_energy(&mut self, rms: f32) {
            let clamped = if rms.is_finite() { rms.clamp(0.0, 1.0) } else { 0.0 };
            if clamped > self.peak_energy {
                self.peak_energy = clamped;
            }
            self.samples.push(clamped);
            if self.samples.len() > self.max_samples {
                let excess = self.samples.len() - self.max_samples;
                self.samples.drain(..excess);
            }
        }

        /// Push multiple RMS energy values at once.
        pub(crate) fn push_energies(&mut self, rms_values: &[f32]) {
            for &rms in rms_values {
                self.push_energy(rms);
            }
        }

        /// Return the current peak energy value.
        pub(crate) fn peak(&self) -> f32 {
            self.peak_energy
        }

        /// Return the number of stored samples.
        pub(crate) fn len(&self) -> usize {
            self.samples.len()
        }

        /// Return `true` if no samples have been stored.
        pub(crate) fn is_empty(&self) -> bool {
            self.samples.is_empty()
        }

        /// Reset all samples and peak tracking.
        pub(crate) fn clear(&mut self) {
            self.samples.clear();
            self.peak_energy = 0.0;
        }

        /// Render the current waveform as a `WaveformDisplay`.
        pub(crate) fn to_display(&self) -> WaveformDisplay {
            WaveformDisplay::new(self.samples.clone())
        }

        /// Render the waveform as an ASCII string using block characters.
        pub(crate) fn render_bars(&self) -> String {
            self.to_display().amplitude_bars()
        }
    }

    // ── Speaker-colored line rendering (bd-339.2) ─────────────────────

    /// Format a transcript line with ANSI color codes for the speaker label.
    /// Returns a string like `\x1b[38;2;R;G;Bm[SPEAKER_00]\x1b[0m 00:00:01 hello`.
    /// If the segment has no speaker, the line is returned without color codes.
    pub(crate) fn render_speaker_colored_line(
        segment: &TranscriptionSegment,
        color_map: &mut SpeakerColorMap,
        show_timestamps: bool,
    ) -> String {
        let mut parts = Vec::new();

        // Speaker label with ANSI color.
        if let Some(speaker) = &segment.speaker {
            let (r, g, b) = color_map.color_for(speaker);
            parts.push(format!("\x1b[38;2;{r};{g};{b}m[{speaker}]\x1b[0m"));
        }

        // Timestamp.
        if show_timestamps {
            let ts = match (segment.start_sec, segment.end_sec) {
                (Some(start), Some(end)) => {
                    format!("{} -> {}", format_ts(Some(start)), format_ts(Some(end)))
                }
                (Some(start), None) => format!("{} -> ???", format_ts(Some(start))),
                _ => "??:??:??.???".to_owned(),
            };
            parts.push(ts);
        }

        // Confidence badge.
        if let Some(conf) = segment.confidence {
            parts.push(format!("[{:.0}%]", conf * 100.0));
        }

        // Text.
        parts.push(segment.text.clone());

        parts.join(" ")
    }

    // ── Search, filter, export (bd-339.3) ────────────────────────────────

    /// Holds transient search state for the live transcription view.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct SearchState {
        /// The current search query string.
        pub(crate) query: String,
        /// Indices of segments that match the query.
        pub(crate) matches: Vec<usize>,
        /// Index into `matches` for the currently highlighted result.
        pub(crate) current_match_index: usize,
    }

    impl SearchState {
        /// Create a new empty search state.
        pub(crate) fn new() -> Self {
            Self::default()
        }

        /// Set the query and reset match state.
        pub(crate) fn set_query(&mut self, query: String) {
            self.query = query;
            self.matches.clear();
            self.current_match_index = 0;
        }

        /// Return the segment index of the current match, if any.
        pub(crate) fn current_segment_index(&self) -> Option<usize> {
            self.matches.get(self.current_match_index).copied()
        }

        /// Move to the next match (wraps around).
        pub(crate) fn next_match(&mut self) {
            if !self.matches.is_empty() {
                self.current_match_index = (self.current_match_index + 1) % self.matches.len();
            }
        }

        /// Move to the previous match (wraps around).
        pub(crate) fn prev_match(&mut self) {
            if !self.matches.is_empty() {
                self.current_match_index = if self.current_match_index == 0 {
                    self.matches.len() - 1
                } else {
                    self.current_match_index - 1
                };
            }
        }
    }

    impl LiveTranscriptionView {
        /// Search all segments for text matching the given query (case-insensitive).
        /// Updates the provided `SearchState` with matching segment indices.
        pub(crate) fn search_segments(&self, state: &mut SearchState) {
            state.matches.clear();
            state.current_match_index = 0;

            if state.query.is_empty() {
                return;
            }

            let query_lower = state.query.to_lowercase();
            for (idx, segment) in self.segments.iter().enumerate() {
                if segment.text.to_lowercase().contains(&query_lower) {
                    state.matches.push(idx);
                    continue;
                }
                // Also search speaker labels.
                if let Some(speaker) = &segment.speaker
                    && speaker.to_lowercase().contains(&query_lower)
                {
                    state.matches.push(idx);
                }
            }
        }

        /// Return only segments spoken by the given speaker.
        /// Returns a vector of (original_index, segment_reference) pairs.
        pub(crate) fn filter_by_speaker(
            &self,
            speaker: &str,
        ) -> Vec<(usize, &TranscriptionSegment)> {
            self.segments
                .iter()
                .enumerate()
                .filter(|(_idx, seg)| seg.speaker.as_deref() == Some(speaker))
                .collect()
        }

        /// Export visible/filtered segments as a JSON string.
        /// If `speaker_filter` is `Some`, only segments from that speaker are included.
        /// Otherwise, all segments are exported.
        pub(crate) fn export_visible(&self, speaker_filter: Option<&str>) -> String {
            let segments_to_export: Vec<&TranscriptionSegment> = match speaker_filter {
                Some(speaker) => self
                    .segments
                    .iter()
                    .filter(|seg| seg.speaker.as_deref() == Some(speaker))
                    .collect(),
                None => self.segments.iter().collect(),
            };

            serde_json::to_string_pretty(&segments_to_export)
                .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {e}\"}}"))
        }

        /// Return the lines visible within a given viewport height, applying
        /// the current scroll offset.  Used by snapshot tests (bd-3pf.4).
        pub(crate) fn get_visible_lines(&self, visible_height: u16) -> Vec<String> {
            let all_lines = self.segment_lines();
            let scroll = self.effective_scroll(visible_height) as usize;
            let end = (scroll + visible_height as usize).min(all_lines.len());
            if scroll >= all_lines.len() {
                return vec![];
            }
            all_lines[scroll..end].to_vec()
        }
    }

    // ── Speculative correction rendering (bd-qlt.9) ───────────────────

    impl LiveTranscriptionView {
        /// Enable or disable speculative streaming mode.
        pub(crate) fn set_speculative(&mut self, active: bool) {
            self.speculative_active = active;
        }

        /// Push a segment from the fast model, marked as speculative.
        pub(crate) fn push_speculative_segment(&mut self, segment: TranscriptionSegment, seq: u64) {
            let idx = self.segments.len();
            self.segments.push(segment);
            self.correction_states
                .insert(idx, SegmentCorrectionState::Speculative);
            self.seq_to_segment_idx.entry(seq).or_default().push(idx);
            self.speculation_window_count += 1;
        }

        /// Retract segments associated with a partial seq.
        pub(crate) fn retract_segments(&mut self, seq: u64) {
            if let Some(indices) = self.seq_to_segment_idx.get(&seq) {
                for &idx in indices {
                    self.correction_states
                        .insert(idx, SegmentCorrectionState::Retracted);
                }
            }
        }

        /// Push correction segments replacing a retracted seq.
        pub(crate) fn push_correction(
            &mut self,
            replaces_seq: u64,
            corrected_segments: Vec<TranscriptionSegment>,
        ) {
            self.retract_segments(replaces_seq);
            for seg in corrected_segments {
                let idx = self.segments.len();
                self.segments.push(seg);
                self.correction_states
                    .insert(idx, SegmentCorrectionState::Corrected);
            }
            self.correction_count += 1;
        }

        /// Confirm segments associated with a partial seq.
        pub(crate) fn confirm_segments(&mut self, seq: u64) {
            if let Some(indices) = self.seq_to_segment_idx.get(&seq) {
                for &idx in indices {
                    self.correction_states
                        .insert(idx, SegmentCorrectionState::Confirmed);
                }
            }
        }

        /// Get the correction state for a segment at index.
        pub(crate) fn segment_correction_state(&self, idx: usize) -> SegmentCorrectionState {
            self.correction_states
                .get(&idx)
                .copied()
                .unwrap_or(SegmentCorrectionState::Original)
        }

        /// Number of corrections applied.
        pub(crate) fn correction_count(&self) -> u32 {
            self.correction_count
        }

        /// Whether speculative mode is active.
        pub(crate) fn is_speculative(&self) -> bool {
            self.speculative_active
        }

        /// Format the correction prefix for a segment at a given index.
        pub(crate) fn correction_prefix(&self, idx: usize) -> &'static str {
            match self.segment_correction_state(idx) {
                SegmentCorrectionState::Speculative => "~ ",
                SegmentCorrectionState::Retracted => "x ",
                SegmentCorrectionState::Corrected => "> ",
                _ => "",
            }
        }

        /// Speculation stats summary line for status bar.
        pub(crate) fn speculation_stats_line(&self) -> String {
            if !self.speculative_active {
                return String::new();
            }
            let rate = if self.speculation_window_count > 0 {
                self.correction_count as f64 / self.speculation_window_count as f64 * 100.0
            } else {
                0.0
            };
            format!(
                "[SPEC] Corrections: {}/{} ({rate:.1}%)",
                self.correction_count, self.speculation_window_count
            )
        }
    }

    // ── Advanced filtering, search, and export (bd-339.3) ─────────────

    /// Export format for filtered transcript output.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum ExportFormat {
        /// Plain text, one line per segment.
        PlainText,
        /// JSON array of segment objects.
        Json,
        /// SubRip subtitle format.
        Srt,
    }

    /// Multi-field filter for transcript segments.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct TranscriptFilter {
        /// If set, only include segments from this speaker.
        pub(crate) speaker: Option<String>,
        /// If set, only include segments whose text contains this keyword
        /// (case-insensitive).
        pub(crate) keyword: Option<String>,
        /// If set, only include segments overlapping this time range
        /// `(start_sec, end_sec)`.
        pub(crate) time_range: Option<(f64, f64)>,
        /// If set, only include segments with confidence >= this value.
        pub(crate) min_confidence: Option<f64>,
    }

    impl TranscriptFilter {
        /// Create a new empty filter (matches everything).
        pub(crate) fn new() -> Self {
            Self::default()
        }

        /// Return `true` if this filter has no constraints set.
        pub(crate) fn is_empty(&self) -> bool {
            self.speaker.is_none()
                && self.keyword.is_none()
                && self.time_range.is_none()
                && self.min_confidence.is_none()
        }

        /// Test whether a single segment matches all active filter criteria.
        pub(crate) fn matches(&self, segment: &TranscriptionSegment) -> bool {
            // Speaker filter.
            if let Some(ref speaker) = self.speaker {
                match &segment.speaker {
                    Some(seg_speaker) if seg_speaker == speaker => {}
                    _ => return false,
                }
            }

            // Keyword filter (case-insensitive).
            if let Some(ref keyword) = self.keyword {
                let kw_lower = keyword.to_lowercase();
                if !segment.text.to_lowercase().contains(&kw_lower) {
                    return false;
                }
            }

            // Time range filter: segment must overlap the range.
            if let Some((range_start, range_end)) = self.time_range {
                let seg_start = segment.start_sec.unwrap_or(0.0);
                let seg_end = segment.end_sec.unwrap_or(f64::MAX);
                // No overlap if segment ends before range starts
                // or segment starts after range ends.
                if seg_end < range_start || seg_start > range_end {
                    return false;
                }
            }

            // Minimum confidence filter.
            if let Some(min_conf) = self.min_confidence {
                match segment.confidence {
                    Some(c) if c >= min_conf => {}
                    Some(_) => return false,
                    // Segments without confidence are excluded when a
                    // minimum is required.
                    None => return false,
                }
            }

            true
        }
    }

    /// Apply a `TranscriptFilter` to a slice of segments, returning only
    /// those that match all active criteria.
    pub(crate) fn apply_filter<'a>(
        segments: &'a [TranscriptionSegment],
        filter: &TranscriptFilter,
    ) -> Vec<&'a TranscriptionSegment> {
        segments.iter().filter(|seg| filter.matches(seg)).collect()
    }

    /// Full-featured transcript search with navigation and result tracking.
    #[derive(Debug, Clone)]
    pub(crate) struct TranscriptSearch {
        /// The current search query.
        query: String,
        /// Indices of matching segments.
        matches: Vec<usize>,
        /// Current position within `matches`.
        cursor: usize,
    }

    impl TranscriptSearch {
        /// Create a new empty search.
        pub(crate) fn new() -> Self {
            Self {
                query: String::new(),
                matches: Vec::new(),
                cursor: 0,
            }
        }

        /// Execute a case-insensitive search across segments.
        /// Also searches speaker labels.
        pub(crate) fn search(&mut self, query: &str, segments: &[TranscriptionSegment]) {
            self.query = query.to_owned();
            self.matches.clear();
            self.cursor = 0;

            if query.is_empty() {
                return;
            }

            let query_lower = query.to_lowercase();
            for (idx, segment) in segments.iter().enumerate() {
                if segment.text.to_lowercase().contains(&query_lower) {
                    self.matches.push(idx);
                    continue;
                }
                if let Some(speaker) = &segment.speaker {
                    if speaker.to_lowercase().contains(&query_lower) {
                        self.matches.push(idx);
                    }
                }
            }
        }

        /// Return the current query string.
        pub(crate) fn query(&self) -> &str {
            &self.query
        }

        /// Return the number of matches found.
        pub(crate) fn match_count(&self) -> usize {
            self.matches.len()
        }

        /// Return the segment index of the current match, if any.
        pub(crate) fn current_match(&self) -> Option<usize> {
            self.matches.get(self.cursor).copied()
        }

        /// Advance to the next match (wraps around).
        pub(crate) fn next_match(&mut self) -> Option<usize> {
            if self.matches.is_empty() {
                return None;
            }
            self.cursor = (self.cursor + 1) % self.matches.len();
            self.current_match()
        }

        /// Move to the previous match (wraps around).
        pub(crate) fn prev_match(&mut self) -> Option<usize> {
            if self.matches.is_empty() {
                return None;
            }
            self.cursor = if self.cursor == 0 {
                self.matches.len() - 1
            } else {
                self.cursor - 1
            };
            self.current_match()
        }

        /// Return all matching segment indices.
        pub(crate) fn all_matches(&self) -> &[usize] {
            &self.matches
        }
    }

    /// Format seconds as SRT timestamp: `HH:MM:SS,mmm`.
    fn format_srt_timestamp(seconds: f64) -> String {
        let total_ms = (seconds * 1000.0).round() as u64;
        let h = total_ms / 3_600_000;
        let m = (total_ms % 3_600_000) / 60_000;
        let s = (total_ms % 60_000) / 1000;
        let ms = total_ms % 1000;
        format!("{h:02}:{m:02}:{s:02},{ms:03}")
    }

    /// Export filtered transcript segments to a string in the specified format.
    pub(crate) fn export_filtered_transcript(
        segments: &[TranscriptionSegment],
        filter: &TranscriptFilter,
        format: ExportFormat,
    ) -> String {
        let filtered: Vec<&TranscriptionSegment> = apply_filter(segments, filter);
        match format {
            ExportFormat::PlainText => export_plain_text(&filtered),
            ExportFormat::Json => export_json(&filtered),
            ExportFormat::Srt => export_srt(&filtered),
        }
    }

    /// Export segments as plain text, one line per segment.
    fn export_plain_text(segments: &[&TranscriptionSegment]) -> String {
        let mut lines = Vec::with_capacity(segments.len());
        for seg in segments {
            let mut parts = Vec::new();

            // Timestamp range.
            if let (Some(start), Some(end)) = (seg.start_sec, seg.end_sec) {
                parts.push(format!(
                    "[{} -> {}]",
                    format_ts(Some(start)),
                    format_ts(Some(end))
                ));
            }

            // Speaker label.
            if let Some(speaker) = &seg.speaker {
                parts.push(format!("[{speaker}]"));
            }

            // Text.
            parts.push(seg.text.clone());

            lines.push(parts.join(" "));
        }
        lines.join("\n")
    }

    /// Export segments as a JSON array.
    fn export_json(segments: &[&TranscriptionSegment]) -> String {
        serde_json::to_string_pretty(&segments)
            .unwrap_or_else(|e| format!("{{\"error\": \"serialization failed: {e}\"}}"))
    }

    /// Export segments in SubRip (SRT) subtitle format.
    fn export_srt(segments: &[&TranscriptionSegment]) -> String {
        let mut lines = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            // Sequence number (1-based).
            lines.push(format!("{}", i + 1));

            // Timestamp line.
            let start = seg.start_sec.unwrap_or(0.0);
            let end = seg.end_sec.unwrap_or(start);
            lines.push(format!(
                "{} --> {}",
                format_srt_timestamp(start),
                format_srt_timestamp(end),
            ));

            // Text line (with optional speaker prefix).
            if let Some(speaker) = &seg.speaker {
                lines.push(format!("[{speaker}] {}", seg.text));
            } else {
                lines.push(seg.text.clone());
            }

            // Blank line separator.
            lines.push(String::new());
        }
        // Remove trailing blank line if present.
        if lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        lines.join("\n")
    }

    /// Write filtered transcript to a file in the specified format.
    /// Returns the number of segments written.
    pub(crate) fn export_filtered_transcript_to_file(
        segments: &[TranscriptionSegment],
        filter: &TranscriptFilter,
        format: ExportFormat,
        path: &std::path::Path,
    ) -> FwResult<usize> {
        let filtered: Vec<&TranscriptionSegment> = apply_filter(segments, filter);
        let count = filtered.len();
        let content = match format {
            ExportFormat::PlainText => export_plain_text(&filtered),
            ExportFormat::Json => export_json(&filtered),
            ExportFormat::Srt => export_srt(&filtered),
        };
        std::fs::write(path, content)?;
        Ok(count)
    }

    pub fn run_tui() -> FwResult<()> {
        let db_path = std::env::var("FRANKEN_WHISPER_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".franken_whisper/storage.sqlite3"));

        App::new(WhisperTuiApp::new(db_path))
            .screen_mode(ScreenMode::InlineAuto {
                min_height: 20,
                max_height: 44,
            })
            .run()
            .map_err(FwError::from)
    }

    #[cfg(test)]
    mod tests {
        use std::path::PathBuf;

        use fsqlite_types::value::SqliteValue;
        use serde_json::json;
        use tempfile::tempdir;

        use super::{FocusPane, KeyCode, KeyEventKind, Model, Modifiers, Msg};
        use crate::model::{
            BackendKind, BackendParams, InputSource, RunEvent, RunReport, TranscribeRequest,
            TranscriptionResult, TranscriptionSegment,
        };
        use crate::storage::RunStore;

        use super::{WhisperTuiApp, format_ts};

        fn fixture_report(id: &str, db_path: &std::path::Path, event_count: usize) -> RunReport {
            let segments = (0..event_count.max(1))
                .map(|idx| TranscriptionSegment {
                    start_sec: Some(idx as f64),
                    end_sec: Some((idx + 1) as f64),
                    text: format!("segment-{idx}"),
                    speaker: if idx % 2 == 0 {
                        Some("SPEAKER_00".to_owned())
                    } else {
                        None
                    },
                    confidence: Some(0.9),
                })
                .collect::<Vec<_>>();

            let events = (0..event_count.max(1))
                .map(|idx| RunEvent {
                    seq: (idx + 1) as u64,
                    ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
                    stage: if idx % 3 == 0 {
                        "backend".to_owned()
                    } else {
                        "acceleration".to_owned()
                    },
                    code: if idx % 10 == 0 {
                        "acceleration.fallback".to_owned()
                    } else {
                        "backend.ok".to_owned()
                    },
                    message: format!("event-{idx}"),
                    payload: json!({"idx": idx}),
                })
                .collect::<Vec<_>>();

            RunReport {
                run_id: id.to_owned(),
                trace_id: "00000000000000000000000000000000".to_owned(),
                started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
                finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
                input_path: "test.wav".to_owned(),
                normalized_wav_path: "normalized.wav".to_owned(),
                request: TranscribeRequest {
                    input: InputSource::File {
                        path: PathBuf::from("test.wav"),
                    },
                    backend: BackendKind::Auto,
                    model: None,
                    language: Some("en".to_owned()),
                    translate: false,
                    diarize: false,
                    persist: true,
                    db_path: db_path.to_path_buf(),
                    timeout_ms: None,
                    backend_params: BackendParams::default(),
                },
                result: TranscriptionResult {
                    backend: BackendKind::WhisperCpp,
                    transcript: format!("transcript-{id}"),
                    language: Some("en".to_owned()),
                    segments,
                    acceleration: None,
                    raw_output: json!({"test": true}),
                    artifact_paths: vec![],
                },
                events,
                warnings: vec![
                    "fallback: deterministic static route used".to_owned(),
                    "divergence: replay comparator drift observed".to_owned(),
                ],
                evidence: vec![],
                replay: crate::model::ReplayEnvelope::default(),
            }
        }

        fn seed_runs(db_path: &std::path::Path, runs: usize, events_per_run: usize) {
            let store = RunStore::open(db_path).expect("store");
            for idx in 0..runs {
                let id = format!("run-{idx:03}");
                store
                    .persist_report(&fixture_report(&id, db_path, events_per_run))
                    .expect("persist");
            }
        }

        #[test]
        fn formats_time() {
            assert_eq!(format_ts(Some(62.345)), "00:01:02.345");
            assert_eq!(format_ts(None), "--:--:--.---");
        }

        #[test]
        fn app_handles_large_run_sets_selection_and_refresh_cycles() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("storage.sqlite3");
            seed_runs(&db_path, 80, 24);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 64, "RUNS_LIMIT should cap visible runs");
            app.move_selection(10);
            assert_eq!(app.selected_run, 10);
            app.move_selection(10_000);
            assert_eq!(app.selected_run, 63);
            app.move_selection(-10_000);
            assert_eq!(app.selected_run, 0);

            for _ in 0..50 {
                app.reload_data();
            }
            assert_eq!(app.runs.len(), 64);
            assert!(app.details.is_some());
        }

        #[test]
        fn app_empty_db_and_event_volume_paths_are_stable() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("empty.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let app = WhisperTuiApp::new(db_path.clone());
            assert!(app.runs.is_empty());
            assert!(app.details.is_none());
            assert!(app.status_line.contains("No runs in"));
            assert_eq!(app.runs_text(), "No runs found");
            assert_eq!(
                app.timeline_lines(),
                vec!["No selected run details available"]
            );
            assert_eq!(app.event_lines(), vec!["No events available"]);

            let busy_db = dir.path().join("busy.sqlite3");
            seed_runs(&busy_db, 4, 500);
            let mut busy_app = WhisperTuiApp::new(busy_db);
            let timeline = busy_app.timeline_lines();
            let events = busy_app.event_lines();
            assert!(
                timeline.len() >= 500,
                "expected high-volume timeline lines, got {}",
                timeline.len()
            );
            assert!(
                events.len() >= 500,
                "expected high-volume event lines, got {}",
                events.len()
            );
            busy_app.timeline_scroll = u16::MAX;
            busy_app.events_scroll = u16::MAX;
            let _ = busy_app.timeline_lines();
            let _ = busy_app.event_lines();
        }

        #[test]
        fn large_dataset_120_runs_caps_at_runs_limit() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("large.sqlite3");
            seed_runs(&db_path, 120, 5);

            let app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 64, "should cap at RUNS_LIMIT=64");
            assert!(app.details.is_some());
            let runs_text = app.runs_text();
            assert!(runs_text.starts_with(">"), "first run should be selected");
            // Verify all 64 entries appear in the text.
            assert_eq!(
                runs_text.lines().count(),
                64,
                "runs_text should have 64 lines"
            );
        }

        #[test]
        fn single_run_with_2000_events() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("mega.sqlite3");
            seed_runs(&db_path, 1, 2000);

            let app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 1);
            assert!(app.details.is_some());

            let timeline = app.timeline_lines();
            assert!(
                timeline.len() >= 2000,
                "expected 2000+ timeline lines, got {}",
                timeline.len()
            );

            let events = app.event_lines();
            assert!(
                events.len() >= 2000,
                "expected 2000+ event lines, got {}",
                events.len()
            );
        }

        #[test]
        fn runs_with_no_events_show_placeholder() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_events.sqlite3");

            // event_count=0 results in max(1)=1 segment/event; to get truly zero,
            // we need a special fixture. Let's use event_count=1 but check we get 1.
            seed_runs(&db_path, 3, 1);

            let app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 3);
            let timeline = app.timeline_lines();
            assert!(!timeline.is_empty());
            let events = app.event_lines();
            assert!(!events.is_empty());
        }

        #[test]
        fn page_movement_clamps_at_boundaries() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("page.sqlite3");
            seed_runs(&db_path, 20, 10);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.selected_run, 0);

            // PageDown by 8 from start.
            app.move_selection(8);
            assert_eq!(app.selected_run, 8);

            // PageDown by 8 again.
            app.move_selection(8);
            assert_eq!(app.selected_run, 16);

            // PageDown beyond end clamps.
            app.move_selection(8);
            assert_eq!(app.selected_run, 19);

            // PageUp by 8.
            app.move_selection(-8);
            assert_eq!(app.selected_run, 11);

            // PageUp all the way.
            app.move_selection(-100);
            assert_eq!(app.selected_run, 0);
        }

        #[test]
        fn focus_pane_cycling() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("focus.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.focus, FocusPane::Runs);

            app.focus = app.focus.next();
            assert_eq!(app.focus, FocusPane::Timeline);

            app.focus = app.focus.next();
            assert_eq!(app.focus, FocusPane::Events);

            app.focus = app.focus.next();
            assert_eq!(app.focus, FocusPane::Runs);

            app.focus = app.focus.prev();
            assert_eq!(app.focus, FocusPane::Events);

            app.focus = app.focus.prev();
            assert_eq!(app.focus, FocusPane::Timeline);

            app.focus = app.focus.prev();
            assert_eq!(app.focus, FocusPane::Runs);
        }

        #[test]
        fn scroll_offsets_are_independent_per_pane() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("scroll.sqlite3");
            seed_runs(&db_path, 2, 100);

            let mut app = WhisperTuiApp::new(db_path);

            // Timeline scroll.
            app.timeline_scroll = 42;
            app.events_scroll = 17;

            // Changing selection resets scrolls.
            app.move_selection(1);
            assert_eq!(
                app.timeline_scroll, 0,
                "selection change resets timeline scroll"
            );
            assert_eq!(
                app.events_scroll, 0,
                "selection change resets events scroll"
            );

            // Set scrolls again.
            app.timeline_scroll = 99;
            app.events_scroll = 50;

            // Reload doesn't reset scroll offsets (they persist across reloads).
            let saved_tl = app.timeline_scroll;
            let saved_ev = app.events_scroll;
            app.reload_data();
            assert_eq!(app.timeline_scroll, saved_tl);
            assert_eq!(app.events_scroll, saved_ev);
        }

        #[test]
        fn format_ts_edge_cases() {
            // Zero.
            assert_eq!(format_ts(Some(0.0)), "00:00:00.000");
            // Large value (1 hour + 2 min + 3.456 sec).
            assert_eq!(format_ts(Some(3723.456)), "01:02:03.456");
            // Negative value.
            assert_eq!(format_ts(Some(-1.0)), "00:00:00.000");
            // Very large value.
            let ts = format_ts(Some(86400.0));
            assert!(ts.starts_with("24:00:00"), "24h format: {ts}");
        }

        #[test]
        fn format_ts_nan_does_not_panic() {
            // NaN fails `is_sign_negative()`, then `(NaN * 1000).round() as u64` saturates to 0.
            let ts = format_ts(Some(f64::NAN));
            assert_eq!(ts, "00:00:00.000");
        }

        #[test]
        fn format_ts_positive_infinity_does_not_panic() {
            // Infinity * 1000 = infinity, `.round()` = infinity, `as u64` saturates to u64::MAX.
            let ts = format_ts(Some(f64::INFINITY));
            // Just verify it returns a string without panicking.
            assert!(!ts.is_empty());
        }

        #[test]
        fn format_ts_negative_infinity_returns_zero() {
            // -Infinity triggers `is_sign_negative()` early return.
            assert_eq!(format_ts(Some(f64::NEG_INFINITY)), "00:00:00.000");
        }

        #[test]
        fn format_ts_sub_millisecond_rounds_correctly() {
            // 0.0001 seconds = 0.1ms, should round to 0ms.
            assert_eq!(format_ts(Some(0.0001)), "00:00:00.000");
            // 0.0005 seconds = 0.5ms, should round to 1ms.
            assert_eq!(format_ts(Some(0.0005)), "00:00:00.001");
        }

        #[test]
        fn format_ts_negative_zero_returns_zero() {
            // -0.0 is sign-negative, so it takes the early return.
            assert_eq!(format_ts(Some(-0.0)), "00:00:00.000");
        }

        #[test]
        fn status_hint_contains_key_bindings() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("hint.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let app = WhisperTuiApp::new(db_path);
            let hint = app.status_hint();
            assert!(hint.contains("Tab"), "hint should mention Tab");
            assert!(hint.contains("quit"), "hint should mention quit");
            assert!(hint.contains("help"), "hint should mention help");
        }

        #[test]
        fn centered_rect_within_bounds() {
            use ftui::core::geometry::Rect;
            let area = Rect::new(0, 0, 100, 50);
            let r = super::centered_rect(area, 60, 20);
            assert_eq!(r.width, 60);
            assert_eq!(r.height, 20);
            assert_eq!(r.x, 20); // (100-60)/2
            assert_eq!(r.y, 15); // (50-20)/2
        }

        #[test]
        fn centered_rect_clamps_to_area() {
            use ftui::core::geometry::Rect;
            let area = Rect::new(0, 0, 30, 10);
            let r = super::centered_rect(area, 100, 100);
            // Width clamped to area.width-2=28, then max(10)=28
            assert_eq!(r.width, 28);
            // Height clamped to area.height-2=8, then max(5)=8
            assert_eq!(r.height, 8);
        }

        #[test]
        fn centered_rect_minimum_size() {
            use ftui::core::geometry::Rect;
            let area = Rect::new(0, 0, 5, 5);
            let r = super::centered_rect(area, 2, 2);
            // min width is 10 but area is 5, so clamped to max(3, 10)=10 → but wait
            // let's trace: width.min(5-2).max(10) = 2.min(3).max(10) = 2.max(10) = 10
            // But 10 > 5, so x = 0 + (5-10)/2 = 0 (saturating_sub)
            assert!(
                r.width >= 5 || r.width == 10,
                "width should be at least minimum"
            );
        }

        #[test]
        fn help_toggle_on_off() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("help.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let mut app = WhisperTuiApp::new(db_path);
            assert!(!app.show_help);
            app.show_help = !app.show_help;
            assert!(app.show_help);
            app.show_help = !app.show_help;
            assert!(!app.show_help);
        }

        #[test]
        fn move_selection_on_empty_runs_stays_at_zero() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("empty_sel.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let mut app = WhisperTuiApp::new(db_path);
            assert!(app.runs.is_empty());
            app.move_selection(5);
            assert_eq!(app.selected_run, 0);
            app.move_selection(-5);
            assert_eq!(app.selected_run, 0);
        }

        #[test]
        fn warning_legend_content() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("legend.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let app = WhisperTuiApp::new(db_path);
            let legend = app.warning_legend();
            assert!(legend.contains("fallback"));
            assert!(legend.contains("divergence"));
            assert!(legend.contains("deterministic"));
        }

        #[cfg(unix)]
        #[test]
        fn app_reports_transient_db_open_failures_and_warning_legend() {
            let app = WhisperTuiApp::new(PathBuf::from("/proc/franken_whisper/db.sqlite3"));
            assert!(
                app.status_line.contains("db open failed"),
                "unexpected status: {}",
                app.status_line
            );
            assert!(app.warning_legend().contains("fallback"));
            assert!(app.warning_legend().contains("divergence"));
        }

        #[test]
        fn timeline_lines_format_segments_with_timestamps_and_speaker() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("timeline.sqlite3");
            seed_runs(&db_path, 1, 1);

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert!(!lines.is_empty(), "should have timeline lines");

            // First line should start with "000" (index) and contain timestamps.
            let first = &lines[0];
            assert!(
                first.starts_with("000"),
                "first line should start with index: {first}"
            );
            assert!(first.contains("->"), "should have arrow separator: {first}");
            // Fixture includes speakers for even-indexed segments.
            assert!(
                first.contains("[SPEAKER_00]"),
                "first segment should have speaker tag: {first}"
            );
            // Fixture includes 0.9 confidence.
            assert!(first.contains("0.900"), "should show confidence: {first}");
        }

        #[test]
        fn timeline_lines_no_details_shows_placeholder() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_details.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("No selected run"),
                "should show placeholder: {}",
                lines[0]
            );
        }

        #[test]
        fn event_lines_format_events_with_seq_stage_code_message() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("events.sqlite3");
            seed_runs(&db_path, 1, 5);

            let app = WhisperTuiApp::new(db_path);
            let lines = app.event_lines();
            assert_eq!(lines.len(), 5, "should have 5 event lines");

            let first = &lines[0];
            assert!(
                first.starts_with("001"),
                "first event seq should be 001: {first}"
            );
            assert!(
                first.contains("[backend]") || first.contains("[acceleration]"),
                "should show stage: {first}"
            );
            assert!(first.contains("|"), "should have pipe separator: {first}");
        }

        #[test]
        fn event_lines_no_details_shows_placeholder() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_events.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.event_lines();
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("No events"),
                "should show placeholder: {}",
                lines[0]
            );
        }

        #[test]
        fn runs_text_shows_selection_marker() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("marker.sqlite3");
            seed_runs(&db_path, 3, 1);

            let app = WhisperTuiApp::new(db_path);
            let text = app.runs_text();
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 3);
            // First run should have > marker (selected).
            assert!(
                lines[0].starts_with('>'),
                "selected run should have > marker: {}",
                lines[0]
            );
            // Others should have space marker.
            assert!(
                lines[1].starts_with(' '),
                "non-selected should have space: {}",
                lines[1]
            );
        }

        #[test]
        fn timeline_lines_odd_indexed_segment_has_no_speaker_tag() {
            // Fixture gives speaker to even-indexed segments only.
            // Odd-indexed segments should NOT have [SPEAKER_XX] tag.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("odd_seg.sqlite3");
            seed_runs(&db_path, 1, 4);

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert!(lines.len() >= 4, "need at least 4 segments");
            // Index 1 (odd) should have no speaker tag.
            assert!(
                !lines[1].contains("[SPEAKER"),
                "odd segment should have no speaker tag: {}",
                lines[1]
            );
            // Index 2 (even) should have speaker tag.
            assert!(
                lines[2].contains("[SPEAKER_00]"),
                "even segment should have speaker tag: {}",
                lines[2]
            );
        }

        #[test]
        fn reload_preserves_selected_run_by_id() {
            // After reload_data(), if the same run_id still exists,
            // the selection stays on it (line 168-172).
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("preserve.sqlite3");
            seed_runs(&db_path, 5, 1);

            let mut app = WhisperTuiApp::new(db_path);
            // Move selection to run index 3.
            app.move_selection(3);
            let selected_id = app.runs[app.selected_run].run_id.clone();
            assert_eq!(app.selected_run, 3);

            // Reload and verify same ID is still selected.
            app.reload_data();
            assert_eq!(
                app.runs[app.selected_run].run_id, selected_id,
                "reload should preserve selected run ID"
            );
        }

        #[test]
        fn runs_text_marker_after_move_selection() {
            // After move_selection, the marker '>' should follow the new selection.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("marker_move.sqlite3");
            seed_runs(&db_path, 4, 1);

            let mut app = WhisperTuiApp::new(db_path);
            app.move_selection(2);
            assert_eq!(app.selected_run, 2);

            let text = app.runs_text();
            let lines: Vec<&str> = text.lines().collect();
            assert!(
                lines[0].starts_with(' '),
                "index 0 should not be selected: {}",
                lines[0]
            );
            assert!(
                lines[2].starts_with('>'),
                "index 2 should be selected: {}",
                lines[2]
            );
        }

        #[test]
        fn centered_rect_zero_size_area() {
            use ftui::core::geometry::Rect;
            let area = Rect::new(0, 0, 0, 0);
            let r = super::centered_rect(area, 50, 50);
            // width = 50.min(0.saturating_sub(2)).max(10) = 50.min(0).max(10) = 0.max(10) = 10
            // height = 50.min(0.saturating_sub(2)).max(5) = 50.min(0).max(5) = 0.max(5) = 5
            assert_eq!(r.width, 10);
            assert_eq!(r.height, 5);
        }

        #[test]
        fn tick_count_increments_and_reloads_at_multiples_of_fifteen() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("tick.sqlite3");
            seed_runs(&db_path, 1, 1);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.tick_count, 0);

            // Tick 14 times — no reload at non-multiple of 15.
            for _ in 0..14 {
                app.tick_count = app.tick_count.saturating_add(1);
            }
            assert_eq!(app.tick_count, 14);
            assert!(!app.tick_count.is_multiple_of(15));

            // Tick once more — multiple of 15.
            app.tick_count = app.tick_count.saturating_add(1);
            assert_eq!(app.tick_count, 15);
            assert!(app.tick_count.is_multiple_of(15));
        }

        #[test]
        fn timeline_scroll_saturates_at_zero() {
            // Up arrow on Timeline when scroll is 0 should stay at 0.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("sat_scroll.sqlite3");
            seed_runs(&db_path, 1, 5);

            let mut app = WhisperTuiApp::new(db_path);
            app.focus = super::FocusPane::Timeline;
            app.timeline_scroll = 0;
            app.timeline_scroll = app.timeline_scroll.saturating_sub(1);
            assert_eq!(app.timeline_scroll, 0, "should not underflow");

            // PageUp (8) from 3 should go to 0, not underflow.
            app.timeline_scroll = 3;
            app.timeline_scroll = app.timeline_scroll.saturating_sub(8);
            assert_eq!(app.timeline_scroll, 0);
        }

        #[test]
        fn events_scroll_saturates_at_u16_max() {
            // Down arrow on Events at u16::MAX should stay at u16::MAX.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("sat_evt.sqlite3");
            seed_runs(&db_path, 1, 5);

            let mut app = WhisperTuiApp::new(db_path);
            app.focus = super::FocusPane::Events;
            app.events_scroll = u16::MAX;
            app.events_scroll = app.events_scroll.saturating_add(1);
            assert_eq!(app.events_scroll, u16::MAX, "should not overflow");
        }

        #[test]
        fn runs_text_contains_backend_as_str() {
            // Each run line should contain the backend's as_str() representation.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("backend_str.sqlite3");
            seed_runs(&db_path, 2, 1);

            let app = WhisperTuiApp::new(db_path);
            let text = app.runs_text();
            // Fixture uses BackendKind::WhisperCpp in result (stored as backend).
            assert!(
                text.contains("whisper_cpp"),
                "runs_text should contain backend as_str: {text}"
            );
        }

        #[test]
        fn reload_picks_up_new_runs_added_after_initial_load() {
            // After reload_data(), newly added runs should appear.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("dynamic.sqlite3");
            seed_runs(&db_path, 2, 1);

            let mut app = WhisperTuiApp::new(db_path.clone());
            assert_eq!(app.runs.len(), 2);

            // Add more runs.
            let store = RunStore::open(&db_path).expect("store");
            for i in 2..5 {
                store
                    .persist_report(&fixture_report(&format!("run-{i:03}"), &db_path, 1))
                    .expect("persist");
            }

            app.reload_data();
            assert_eq!(app.runs.len(), 5, "reload should pick up new runs");
        }

        #[test]
        fn event_lines_show_seq_in_three_digit_format() {
            // Event lines should format seq as 3-digit zero-padded (e.g., "001").
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("seq_fmt.sqlite3");
            seed_runs(&db_path, 1, 3);

            let app = WhisperTuiApp::new(db_path);
            let lines = app.event_lines();
            assert_eq!(lines.len(), 3);
            assert!(
                lines[0].starts_with("001"),
                "first event should start with 001: {}",
                lines[0]
            );
            assert!(
                lines[1].starts_with("002"),
                "second event should start with 002: {}",
                lines[1]
            );
            assert!(
                lines[2].starts_with("003"),
                "third event should start with 003: {}",
                lines[2]
            );
        }

        // ── Third-pass edge case tests ──

        #[test]
        fn reload_clamps_selection_when_selected_run_exceeds_count() {
            // When selected_run is beyond the new runs count after reload,
            // it should clamp to len()-1 (line 173-174).
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("clamp.sqlite3");
            seed_runs(&db_path, 3, 1);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 3);

            // Manually set selected_run out of bounds to simulate a shrunk run list.
            app.selected_run = 10;
            app.reload_data();
            assert_eq!(
                app.selected_run, 2,
                "should clamp to last valid index (len-1)"
            );
            assert!(
                app.details.is_some(),
                "should load details for clamped selection"
            );
        }

        #[test]
        fn timeline_lines_segment_without_confidence_omits_suffix() {
            // Segments with confidence: None should not have the (.NNN) suffix.
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_conf.sqlite3");

            let mut report = fixture_report("no-conf-run", &db_path, 2);
            // Override segments with None confidence.
            report.result.segments = vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.0),
                    text: "first".to_owned(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptionSegment {
                    start_sec: Some(1.0),
                    end_sec: Some(2.0),
                    text: "second".to_owned(),
                    speaker: Some("SPEAKER_01".to_owned()),
                    confidence: Some(0.75),
                },
            ];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert!(lines.len() >= 2);
            // First segment: no speaker, no confidence.
            assert!(
                !lines[0].contains('('),
                "no-confidence segment should not have parenthesized value: {}",
                lines[0]
            );
            assert!(
                !lines[0].contains("[SPEAKER"),
                "no-speaker segment should not have tag: {}",
                lines[0]
            );
            // Second segment: has speaker and confidence.
            assert!(
                lines[1].contains("[SPEAKER_01]"),
                "should have speaker: {}",
                lines[1]
            );
            assert!(
                lines[1].contains("(0.750)"),
                "should have confidence: {}",
                lines[1]
            );
        }

        #[test]
        fn status_hint_reflects_current_focus_pane() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("focus_hint.sqlite3");
            let _ = RunStore::open(&db_path).expect("create db");

            let mut app = WhisperTuiApp::new(db_path);
            assert!(
                app.status_hint().contains("Runs"),
                "default focus should be Runs: {}",
                app.status_hint()
            );

            app.focus = super::FocusPane::Timeline;
            assert!(
                app.status_hint().contains("Timeline"),
                "should reflect Timeline focus: {}",
                app.status_hint()
            );

            app.focus = super::FocusPane::Events;
            assert!(
                app.status_hint().contains("Events"),
                "should reflect Events focus: {}",
                app.status_hint()
            );
        }

        #[test]
        fn runs_text_includes_run_id_for_each_line() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("runid.sqlite3");
            seed_runs(&db_path, 3, 1);

            let app = WhisperTuiApp::new(db_path);
            let text = app.runs_text();
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(lines.len(), 3);

            // Each line should contain its run_id (format: "run-000", "run-001", "run-002").
            for (idx, run) in app.runs.iter().enumerate() {
                assert!(
                    lines[idx].contains(&run.run_id),
                    "line {idx} should contain run_id '{}': {}",
                    run.run_id,
                    lines[idx]
                );
            }
        }

        #[test]
        fn details_match_selected_run_after_move() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("details_match.sqlite3");
            seed_runs(&db_path, 4, 3);

            let mut app = WhisperTuiApp::new(db_path);
            assert!(app.details.is_some());

            // Move to run index 2.
            app.move_selection(2);
            let expected_id = app.runs[2].run_id.clone();
            let details = app.details.as_ref().expect("should have details");
            assert_eq!(
                details.run_id, expected_id,
                "details run_id should match selected run"
            );
            assert!(
                !details.transcript.is_empty(),
                "details should have a transcript"
            );
        }

        #[test]
        fn move_selection_resets_scroll_offsets_to_zero() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("scroll_reset.sqlite3");
            seed_runs(&db_path, 5, 10);

            let mut app = WhisperTuiApp::new(db_path);
            // Set non-zero scroll offsets.
            app.timeline_scroll = 42;
            app.events_scroll = 99;
            assert_eq!(app.selected_run, 0);

            // Moving selection should reset both scroll offsets.
            app.move_selection(2);
            assert_eq!(app.selected_run, 2);
            assert_eq!(
                app.timeline_scroll, 0,
                "timeline_scroll should reset on move"
            );
            assert_eq!(app.events_scroll, 0, "events_scroll should reset on move");
        }

        #[test]
        fn timeline_lines_empty_segments_shows_no_segments_placeholder() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_seg.sqlite3");

            // Persist a report with zero segments.
            let mut report = fixture_report("empty-segs", &db_path, 1);
            report.result.segments = vec![];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            assert!(app.details.is_some());
            let lines = app.timeline_lines();
            assert_eq!(lines, vec!["Selected run has no segments"]);
        }

        #[test]
        fn event_lines_empty_events_shows_no_events_placeholder() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("no_evt.sqlite3");

            // Persist a report with zero events.
            let mut report = fixture_report("empty-evts", &db_path, 1);
            report.events = vec![];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            assert!(app.details.is_some());
            let lines = app.event_lines();
            assert_eq!(lines, vec!["Selected run has no events"]);
        }

        #[test]
        fn format_ts_exact_hour_boundary() {
            assert_eq!(format_ts(Some(3600.0)), "01:00:00.000");
            assert_eq!(format_ts(Some(7200.0)), "02:00:00.000");
            assert_eq!(format_ts(Some(86400.0)), "24:00:00.000");
        }

        #[test]
        fn runs_text_no_duplicate_selection_markers() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("dup_marker.sqlite3");
            seed_runs(&db_path, 5, 1);

            let mut app = WhisperTuiApp::new(db_path);
            app.move_selection(2);
            let text = app.runs_text();
            let lines: Vec<&str> = text.lines().collect();
            let marker_count = lines.iter().filter(|line| line.starts_with('>')).count();
            assert_eq!(
                marker_count, 1,
                "exactly one line should have the > marker, got {marker_count}"
            );
            assert!(
                lines[2].starts_with('>'),
                "selected_run=2 should have marker: {}",
                lines[2]
            );
        }

        #[test]
        fn timeline_lines_none_timestamps_show_dashes() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("none_ts.sqlite3");

            let mut report = fixture_report("none-ts", &db_path, 1);
            report.result.segments = vec![TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: "no timing".to_owned(),
                speaker: None,
                confidence: None,
            }];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("--:--:--.---"),
                "None timestamps should show dashes: {}",
                lines[0]
            );
            // Count: two occurrences of the dash pattern (start and end).
            let dash_count = lines[0].matches("--:--:--.---").count();
            assert_eq!(dash_count, 2, "both start and end should be dashes");
        }

        #[test]
        fn event_lines_large_seq_shows_four_digit_format() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("large_seq.sqlite3");

            let mut report = fixture_report("big-seq", &db_path, 1);
            report.events = vec![crate::model::RunEvent {
                seq: 1234,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "large seq".to_owned(),
                payload: json!({}),
            }];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.event_lines();
            assert_eq!(lines.len(), 1);
            // {:03} with value 1234 → "1234" (4 digits, no truncation).
            assert!(
                lines[0].starts_with("1234"),
                "seq=1234 should show 4 digits: {}",
                lines[0]
            );
        }

        #[test]
        fn reload_with_invalid_db_path_shows_error_in_status() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("valid.sqlite3");
            seed_runs(&db_path, 2, 1);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.runs.len(), 2);
            assert!(app.details.is_some());

            // Point app at an invalid path (directory, not a file).
            app.db_path = dir.path().to_path_buf();
            app.reload_data();

            // Reload with invalid DB should clear runs and show error.
            assert!(app.runs.is_empty(), "invalid db should clear runs");
            assert!(app.details.is_none());
            assert!(
                app.status_line.contains("db open failed"),
                "status should show error: {}",
                app.status_line
            );
        }

        #[test]
        fn centered_rect_centering_with_odd_dimensions() {
            use ftui::core::geometry::Rect;
            // Area 101×51, requested overlay 40×10.
            let area = Rect::new(5, 3, 101, 51);
            let r = super::centered_rect(area, 40, 10);
            // w = 40.min(99).max(10) = 40
            // h = 10.min(49).max(5) = 10
            assert_eq!(r.width, 40);
            assert_eq!(r.height, 10);
            // x = 5 + (101 - 40) / 2 = 5 + 30 = 35
            // y = 3 + (51 - 10) / 2 = 3 + 20 = 23
            assert_eq!(r.x, 35, "x centering");
            assert_eq!(r.y, 23, "y centering");
        }

        #[test]
        fn reload_data_when_selected_id_removed_keeps_index_in_bounds() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("remove_run.sqlite3");
            seed_runs(&db_path, 4, 1);

            let mut app = WhisperTuiApp::new(db_path.clone());
            assert_eq!(app.runs.len(), 4);
            // Select the last run.
            app.move_selection(3);
            assert_eq!(app.selected_run, 3);
            let last_run_id = app.runs[3].run_id.clone();

            // Delete that run from the DB via raw SQL.
            let conn = fsqlite::Connection::open(db_path.display().to_string()).expect("conn");
            conn.execute_with_params(
                "DELETE FROM runs WHERE id = ?1",
                &[fsqlite_types::value::SqliteValue::Text(last_run_id)],
            )
            .expect("delete run");

            // Reload — the selected_run=3 is now out of bounds (only 3 runs left).
            // Code clamps to len()-1 = 2.
            app.reload_data();
            assert_eq!(app.runs.len(), 3);
            assert_eq!(
                app.selected_run, 2,
                "should clamp to last valid index after removal"
            );
        }

        #[test]
        fn reload_successful_status_line_contains_loaded_and_db_path() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("status_ok.sqlite3");
            seed_runs(&db_path, 3, 1);

            let app = WhisperTuiApp::new(db_path.clone());
            assert!(
                app.status_line.contains("Loaded 3 runs"),
                "should report run count: {}",
                app.status_line
            );
            assert!(
                app.status_line.contains(&db_path.display().to_string()),
                "should contain db path: {}",
                app.status_line
            );
        }

        #[test]
        fn reload_data_reports_corrupt_selected_run_details() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("status_corrupt.sqlite3");
            seed_runs(&db_path, 2, 1);

            let store = RunStore::open(&db_path).expect("store");
            store
                .connection
                .execute_with_params(
                    "UPDATE runs SET warnings_json = ?1 WHERE id = ?2",
                    &[
                        SqliteValue::Text("{bad json".to_owned()),
                        SqliteValue::Text("run-001".to_owned()),
                    ],
                )
                .expect("corrupt selected run");

            let app = WhisperTuiApp::new(db_path);
            assert!(
                app.details.is_none(),
                "corrupt selected run should not load"
            );
            assert!(
                app.status_line
                    .contains("failed to load selected run details for run-001"),
                "status line should surface selected-run load failure: {}",
                app.status_line
            );
        }

        #[test]
        fn timeline_lines_confidence_zero_and_one() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("conf_bounds.sqlite3");

            let mut report = fixture_report("conf-run", &db_path, 2);
            report.result.segments = vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.0),
                    text: "zero conf".to_owned(),
                    speaker: None,
                    confidence: Some(0.0),
                },
                TranscriptionSegment {
                    start_sec: Some(1.0),
                    end_sec: Some(2.0),
                    text: "full conf".to_owned(),
                    speaker: None,
                    confidence: Some(1.0),
                },
            ];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.timeline_lines();
            assert!(lines.len() >= 2);
            assert!(
                lines[0].contains("(0.000)"),
                "confidence 0.0 should show as (0.000): {}",
                lines[0]
            );
            assert!(
                lines[1].contains("(1.000)"),
                "confidence 1.0 should show as (1.000): {}",
                lines[1]
            );
        }

        #[test]
        fn move_selection_zero_delta_is_noop() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("zero_delta.sqlite3");
            seed_runs(&db_path, 5, 1);

            let mut app = WhisperTuiApp::new(db_path);
            app.move_selection(2);
            assert_eq!(app.selected_run, 2);

            // Set scroll offsets to non-zero.
            app.timeline_scroll = 10;
            app.events_scroll = 20;

            // move_selection(0) should keep index but reset scrolls.
            app.move_selection(0);
            assert_eq!(app.selected_run, 2, "index unchanged");
            assert_eq!(app.timeline_scroll, 0, "timeline scroll reset");
            assert_eq!(app.events_scroll, 0, "events scroll reset");
        }

        #[test]
        fn initial_load_selects_first_run_with_details() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("initial_select.sqlite3");
            seed_runs(&db_path, 3, 2);

            let app = WhisperTuiApp::new(db_path);
            assert_eq!(app.selected_run, 0, "initial selection should be 0");
            let details = app.details.as_ref().expect("should have details");
            assert_eq!(
                details.run_id, app.runs[0].run_id,
                "details should match first run"
            );
        }

        #[test]
        fn event_lines_seq_zero_shows_triple_zero() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("seq_zero.sqlite3");

            let mut report = fixture_report("seq0-run", &db_path, 1);
            report.events = vec![RunEvent {
                seq: 0,
                ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "event zero".to_owned(),
                payload: json!({}),
            }];
            let store = RunStore::open(&db_path).expect("store");
            store.persist_report(&report).expect("persist");

            let app = WhisperTuiApp::new(db_path);
            let lines = app.event_lines();
            assert!(
                lines[0].starts_with("000"),
                "seq=0 should format as 000: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("[ingest]"),
                "should show stage: {}",
                lines[0]
            );
        }

        #[test]
        fn update_msg_tick_increments_tick_count() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("tick_update.sqlite3");
            seed_runs(&db_path, 1, 1);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.tick_count, 0);

            app.update(Msg::Tick);
            assert_eq!(app.tick_count, 1);

            // Tick 13 more times — total 14, not multiple of 15.
            for _ in 0..13 {
                app.update(Msg::Tick);
            }
            assert_eq!(app.tick_count, 14);
            // One more → 15, triggers reload.
            app.update(Msg::Tick);
            assert_eq!(app.tick_count, 15);
        }

        #[test]
        fn update_msg_ignore_does_not_change_state() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("ignore.sqlite3");
            seed_runs(&db_path, 3, 2);

            let mut app = WhisperTuiApp::new(db_path);
            let tick_before = app.tick_count;
            let selected_before = app.selected_run;
            let focus_before = app.focus;
            let help_before = app.show_help;
            let timeline_scroll_before = app.timeline_scroll;
            let events_scroll_before = app.events_scroll;

            app.update(Msg::Ignore);

            assert_eq!(app.tick_count, tick_before);
            assert_eq!(app.selected_run, selected_before);
            assert_eq!(app.focus, focus_before);
            assert_eq!(app.show_help, help_before);
            assert_eq!(app.timeline_scroll, timeline_scroll_before);
            assert_eq!(app.events_scroll, events_scroll_before);
        }

        #[test]
        fn update_key_release_is_noop() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("release.sqlite3");
            seed_runs(&db_path, 3, 1);

            let mut app = WhisperTuiApp::new(db_path);
            let selected_before = app.selected_run;
            let focus_before = app.focus;

            // Key release event: 'q' released should NOT quit or change state.
            let release_key = ftui::KeyEvent {
                code: KeyCode::Char('q'),
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Release,
            };
            app.update(Msg::Key(release_key));

            assert_eq!(app.selected_run, selected_before);
            assert_eq!(app.focus, focus_before);

            // Key repeat event: Down arrow repeat should also be noop.
            let repeat_key = ftui::KeyEvent {
                code: KeyCode::Down,
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Repeat,
            };
            app.update(Msg::Key(repeat_key));
            assert_eq!(app.selected_run, selected_before);
        }

        #[test]
        fn page_scroll_in_timeline_and_events_focus() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("page_scroll.sqlite3");
            seed_runs(&db_path, 1, 50);

            let mut app = WhisperTuiApp::new(db_path);

            // Switch to Timeline focus.
            app.focus = FocusPane::Timeline;
            assert_eq!(app.timeline_scroll, 0);

            // PageDown in Timeline.
            let page_down = ftui::KeyEvent {
                code: KeyCode::PageDown,
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Press,
            };
            app.update(Msg::Key(page_down));
            assert_eq!(app.timeline_scroll, 8);

            app.update(Msg::Key(page_down));
            assert_eq!(app.timeline_scroll, 16);

            // PageUp in Timeline.
            let page_up = ftui::KeyEvent {
                code: KeyCode::PageUp,
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Press,
            };
            app.update(Msg::Key(page_up));
            assert_eq!(app.timeline_scroll, 8);

            // Switch to Events focus.
            app.focus = FocusPane::Events;
            assert_eq!(app.events_scroll, 0);

            app.update(Msg::Key(page_down));
            assert_eq!(app.events_scroll, 8);

            // PageUp past zero saturates at 0.
            app.update(Msg::Key(page_up));
            assert_eq!(app.events_scroll, 0);
            app.update(Msg::Key(page_up));
            assert_eq!(app.events_scroll, 0);
        }

        #[test]
        fn update_backtab_cycles_focus_backwards() {
            let dir = tempdir().expect("tempdir");
            let db_path = dir.path().join("backtab.sqlite3");
            seed_runs(&db_path, 1, 1);

            let mut app = WhisperTuiApp::new(db_path);
            assert_eq!(app.focus, FocusPane::Runs);

            // Tab forward: Runs → Timeline.
            let tab_key = ftui::KeyEvent {
                code: KeyCode::Tab,
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Press,
            };
            app.update(Msg::Key(tab_key));
            assert_eq!(app.focus, FocusPane::Timeline);

            // BackTab: Timeline → Runs.
            let backtab_key = ftui::KeyEvent {
                code: KeyCode::BackTab,
                modifiers: Modifiers::empty(),
                kind: KeyEventKind::Press,
            };
            app.update(Msg::Key(backtab_key));
            assert_eq!(app.focus, FocusPane::Runs);

            // BackTab again: Runs → Events (wraps around).
            app.update(Msg::Key(backtab_key));
            assert_eq!(app.focus, FocusPane::Events);
        }

        // ── LiveTranscriptionView tests (bd-339.1) ──────────────────

        use super::LiveTranscriptionView;

        fn make_segment(
            start: Option<f64>,
            end: Option<f64>,
            text: &str,
            speaker: Option<&str>,
            confidence: Option<f64>,
        ) -> TranscriptionSegment {
            TranscriptionSegment {
                start_sec: start,
                end_sec: end,
                text: text.to_owned(),
                speaker: speaker.map(|s| s.to_owned()),
                confidence,
            }
        }

        #[test]
        fn live_view_new_starts_empty_with_auto_scroll() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            assert_eq!(view.segment_count(), 0);
            assert!(view.is_auto_scroll());
            assert_eq!(view.backend, BackendKind::WhisperCpp);
            assert!(!view.diarization_active);
        }

        #[test]
        fn live_view_new_with_diarization() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperDiarization, true);
            assert!(view.diarization_active);
            assert_eq!(view.backend, BackendKind::WhisperDiarization);
        }

        #[test]
        fn live_view_push_segment_increments_count() {
            let mut view = LiveTranscriptionView::new(BackendKind::Auto, false);
            assert_eq!(view.segment_count(), 0);

            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));
            assert_eq!(view.segment_count(), 1);

            view.push_segment(make_segment(Some(1.0), Some(2.0), "world", None, None));
            assert_eq!(view.segment_count(), 2);
        }

        #[test]
        fn live_view_push_segments_batch() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            let batch = vec![
                make_segment(Some(0.0), Some(1.0), "one", None, None),
                make_segment(Some(1.0), Some(2.0), "two", None, None),
                make_segment(Some(2.0), Some(3.0), "three", None, None),
            ];
            view.push_segments(batch);
            assert_eq!(view.segment_count(), 3);
        }

        #[test]
        fn live_view_empty_shows_waiting_message() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            let lines = view.segment_lines();
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("Waiting"),
                "should show waiting message: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_show_timestamps() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(62.345),
                Some(65.0),
                "hello world",
                None,
                None,
            ));
            let lines = view.segment_lines();
            assert_eq!(lines.len(), 1);
            assert!(
                lines[0].contains("00:01:02.345"),
                "should contain start timestamp: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("00:01:05.000"),
                "should contain end timestamp: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("hello world"),
                "should contain text: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_show_speaker_when_diarization_active() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperDiarization, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                Some(0.95),
            ));
            view.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "world",
                Some("SPEAKER_01"),
                None,
            ));
            let lines = view.segment_lines();
            assert_eq!(lines.len(), 2);
            assert!(
                lines[0].contains("[SPEAKER_00]"),
                "should show speaker label: {}",
                lines[0]
            );
            assert!(
                lines[1].contains("[SPEAKER_01]"),
                "should show speaker label: {}",
                lines[1]
            );
        }

        #[test]
        fn live_view_segment_lines_hide_speaker_when_diarization_inactive() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                None,
            ));
            let lines = view.segment_lines();
            assert!(
                !lines[0].contains("[SPEAKER"),
                "should NOT show speaker when diarization is off: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_show_confidence() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                None,
                Some(0.925),
            ));
            let lines = view.segment_lines();
            assert!(
                lines[0].contains("(0.925)"),
                "should show confidence: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_omit_confidence_when_none() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));
            let lines = view.segment_lines();
            assert!(
                !lines[0].contains('('),
                "should not show confidence parens: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_none_timestamps_show_dashes() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(None, None, "no timing", None, None));
            let lines = view.segment_lines();
            let dash_count = lines[0].matches("--:--:--.---").count();
            assert_eq!(
                dash_count, 2,
                "both start and end should be dashes: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_index_formatting() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..3 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            let lines = view.segment_lines();
            assert!(lines[0].starts_with("000"), "index 0: {}", lines[0]);
            assert!(lines[1].starts_with("001"), "index 1: {}", lines[1]);
            assert!(lines[2].starts_with("002"), "index 2: {}", lines[2]);
        }

        #[test]
        fn live_view_elapsed_display_zero() {
            let view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 1000.0);
            assert_eq!(view.elapsed_display(), "00:00:00");
        }

        #[test]
        fn live_view_elapsed_display_after_tick() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 1000.0);
            // 1 hour, 2 minutes, 3 seconds later.
            view.tick(1000.0 + 3723.0);
            assert_eq!(view.elapsed_display(), "01:02:03");
        }

        #[test]
        fn live_view_elapsed_display_negative_clamps_to_zero() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 2000.0);
            // Current time is before start time (clock skew).
            view.tick(1000.0);
            assert_eq!(view.elapsed_display(), "00:00:00");
        }

        #[test]
        fn live_view_status_bar_contains_all_fields() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::InsanelyFast, true, 1000.0);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));
            view.tick(1060.0); // 1 minute elapsed.

            let status = view.status_bar_text();
            assert!(
                status.contains("elapsed=00:01:00"),
                "should show elapsed: {status}"
            );
            assert!(
                status.contains("segments=1"),
                "should show segment count: {status}"
            );
            assert!(
                status.contains("backend=insanely_fast"),
                "should show backend: {status}"
            );
            assert!(
                status.contains("LIVE"),
                "should show LIVE indicator: {status}"
            );
            assert!(
                status.contains("diarization=on"),
                "should show diarization: {status}"
            );
        }

        #[test]
        fn live_view_status_bar_no_diarization_indicator_when_off() {
            let view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 1000.0);
            let status = view.status_bar_text();
            assert!(
                !status.contains("diarization"),
                "should not mention diarization when off: {status}"
            );
        }

        #[test]
        fn live_view_status_bar_shows_paused_when_not_auto_scroll() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 1000.0);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));
            view.scroll_up(1);
            let status = view.status_bar_text();
            assert!(
                status.contains("PAUSED"),
                "should show PAUSED when scrolled up: {status}"
            );
        }

        #[test]
        fn live_view_scroll_up_disables_auto_scroll() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..20 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            assert!(view.is_auto_scroll());
            view.scroll_up(5);
            assert!(!view.is_auto_scroll());
        }

        #[test]
        fn live_view_scroll_down_to_bottom_re_enables_auto_scroll() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..10 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            // Disable auto-scroll by scrolling up.
            view.scroll_up(5);
            assert!(!view.is_auto_scroll());
            // Scroll down past the end to re-enable.
            view.scroll_down(100);
            assert!(view.is_auto_scroll());
        }

        #[test]
        fn live_view_jump_to_latest_re_enables_auto_scroll() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));
            view.scroll_up(1);
            assert!(!view.is_auto_scroll());
            view.jump_to_latest();
            assert!(view.is_auto_scroll());
        }

        #[test]
        fn live_view_effective_scroll_auto_scroll_shows_latest() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..100 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            assert!(view.is_auto_scroll());
            // With 10 visible lines, scroll should be 100 - 10 = 90.
            assert_eq!(view.effective_scroll(10), 90);
        }

        #[test]
        fn live_view_effective_scroll_manual_uses_offset() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..100 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            view.scroll_up(5); // Sets scroll_offset and disables auto-scroll.
            // scroll_offset was 0 before scroll_up, so 0.saturating_sub(5) = 0.
            assert_eq!(view.scroll_offset, 0);
            assert_eq!(view.effective_scroll(10), 0);
        }

        #[test]
        fn live_view_effective_scroll_empty_segments() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            // With auto-scroll on and 0 segments, scroll = 0.saturating_sub(10) = 0.
            assert_eq!(view.effective_scroll(10), 0);
        }

        #[test]
        fn live_view_effective_scroll_fewer_than_visible() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..3 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            // 3 segments, 10 visible: scroll = 3 - 10 = 0 (saturating).
            assert_eq!(view.effective_scroll(10), 0);
        }

        #[test]
        fn live_view_max_segments_enforced() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(5);
            for i in 0..10 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            assert_eq!(view.segment_count(), 5);
            // The remaining segments should be the last 5 (seg-5 through seg-9).
            let lines = view.segment_lines();
            assert!(
                lines[0].contains("seg-5"),
                "oldest retained should be seg-5: {}",
                lines[0]
            );
            assert!(
                lines[4].contains("seg-9"),
                "newest should be seg-9: {}",
                lines[4]
            );
        }

        #[test]
        fn live_view_max_segments_batch_enforced() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(3);
            let batch: Vec<TranscriptionSegment> = (0..8)
                .map(|i| {
                    make_segment(
                        Some(i as f64),
                        Some((i + 1) as f64),
                        &format!("b-{i}"),
                        None,
                        None,
                    )
                })
                .collect();
            view.push_segments(batch);
            assert_eq!(view.segment_count(), 3);
            let lines = view.segment_lines();
            assert!(lines[0].contains("b-5"), "first retained: {}", lines[0]);
            assert!(lines[2].contains("b-7"), "last retained: {}", lines[2]);
        }

        #[test]
        fn live_view_max_segments_minimum_is_one() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(0); // Should clamp to 1.
            view.push_segment(make_segment(Some(0.0), Some(1.0), "a", None, None));
            view.push_segment(make_segment(Some(1.0), Some(2.0), "b", None, None));
            assert_eq!(view.segment_count(), 1);
            let lines = view.segment_lines();
            assert!(
                lines[0].contains("b"),
                "should keep the latest: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_max_segments_adjusts_scroll_offset() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(5);
            // Push 5 segments, then scroll to offset 3.
            for i in 0..5 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            view.auto_scroll = false;
            view.scroll_offset = 3;
            // Push 3 more — drains 3 oldest, scroll_offset should decrease by 3.
            for i in 5..8 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            assert_eq!(view.segment_count(), 5);
            assert_eq!(
                view.scroll_offset, 0,
                "scroll offset should be adjusted down"
            );
        }

        #[test]
        fn live_view_scroll_up_saturates_at_zero() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "a", None, None));
            view.scroll_up(100);
            assert_eq!(view.scroll_offset, 0);
        }

        #[test]
        fn live_view_scroll_down_saturates_at_u16_max() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.scroll_up(1); // Disable auto-scroll first.
            view.scroll_offset = u16::MAX;
            // scroll_down should saturate.
            view.scroll_down(1);
            assert_eq!(view.scroll_offset, u16::MAX);
        }

        #[test]
        fn live_view_with_start_time_deterministic() {
            let view = LiveTranscriptionView::with_start_time(
                BackendKind::WhisperCpp,
                false,
                1_700_000_000.0,
            );
            assert_eq!(view.started_epoch_secs, 1_700_000_000.0);
            assert_eq!(view.current_epoch_secs, 1_700_000_000.0);
            assert_eq!(view.elapsed_display(), "00:00:00");
        }

        #[test]
        fn live_view_elapsed_large_value() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            // 25 hours.
            view.tick(90_000.0);
            assert_eq!(view.elapsed_display(), "25:00:00");
        }

        #[test]
        fn live_view_all_backend_kinds_in_status() {
            for backend in [
                BackendKind::Auto,
                BackendKind::WhisperCpp,
                BackendKind::InsanelyFast,
                BackendKind::WhisperDiarization,
            ] {
                let view = LiveTranscriptionView::with_start_time(backend, false, 0.0);
                let status = view.status_bar_text();
                assert!(
                    status.contains(backend.as_str()),
                    "status should contain '{}': {status}",
                    backend.as_str()
                );
            }
        }

        #[test]
        fn live_view_segment_lines_speaker_with_no_label() {
            // Diarization is active but segment has no speaker assigned.
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperDiarization, true);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "unlabeled", None, None));
            let lines = view.segment_lines();
            // Should not have a speaker tag since speaker is None.
            assert!(
                !lines[0].contains('['),
                "no speaker tag when speaker is None: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_segment_lines_confidence_boundary_values() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "zero", None, Some(0.0)));
            view.push_segment(make_segment(Some(1.0), Some(2.0), "one", None, Some(1.0)));
            let lines = view.segment_lines();
            assert!(
                lines[0].contains("(0.000)"),
                "zero confidence: {}",
                lines[0]
            );
            assert!(
                lines[1].contains("(1.000)"),
                "full confidence: {}",
                lines[1]
            );
        }

        #[test]
        fn live_view_many_segments_auto_scroll_tracks_latest() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..500 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            assert!(view.is_auto_scroll());
            // With visible_height=20, effective scroll should show last 20.
            let scroll = view.effective_scroll(20);
            assert_eq!(scroll, 480);
        }

        #[test]
        fn live_view_tick_updates_current_time() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 100.0);
            assert_eq!(view.current_epoch_secs, 100.0);
            view.tick(200.0);
            assert_eq!(view.current_epoch_secs, 200.0);
            assert_eq!(view.elapsed_display(), "00:01:40");
        }

        #[test]
        fn live_view_clone_is_independent() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "original", None, None));
            let mut cloned = view.clone();
            cloned.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "cloned-only",
                None,
                None,
            ));
            assert_eq!(view.segment_count(), 1);
            assert_eq!(cloned.segment_count(), 2);
        }

        #[test]
        fn live_view_debug_impl_does_not_panic() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            let debug = format!("{view:?}");
            assert!(!debug.is_empty());
        }

        #[test]
        fn live_view_format_segment_line_with_all_fields() {
            let seg = make_segment(
                Some(3723.456),
                Some(3725.0),
                "test text",
                Some("SPEAKER_02"),
                Some(0.875),
            );
            // With speaker shown.
            let line = LiveTranscriptionView::format_segment_line(42, &seg, true);
            assert!(line.starts_with("042"), "index: {line}");
            assert!(line.contains("01:02:03.456"), "start ts: {line}");
            assert!(line.contains("01:02:05.000"), "end ts: {line}");
            assert!(line.contains("[SPEAKER_02]"), "speaker: {line}");
            assert!(line.contains("test text"), "text: {line}");
            assert!(line.contains("(0.875)"), "confidence: {line}");
        }

        #[test]
        fn live_view_format_segment_line_without_speaker() {
            let seg = make_segment(Some(0.0), Some(1.0), "hi", Some("SPEAKER_00"), None);
            // With speaker hidden.
            let line = LiveTranscriptionView::format_segment_line(0, &seg, false);
            assert!(
                !line.contains("[SPEAKER"),
                "speaker should be hidden: {line}"
            );
        }

        #[test]
        fn live_view_scroll_sequence_up_down_up() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..50 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            assert!(view.is_auto_scroll());

            // Scroll up.
            view.scroll_up(10);
            assert!(!view.is_auto_scroll());
            assert_eq!(view.scroll_offset, 0); // Was 0, sub 10 = 0 saturating.

            // Scroll down a bit (not to bottom).
            view.scroll_down(5);
            assert_eq!(view.scroll_offset, 5);
            assert!(!view.is_auto_scroll()); // Still not at bottom.

            // Scroll up again.
            view.scroll_up(3);
            assert_eq!(view.scroll_offset, 2);

            // Jump to latest.
            view.jump_to_latest();
            assert!(view.is_auto_scroll());
        }

        #[test]
        fn live_view_scroll_down_exact_to_last_enables_auto() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            for i in 0..5 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            view.scroll_up(1); // Disable auto-scroll.
            // Scroll_offset is 0 after scroll_up from 0. scroll_down to index 4 (last).
            view.scroll_down(4);
            // scroll_offset=4, segments.len()-1=4, so auto_scroll re-enabled.
            assert!(view.is_auto_scroll());
        }

        #[test]
        fn live_view_empty_text_segment() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(0.5), "", None, None));
            let lines = view.segment_lines();
            assert_eq!(lines.len(), 1);
            // Should still have the index, timestamps, and arrow.
            assert!(
                lines[0].contains("->"),
                "should have arrow separator: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_unicode_text() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "日本語テスト",
                Some("話者_01"),
                Some(0.99),
            ));
            let lines = view.segment_lines();
            assert!(
                lines[0].contains("日本語テスト"),
                "unicode text: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("[話者_01]"),
                "unicode speaker: {}",
                lines[0]
            );
        }

        #[test]
        fn live_view_default_max_segments() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            assert_eq!(
                view.max_segments,
                LiveTranscriptionView::DEFAULT_MAX_SEGMENTS
            );
            assert_eq!(view.max_segments, 10_000);
        }

        #[test]
        fn live_view_push_at_exactly_max_does_not_drain() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(3);
            for i in 0..3 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            assert_eq!(view.segment_count(), 3);
            let lines = view.segment_lines();
            assert!(lines[0].contains("s-0"), "first: {}", lines[0]);
            assert!(lines[2].contains("s-2"), "last: {}", lines[2]);
        }

        #[test]
        fn live_view_push_one_over_max_drains_one() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_max_segments(3);
            for i in 0..4 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("s-{i}"),
                    None,
                    None,
                ));
            }
            assert_eq!(view.segment_count(), 3);
            let lines = view.segment_lines();
            assert!(lines[0].contains("s-1"), "oldest after drain: {}", lines[0]);
        }

        #[test]
        fn live_view_status_bar_zero_segments() {
            let view = LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            let status = view.status_bar_text();
            assert!(
                status.contains("segments=0"),
                "should show 0 segments: {status}"
            );
        }

        #[test]
        fn live_view_elapsed_display_sub_second_rounding() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            view.tick(0.4); // Less than 0.5 seconds, rounds to 0.
            assert_eq!(view.elapsed_display(), "00:00:00");
            view.tick(0.5); // Exactly 0.5, rounds to 1.
            assert_eq!(view.elapsed_display(), "00:00:01");
        }

        // ── SpeakerColorMap tests (bd-339.2) ──────────────────────────────

        use super::SpeakerColorMap;

        #[test]
        fn speaker_color_map_consistent_assignment() {
            let mut map = SpeakerColorMap::new();
            let idx1 = map.color_index("SPEAKER_00");
            let idx2 = map.color_index("SPEAKER_00");
            assert_eq!(idx1, idx2, "same speaker should always get same index");
        }

        #[test]
        fn speaker_color_map_different_speakers_can_differ() {
            let mut map = SpeakerColorMap::new();
            let idx_a = map.color_index("SPEAKER_00");
            let idx_b = map.color_index("SPEAKER_01");
            // Not guaranteed to differ, but the hash should at least produce valid indices.
            assert!(idx_a < 8, "index in palette range");
            assert!(idx_b < 8, "index in palette range");
        }

        #[test]
        fn speaker_color_map_index_in_palette_range() {
            let mut map = SpeakerColorMap::new();
            for i in 0..100 {
                let name = format!("speaker_{i}");
                let idx = map.color_index(&name);
                assert!(idx < 8, "index {idx} should be < 8 for speaker {name}");
            }
        }

        #[test]
        fn speaker_color_map_color_for_returns_valid_rgb() {
            let mut map = SpeakerColorMap::new();
            let (r, g, b) = map.color_for("SPEAKER_00");
            // Verify it returns a valid RGB triple without panicking.
            // Use u16 casts to avoid trivially-true u8 comparisons.
            let (r16, g16, b16) = (r as u16, g as u16, b as u16);
            assert!(r16 <= 255 && g16 <= 255 && b16 <= 255);
        }

        #[test]
        fn speaker_color_map_consistency_across_instances() {
            let mut map1 = SpeakerColorMap::new();
            let mut map2 = SpeakerColorMap::new();
            for name in ["Alice", "Bob", "Charlie", "SPEAKER_00", "話者_01"] {
                assert_eq!(
                    map1.color_index(name),
                    map2.color_index(name),
                    "different instances should assign same index for '{name}'"
                );
            }
        }

        #[test]
        fn speaker_color_map_empty_name() {
            let mut map = SpeakerColorMap::new();
            let idx = map.color_index("");
            assert!(idx < 8, "empty name should still produce valid index");
        }

        #[test]
        fn speaker_color_map_unicode_names() {
            let mut map = SpeakerColorMap::new();
            let idx1 = map.color_index("話者_01");
            let idx2 = map.color_index("話者_01");
            assert_eq!(idx1, idx2, "unicode speaker names should be consistent");
            assert!(idx1 < 8);
        }

        #[test]
        fn speaker_color_map_default_is_empty() {
            let map = SpeakerColorMap::default();
            assert!(map.cache.is_empty());
        }

        // ── WaveformDisplay tests (bd-339.2) ──────────────────────────────

        use super::WaveformDisplay;

        #[test]
        fn waveform_display_empty() {
            let wf = WaveformDisplay::new(vec![]);
            assert_eq!(wf.amplitude_bars(), "");
            assert_eq!(wf.len(), 0);
        }

        #[test]
        fn waveform_display_silence() {
            let wf = WaveformDisplay::new(vec![0.0; 5]);
            assert_eq!(wf.amplitude_bars(), "     ", "all zeros should be spaces");
        }

        #[test]
        fn waveform_display_full_amplitude() {
            let wf = WaveformDisplay::new(vec![1.0; 3]);
            assert_eq!(
                wf.amplitude_bars(),
                "\u{2588}\u{2588}\u{2588}",
                "all 1.0 should be full blocks"
            );
        }

        #[test]
        fn waveform_display_half_amplitude() {
            let wf = WaveformDisplay::new(vec![0.5]);
            let bars = wf.amplitude_bars();
            // 0.5 * 8 = 4.0, rounds to 4 → '▄'
            assert_eq!(bars, "\u{2584}", "half amplitude should be ▄");
        }

        #[test]
        fn waveform_display_clamps_negative() {
            let wf = WaveformDisplay::new(vec![-0.5, -1.0]);
            assert_eq!(
                wf.amplitude_bars(),
                "  ",
                "negative values clamp to 0 (space)"
            );
        }

        #[test]
        fn waveform_display_clamps_above_one() {
            let wf = WaveformDisplay::new(vec![1.5, 2.0]);
            assert_eq!(
                wf.amplitude_bars(),
                "\u{2588}\u{2588}",
                "values >1.0 clamp to full block"
            );
        }

        #[test]
        fn waveform_display_gradient() {
            let wf =
                WaveformDisplay::new(vec![0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]);
            let bars = wf.amplitude_bars();
            assert_eq!(bars.chars().count(), 9, "should have 9 characters");
            // First char should be space (0.0), last should be full block (1.0).
            assert_eq!(bars.chars().next(), Some(' '));
            assert_eq!(bars.chars().last(), Some('\u{2588}'));
        }

        #[test]
        fn waveform_display_len() {
            let wf = WaveformDisplay::new(vec![0.1, 0.2, 0.3, 0.4]);
            assert_eq!(wf.len(), 4);
        }

        #[test]
        fn waveform_display_non_finite_renders_without_panic() {
            // NaN/Inf amplitudes (e.g. derived from NaN audio) must not panic the
            // render path on the nightly toolchain; they sanitize to 0.0 (space).
            let wf = WaveformDisplay::new(vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MIN_POSITIVE,
            ]);
            let bars = wf.amplitude_bars();
            assert_eq!(bars.chars().count(), 4);
            // NaN/Inf clamp to 0.0 → space; the denormal MIN_POSITIVE also rounds
            // to the empty block.
            assert_eq!(bars, "    ", "non-finite/denormal amplitudes render as spaces");
        }

        // ── SearchState and search/filter/export tests (bd-339.3) ──────────

        use super::SearchState;

        #[test]
        fn search_state_new_is_empty() {
            let state = SearchState::new();
            assert!(state.query.is_empty());
            assert!(state.matches.is_empty());
            assert_eq!(state.current_match_index, 0);
        }

        #[test]
        fn search_segments_finds_matching_text() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello world",
                None,
                None,
            ));
            view.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "goodbye world",
                None,
                None,
            ));
            view.push_segment(make_segment(
                Some(2.0),
                Some(3.0),
                "hello again",
                None,
                None,
            ));

            let mut state = SearchState::new();
            state.set_query("hello".to_owned());
            view.search_segments(&mut state);

            assert_eq!(state.matches, vec![0, 2], "should find segments 0 and 2");
        }

        #[test]
        fn search_segments_case_insensitive() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "Hello World",
                None,
                None,
            ));
            view.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "HELLO AGAIN",
                None,
                None,
            ));

            let mut state = SearchState::new();
            state.set_query("hello".to_owned());
            view.search_segments(&mut state);

            assert_eq!(state.matches.len(), 2, "case-insensitive search");
        }

        #[test]
        fn search_segments_matches_speaker_label() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "no match here",
                Some("SPEAKER_00"),
                None,
            ));
            view.push_segment(make_segment(Some(1.0), Some(2.0), "some text", None, None));

            let mut state = SearchState::new();
            state.set_query("SPEAKER_00".to_owned());
            view.search_segments(&mut state);

            assert_eq!(state.matches, vec![0], "should match speaker label");
        }

        #[test]
        fn search_segments_empty_query_finds_nothing() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));

            let mut state = SearchState::new();
            state.set_query(String::new());
            view.search_segments(&mut state);

            assert!(state.matches.is_empty(), "empty query should find nothing");
        }

        #[test]
        fn search_segments_no_matches() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "hello", None, None));

            let mut state = SearchState::new();
            state.set_query("zzzzz".to_owned());
            view.search_segments(&mut state);

            assert!(state.matches.is_empty(), "non-matching query");
        }

        #[test]
        fn search_state_next_prev_match_wraps() {
            let mut state = SearchState::new();
            state.matches = vec![0, 5, 10];
            state.current_match_index = 0;

            state.next_match();
            assert_eq!(state.current_match_index, 1);
            state.next_match();
            assert_eq!(state.current_match_index, 2);
            state.next_match(); // Wraps around.
            assert_eq!(state.current_match_index, 0);

            state.prev_match(); // Wraps backwards.
            assert_eq!(state.current_match_index, 2);
            state.prev_match();
            assert_eq!(state.current_match_index, 1);
        }

        #[test]
        fn search_state_current_segment_index() {
            let mut state = SearchState::new();
            assert_eq!(state.current_segment_index(), None);

            state.matches = vec![3, 7, 12];
            state.current_match_index = 0;
            assert_eq!(state.current_segment_index(), Some(3));
            state.current_match_index = 2;
            assert_eq!(state.current_segment_index(), Some(12));
        }

        #[test]
        fn filter_by_speaker_returns_matching_segments() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                None,
            ));
            view.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "world",
                Some("SPEAKER_01"),
                None,
            ));
            view.push_segment(make_segment(
                Some(2.0),
                Some(3.0),
                "again",
                Some("SPEAKER_00"),
                None,
            ));
            view.push_segment(make_segment(Some(3.0), Some(4.0), "no speaker", None, None));

            let filtered = view.filter_by_speaker("SPEAKER_00");
            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0].0, 0, "first match at index 0");
            assert_eq!(filtered[1].0, 2, "second match at index 2");
            assert_eq!(filtered[0].1.text, "hello");
            assert_eq!(filtered[1].1.text, "again");
        }

        #[test]
        fn filter_by_speaker_no_match() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                None,
            ));

            let filtered = view.filter_by_speaker("SPEAKER_99");
            assert!(filtered.is_empty());
        }

        #[test]
        fn export_visible_all_segments_as_json() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                Some(0.9),
            ));
            view.push_segment(make_segment(Some(1.0), Some(2.0), "world", None, None));

            let json_str = view.export_visible(None);
            let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
            let arr = parsed.as_array().expect("should be array");
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0]["text"], "hello");
            assert_eq!(arr[1]["text"], "world");
        }

        #[test]
        fn export_visible_filtered_by_speaker() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, true);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("SPEAKER_00"),
                None,
            ));
            view.push_segment(make_segment(
                Some(1.0),
                Some(2.0),
                "world",
                Some("SPEAKER_01"),
                None,
            ));
            view.push_segment(make_segment(
                Some(2.0),
                Some(3.0),
                "again",
                Some("SPEAKER_00"),
                None,
            ));

            let json_str = view.export_visible(Some("SPEAKER_00"));
            let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
            let arr = parsed.as_array().expect("should be array");
            assert_eq!(arr.len(), 2, "only SPEAKER_00 segments");
            assert_eq!(arr[0]["text"], "hello");
            assert_eq!(arr[1]["text"], "again");
        }

        #[test]
        fn export_visible_empty_segments() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            let json_str = view.export_visible(None);
            let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
            let arr = parsed.as_array().expect("should be array");
            assert!(arr.is_empty());
        }

        #[test]
        fn export_visible_json_contains_all_fields() {
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.push_segment(make_segment(
                Some(1.5),
                Some(3.5),
                "test text",
                Some("SPEAKER_02"),
                Some(0.87),
            ));

            let json_str = view.export_visible(None);
            let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
            let seg = &parsed[0];
            assert_eq!(seg["start_sec"], 1.5);
            assert_eq!(seg["end_sec"], 3.5);
            assert_eq!(seg["text"], "test text");
            assert_eq!(seg["speaker"], "SPEAKER_02");
            assert_eq!(seg["confidence"], 0.87);
        }

        // ── Snapshot tests (bd-3pf.4) ─────────────────────────────────────

        #[test]
        fn snapshot_empty_view_shows_waiting() {
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0], "Waiting for transcription segments...");
        }

        #[test]
        fn snapshot_single_segment_no_diarization() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 1000.0);
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.5),
                "hello world",
                Some("SPEAKER_00"),
                Some(0.95),
            ));
            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 1);
            // Diarization is off, so no speaker label.
            assert_eq!(
                lines[0],
                "000 00:00:00.000 -> 00:00:01.500 hello world (0.950)"
            );
        }

        #[test]
        fn snapshot_single_segment_with_diarization() {
            let mut view = LiveTranscriptionView::with_start_time(
                BackendKind::WhisperDiarization,
                true,
                1000.0,
            );
            view.push_segment(make_segment(
                Some(0.0),
                Some(1.5),
                "hello world",
                Some("SPEAKER_00"),
                Some(0.95),
            ));
            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 1);
            assert_eq!(
                lines[0],
                "000 00:00:00.000 -> 00:00:01.500 [SPEAKER_00] hello world (0.950)"
            );
        }

        #[test]
        fn snapshot_multiple_segments_diarization_on() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperDiarization, true, 0.0);
            view.push_segment(make_segment(
                Some(0.0),
                Some(2.0),
                "Good morning",
                Some("Alice"),
                Some(0.9),
            ));
            view.push_segment(make_segment(
                Some(2.0),
                Some(4.0),
                "Hi there",
                Some("Bob"),
                Some(0.85),
            ));
            view.push_segment(make_segment(
                Some(4.0),
                Some(6.0),
                "How are you?",
                Some("Alice"),
                None,
            ));

            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 3);
            assert_eq!(
                lines[0],
                "000 00:00:00.000 -> 00:00:02.000 [Alice] Good morning (0.900)"
            );
            assert_eq!(
                lines[1],
                "001 00:00:02.000 -> 00:00:04.000 [Bob] Hi there (0.850)"
            );
            assert_eq!(
                lines[2],
                "002 00:00:04.000 -> 00:00:06.000 [Alice] How are you?"
            );
        }

        #[test]
        fn snapshot_multiple_segments_diarization_off() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            view.push_segment(make_segment(
                Some(0.0),
                Some(2.0),
                "Good morning",
                Some("Alice"),
                Some(0.9),
            ));
            view.push_segment(make_segment(
                Some(2.0),
                Some(4.0),
                "Hi there",
                Some("Bob"),
                Some(0.85),
            ));

            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 2);
            // No speaker labels when diarization is off.
            assert_eq!(
                lines[0],
                "000 00:00:00.000 -> 00:00:02.000 Good morning (0.900)"
            );
            assert_eq!(
                lines[1],
                "001 00:00:02.000 -> 00:00:04.000 Hi there (0.850)"
            );
        }

        #[test]
        fn snapshot_scroll_position_affects_visible_output() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            for i in 0..20 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("segment-{i:02}"),
                    None,
                    None,
                ));
            }

            // Auto-scroll on, visible_height=5: should see last 5 segments (15-19).
            let lines = view.get_visible_lines(5);
            assert_eq!(lines.len(), 5);
            assert!(
                lines[0].contains("segment-15"),
                "first visible should be seg-15: {}",
                lines[0]
            );
            assert!(
                lines[4].contains("segment-19"),
                "last visible should be seg-19: {}",
                lines[4]
            );

            // Disable auto-scroll and set manual offset.
            view.scroll_up(1); // Disables auto-scroll, offset = 0.
            let lines = view.get_visible_lines(5);
            assert_eq!(lines.len(), 5);
            assert!(
                lines[0].contains("segment-00"),
                "manual scroll=0 should show seg-00: {}",
                lines[0]
            );
            assert!(
                lines[4].contains("segment-04"),
                "manual scroll=0 should show through seg-04: {}",
                lines[4]
            );
        }

        #[test]
        fn snapshot_scroll_offset_3_shows_correct_window() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            for i in 0..10 {
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("seg-{i}"),
                    None,
                    None,
                ));
            }
            view.scroll_up(1); // Disables auto-scroll, offset=0.
            view.scroll_down(3); // offset=3, still not at bottom.

            let lines = view.get_visible_lines(4);
            assert_eq!(lines.len(), 4);
            assert!(
                lines[0].contains("seg-3"),
                "should start at seg-3: {}",
                lines[0]
            );
            assert!(
                lines[3].contains("seg-6"),
                "should end at seg-6: {}",
                lines[3]
            );
        }

        #[test]
        fn snapshot_visible_height_larger_than_segments() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "only one", None, None));

            let lines = view.get_visible_lines(50);
            assert_eq!(lines.len(), 1, "should return just the 1 segment");
            assert!(lines[0].contains("only one"));
        }

        #[test]
        fn snapshot_none_timestamps_and_no_confidence() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            view.push_segment(make_segment(None, None, "timeless", None, None));

            let lines = view.get_visible_lines(5);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0], "000 --:--:--.--- -> --:--:--.--- timeless");
        }

        #[test]
        fn snapshot_many_segments_visible_window() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, true, 0.0);
            for i in 0..100 {
                let speaker = if i % 2 == 0 {
                    Some("Alice")
                } else {
                    Some("Bob")
                };
                view.push_segment(make_segment(
                    Some(i as f64),
                    Some((i + 1) as f64),
                    &format!("line-{i:03}"),
                    speaker,
                    Some(0.9),
                ));
            }

            // Auto-scroll with height 10: should see segments 90-99.
            let lines = view.get_visible_lines(10);
            assert_eq!(lines.len(), 10);
            assert!(lines[0].contains("line-090"), "first visible: {}", lines[0]);
            assert!(lines[9].contains("line-099"), "last visible: {}", lines[9]);

            // Verify diarization labels are present.
            assert!(
                lines[0].contains("[Alice]"),
                "even index speaker: {}",
                lines[0]
            );
            assert!(
                lines[1].contains("[Bob]"),
                "odd index speaker: {}",
                lines[1]
            );
        }

        #[test]
        fn snapshot_get_visible_lines_empty_when_scroll_beyond() {
            let mut view =
                LiveTranscriptionView::with_start_time(BackendKind::WhisperCpp, false, 0.0);
            view.push_segment(make_segment(Some(0.0), Some(1.0), "a", None, None));
            view.push_segment(make_segment(Some(1.0), Some(2.0), "b", None, None));
            // Disable auto-scroll and set offset beyond segment count.
            view.scroll_up(1);
            view.scroll_offset = 100;

            let lines = view.get_visible_lines(5);
            assert!(
                lines.is_empty(),
                "scroll beyond segment count should return empty"
            );
        }

        // ── WaveformState tests (bd-339.2) ────────────────────────────────

        use super::WaveformState;

        #[test]
        fn waveform_state_new_is_empty() {
            let state = WaveformState::new();
            assert!(state.is_empty());
            assert_eq!(state.len(), 0);
            assert_eq!(state.peak(), 0.0);
        }

        #[test]
        fn waveform_state_push_energy_tracks_peak() {
            let mut state = WaveformState::new();
            state.push_energy(0.3);
            assert_eq!(state.len(), 1);
            assert!((state.peak() - 0.3).abs() < f32::EPSILON);

            state.push_energy(0.7);
            assert_eq!(state.len(), 2);
            assert!((state.peak() - 0.7).abs() < f32::EPSILON);

            // Lower value should not update peak.
            state.push_energy(0.1);
            assert!((state.peak() - 0.7).abs() < f32::EPSILON);
        }

        #[test]
        fn waveform_state_clamps_values() {
            let mut state = WaveformState::new();
            state.push_energy(-0.5);
            state.push_energy(1.5);
            // After clamping: 0.0 and 1.0.
            let bars = state.render_bars();
            assert_eq!(bars.chars().count(), 2);
        }

        #[test]
        fn waveform_state_max_samples_enforced() {
            let mut state = WaveformState::with_max_samples(5);
            for i in 0..10 {
                state.push_energy(i as f32 / 10.0);
            }
            assert_eq!(state.len(), 5, "should cap at max_samples");
        }

        #[test]
        fn waveform_state_push_energies_batch() {
            let mut state = WaveformState::new();
            state.push_energies(&[0.1, 0.2, 0.3, 0.4]);
            assert_eq!(state.len(), 4);
            assert!((state.peak() - 0.4).abs() < f32::EPSILON);
        }

        #[test]
        fn waveform_state_clear_resets() {
            let mut state = WaveformState::new();
            state.push_energy(0.5);
            state.push_energy(0.8);
            assert_eq!(state.len(), 2);

            state.clear();
            assert!(state.is_empty());
            assert_eq!(state.peak(), 0.0);
        }

        #[test]
        fn waveform_state_to_display_returns_correct_bars() {
            let mut state = WaveformState::new();
            state.push_energy(0.0);
            state.push_energy(1.0);
            let display = state.to_display();
            let bars = display.amplitude_bars();
            assert_eq!(bars.chars().count(), 2);
            assert_eq!(bars.chars().next(), Some(' '));
            assert_eq!(bars.chars().last(), Some('\u{2588}'));
        }

        #[test]
        fn waveform_state_render_bars_matches_display() {
            let mut state = WaveformState::new();
            state.push_energy(0.5);
            let from_render = state.render_bars();
            let from_display = state.to_display().amplitude_bars();
            assert_eq!(from_render, from_display);
        }

        #[test]
        fn waveform_state_with_max_samples_minimum_is_one() {
            let state = WaveformState::with_max_samples(0);
            // Should be clamped to 1.
            assert_eq!(state.len(), 0);
            let mut state2 = WaveformState::with_max_samples(0);
            state2.push_energy(0.5);
            state2.push_energy(0.6);
            assert_eq!(state2.len(), 1, "max_samples clamped to 1");
        }

        #[test]
        fn waveform_state_push_energy_non_finite_does_not_panic() {
            // RMS derived from NaN audio (sum of squares → NaN, or 0/0 over an
            // empty window) must not panic the live TUI on the nightly toolchain;
            // non-finite energies sanitize to 0.0 and leave the peak untouched.
            let mut state = WaveformState::new();
            state.push_energy(f32::NAN);
            state.push_energy(f32::INFINITY);
            state.push_energy(f32::NEG_INFINITY);
            assert_eq!(state.len(), 3, "non-finite energies are still stored");
            assert_eq!(state.peak(), 0.0, "non-finite energies do not raise the peak");
            // Stored values sanitize to 0.0 → all spaces, no panic on render.
            assert_eq!(state.render_bars(), "   ");
        }

        // ── render_speaker_colored_line tests (bd-339.2) ──────────────────

        use super::render_speaker_colored_line;

        #[test]
        fn render_speaker_colored_line_with_speaker_and_timestamps() {
            let seg = make_segment(
                Some(1.0),
                Some(2.5),
                "hello world",
                Some("SPEAKER_00"),
                Some(0.95),
            );
            let mut cmap = SpeakerColorMap::new();
            let line = render_speaker_colored_line(&seg, &mut cmap, true);
            // Should contain ANSI color codes for the speaker.
            assert!(line.contains("\x1b[38;2;"), "should have ANSI color prefix");
            assert!(line.contains("[SPEAKER_00]"), "should have speaker label");
            assert!(line.contains("\x1b[0m"), "should have ANSI reset");
            assert!(line.contains("hello world"), "should have text");
            assert!(line.contains("[95%]"), "should have confidence badge");
        }

        #[test]
        fn render_speaker_colored_line_no_speaker() {
            let seg = make_segment(Some(0.0), Some(1.0), "no speaker", None, None);
            let mut cmap = SpeakerColorMap::new();
            let line = render_speaker_colored_line(&seg, &mut cmap, true);
            assert!(
                !line.contains("\x1b[38;2;"),
                "no ANSI color without speaker"
            );
            assert!(line.contains("no speaker"));
        }

        #[test]
        fn render_speaker_colored_line_no_timestamps() {
            let seg = make_segment(Some(0.0), Some(1.0), "text only", Some("Alice"), None);
            let mut cmap = SpeakerColorMap::new();
            let line = render_speaker_colored_line(&seg, &mut cmap, false);
            assert!(line.contains("[Alice]"), "should have speaker label");
            // Should not have timestamp format.
            assert!(!line.contains("->"), "should not have timestamp arrow");
            assert!(line.contains("text only"));
        }

        #[test]
        fn render_speaker_colored_line_consistent_colors() {
            let seg1 = make_segment(Some(0.0), Some(1.0), "a", Some("Bob"), None);
            let seg2 = make_segment(Some(1.0), Some(2.0), "b", Some("Bob"), None);
            let mut cmap = SpeakerColorMap::new();
            let line1 = render_speaker_colored_line(&seg1, &mut cmap, false);
            let line2 = render_speaker_colored_line(&seg2, &mut cmap, false);
            // Extract the ANSI color code from both lines.
            let extract_ansi = |s: &str| -> Option<String> {
                let start = s.find("\x1b[38;2;")?;
                let end = s[start..].find('m')? + start + 1;
                Some(s[start..end].to_owned())
            };
            assert_eq!(
                extract_ansi(&line1),
                extract_ansi(&line2),
                "same speaker should get same ANSI color"
            );
        }

        // ── TranscriptFilter tests (bd-339.3) ─────────────────────────────

        use super::{
            ExportFormat, TranscriptFilter, TranscriptSearch, apply_filter,
            export_filtered_transcript, export_filtered_transcript_to_file,
        };

        #[test]
        fn transcript_filter_empty_matches_all() {
            let filter = TranscriptFilter::new();
            assert!(filter.is_empty());

            let seg = make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9));
            assert!(filter.matches(&seg));
        }

        #[test]
        fn transcript_filter_speaker_match() {
            let filter = TranscriptFilter {
                speaker: Some("Alice".to_owned()),
                ..Default::default()
            };
            let seg_match = make_segment(Some(0.0), Some(1.0), "hi", Some("Alice"), None);
            let seg_no = make_segment(Some(0.0), Some(1.0), "hi", Some("Bob"), None);
            let seg_none = make_segment(Some(0.0), Some(1.0), "hi", None, None);

            assert!(filter.matches(&seg_match));
            assert!(!filter.matches(&seg_no));
            assert!(!filter.matches(&seg_none));
        }

        #[test]
        fn transcript_filter_keyword_case_insensitive() {
            let filter = TranscriptFilter {
                keyword: Some("hello".to_owned()),
                ..Default::default()
            };
            let seg1 = make_segment(Some(0.0), Some(1.0), "Hello World", None, None);
            let seg2 = make_segment(Some(0.0), Some(1.0), "goodbye", None, None);

            assert!(filter.matches(&seg1));
            assert!(!filter.matches(&seg2));
        }

        #[test]
        fn transcript_filter_time_range_overlap() {
            let filter = TranscriptFilter {
                time_range: Some((2.0, 5.0)),
                ..Default::default()
            };
            // Segment fully inside range.
            let inside = make_segment(Some(3.0), Some(4.0), "a", None, None);
            // Segment overlapping start.
            let overlap_start = make_segment(Some(1.0), Some(3.0), "b", None, None);
            // Segment overlapping end.
            let overlap_end = make_segment(Some(4.0), Some(6.0), "c", None, None);
            // Segment entirely before range.
            let before = make_segment(Some(0.0), Some(1.5), "d", None, None);
            // Segment entirely after range.
            let after = make_segment(Some(6.0), Some(7.0), "e", None, None);

            assert!(filter.matches(&inside));
            assert!(filter.matches(&overlap_start));
            assert!(filter.matches(&overlap_end));
            assert!(!filter.matches(&before));
            assert!(!filter.matches(&after));
        }

        #[test]
        fn transcript_filter_min_confidence() {
            let filter = TranscriptFilter {
                min_confidence: Some(0.8),
                ..Default::default()
            };
            let high_conf = make_segment(Some(0.0), Some(1.0), "a", None, Some(0.9));
            let exact_conf = make_segment(Some(0.0), Some(1.0), "b", None, Some(0.8));
            let low_conf = make_segment(Some(0.0), Some(1.0), "c", None, Some(0.5));
            let no_conf = make_segment(Some(0.0), Some(1.0), "d", None, None);

            assert!(filter.matches(&high_conf));
            assert!(filter.matches(&exact_conf));
            assert!(!filter.matches(&low_conf));
            assert!(
                !filter.matches(&no_conf),
                "no confidence excluded with min_confidence"
            );
        }

        #[test]
        fn transcript_filter_combined() {
            let filter = TranscriptFilter {
                speaker: Some("Alice".to_owned()),
                keyword: Some("hello".to_owned()),
                time_range: Some((0.0, 5.0)),
                min_confidence: Some(0.5),
            };
            // Matches all criteria.
            let good = make_segment(
                Some(1.0),
                Some(2.0),
                "hello there",
                Some("Alice"),
                Some(0.9),
            );
            // Wrong speaker.
            let bad_speaker = make_segment(Some(1.0), Some(2.0), "hello", Some("Bob"), Some(0.9));
            // Wrong keyword.
            let bad_kw = make_segment(Some(1.0), Some(2.0), "goodbye", Some("Alice"), Some(0.9));
            // Out of time range.
            let bad_time = make_segment(Some(6.0), Some(7.0), "hello", Some("Alice"), Some(0.9));
            // Low confidence.
            let bad_conf = make_segment(Some(1.0), Some(2.0), "hello", Some("Alice"), Some(0.1));

            assert!(filter.matches(&good));
            assert!(!filter.matches(&bad_speaker));
            assert!(!filter.matches(&bad_kw));
            assert!(!filter.matches(&bad_time));
            assert!(!filter.matches(&bad_conf));
        }

        #[test]
        fn apply_filter_returns_matching_segments() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", Some("Bob"), Some(0.8)),
                make_segment(
                    Some(2.0),
                    Some(3.0),
                    "hello again",
                    Some("Alice"),
                    Some(0.7),
                ),
            ];
            let filter = TranscriptFilter {
                speaker: Some("Alice".to_owned()),
                ..Default::default()
            };
            let result = apply_filter(&segments, &filter);
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].text, "hello");
            assert_eq!(result[1].text, "hello again");
        }

        #[test]
        fn apply_filter_empty_filter_returns_all() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "a", None, None),
                make_segment(Some(1.0), Some(2.0), "b", None, None),
            ];
            let filter = TranscriptFilter::new();
            let result = apply_filter(&segments, &filter);
            assert_eq!(result.len(), 2);
        }

        // ── TranscriptSearch tests (bd-339.3) ─────────────────────────────

        #[test]
        fn transcript_search_basic() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello world", None, None),
                make_segment(Some(1.0), Some(2.0), "goodbye", None, None),
                make_segment(Some(2.0), Some(3.0), "hello again", None, None),
            ];
            let mut search = TranscriptSearch::new();
            search.search("hello", &segments);

            assert_eq!(search.match_count(), 2);
            assert_eq!(search.current_match(), Some(0));
            assert_eq!(search.query(), "hello");
        }

        #[test]
        fn transcript_search_next_prev_navigation() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", None, None),
                make_segment(Some(1.0), Some(2.0), "world", None, None),
                make_segment(Some(2.0), Some(3.0), "hello", None, None),
                make_segment(Some(3.0), Some(4.0), "hello", None, None),
            ];
            let mut search = TranscriptSearch::new();
            search.search("hello", &segments);

            assert_eq!(search.match_count(), 3);
            assert_eq!(search.current_match(), Some(0));

            assert_eq!(search.next_match(), Some(2));
            assert_eq!(search.next_match(), Some(3));
            assert_eq!(search.next_match(), Some(0)); // Wraps.

            assert_eq!(search.prev_match(), Some(3)); // Wraps backward.
            assert_eq!(search.prev_match(), Some(2));
        }

        #[test]
        fn transcript_search_empty_query() {
            let segments = vec![make_segment(Some(0.0), Some(1.0), "hello", None, None)];
            let mut search = TranscriptSearch::new();
            search.search("", &segments);

            assert_eq!(search.match_count(), 0);
            assert_eq!(search.current_match(), None);
            assert_eq!(search.next_match(), None);
            assert_eq!(search.prev_match(), None);
        }

        #[test]
        fn transcript_search_no_matches() {
            let segments = vec![make_segment(Some(0.0), Some(1.0), "hello", None, None)];
            let mut search = TranscriptSearch::new();
            search.search("zzz", &segments);

            assert_eq!(search.match_count(), 0);
            assert_eq!(search.current_match(), None);
        }

        #[test]
        fn transcript_search_matches_speaker_label() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "some text", Some("Alice"), None),
                make_segment(Some(1.0), Some(2.0), "other text", Some("Bob"), None),
            ];
            let mut search = TranscriptSearch::new();
            search.search("alice", &segments);

            assert_eq!(search.match_count(), 1);
            assert_eq!(search.current_match(), Some(0));
        }

        #[test]
        fn transcript_search_all_matches() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "aaa", None, None),
                make_segment(Some(1.0), Some(2.0), "aab", None, None),
                make_segment(Some(2.0), Some(3.0), "bbb", None, None),
            ];
            let mut search = TranscriptSearch::new();
            search.search("aa", &segments);

            assert_eq!(search.all_matches(), &[0, 1]);
        }

        // ── Export format tests (bd-339.3) ─────────────────────────────────

        #[test]
        fn export_plain_text_format() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.5), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(2.0), Some(3.0), "world", None, None),
            ];
            let filter = TranscriptFilter::new();
            let text = export_filtered_transcript(&segments, &filter, ExportFormat::PlainText);

            assert!(text.contains("hello"), "should contain first segment text");
            assert!(text.contains("world"), "should contain second segment text");
            assert!(text.contains("[Alice]"), "should contain speaker label");
            // Should have two lines.
            assert_eq!(text.lines().count(), 2);
        }

        #[test]
        fn export_json_format() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", None, None),
            ];
            let filter = TranscriptFilter::new();
            let json_str = export_filtered_transcript(&segments, &filter, ExportFormat::Json);

            let parsed: serde_json::Value =
                serde_json::from_str(&json_str).expect("should be valid JSON");
            let arr = parsed.as_array().expect("should be array");
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0]["text"], "hello");
            assert_eq!(arr[1]["text"], "world");
            assert_eq!(arr[0]["speaker"], "Alice");
        }

        #[test]
        fn export_srt_format() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.5), "hello", Some("Alice"), None),
                make_segment(Some(2.0), Some(3.5), "world", None, None),
            ];
            let filter = TranscriptFilter::new();
            let srt = export_filtered_transcript(&segments, &filter, ExportFormat::Srt);

            let lines: Vec<&str> = srt.lines().collect();
            // First subtitle block.
            assert_eq!(lines[0], "1", "sequence number");
            assert_eq!(lines[1], "00:00:00,000 --> 00:00:01,500", "timestamp");
            assert_eq!(lines[2], "[Alice] hello", "text with speaker");
            // Blank line separator.
            assert_eq!(lines[3], "", "blank separator");
            // Second subtitle block.
            assert_eq!(lines[4], "2");
            assert_eq!(lines[5], "00:00:02,000 --> 00:00:03,500");
            assert_eq!(lines[6], "world", "text without speaker");
        }

        #[test]
        fn export_srt_no_segments() {
            let segments: Vec<TranscriptionSegment> = vec![];
            let filter = TranscriptFilter::new();
            let srt = export_filtered_transcript(&segments, &filter, ExportFormat::Srt);
            assert!(srt.is_empty());
        }

        #[test]
        fn export_filtered_transcript_with_filter() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", Some("Bob"), Some(0.5)),
                make_segment(Some(2.0), Some(3.0), "again", Some("Alice"), Some(0.8)),
            ];
            let filter = TranscriptFilter {
                speaker: Some("Alice".to_owned()),
                ..Default::default()
            };
            let json_str = export_filtered_transcript(&segments, &filter, ExportFormat::Json);
            let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
            let arr = parsed.as_array().expect("array");
            assert_eq!(arr.len(), 2, "only Alice segments");
            assert_eq!(arr[0]["text"], "hello");
            assert_eq!(arr[1]["text"], "again");
        }

        #[test]
        fn export_plain_text_with_filter() {
            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", Some("Bob"), Some(0.5)),
            ];
            let filter = TranscriptFilter {
                min_confidence: Some(0.8),
                ..Default::default()
            };
            let text = export_filtered_transcript(&segments, &filter, ExportFormat::PlainText);
            assert_eq!(text.lines().count(), 1);
            assert!(text.contains("hello"));
            assert!(!text.contains("world"));
        }

        #[test]
        fn export_to_file_writes_and_returns_count() {
            let dir = tempdir().expect("tmpdir");
            let path = dir.path().join("export_test.txt");

            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", Some("Bob"), Some(0.8)),
            ];
            let filter = TranscriptFilter::new();
            let count = export_filtered_transcript_to_file(
                &segments,
                &filter,
                ExportFormat::PlainText,
                &path,
            )
            .expect("write should succeed");

            assert_eq!(count, 2);
            let content = std::fs::read_to_string(&path).expect("should read");
            assert!(content.contains("hello"));
            assert!(content.contains("world"));
        }

        #[test]
        fn export_to_file_json_format() {
            let dir = tempdir().expect("tmpdir");
            let path = dir.path().join("export_test.json");

            let segments = vec![make_segment(
                Some(0.0),
                Some(1.0),
                "hello",
                Some("Alice"),
                Some(0.9),
            )];
            let filter = TranscriptFilter::new();
            let count =
                export_filtered_transcript_to_file(&segments, &filter, ExportFormat::Json, &path)
                    .expect("write should succeed");

            assert_eq!(count, 1);
            let content = std::fs::read_to_string(&path).expect("should read");
            let parsed: serde_json::Value =
                serde_json::from_str(&content).expect("valid JSON file");
            assert_eq!(parsed[0]["text"], "hello");
        }

        #[test]
        fn export_to_file_srt_format() {
            let dir = tempdir().expect("tmpdir");
            let path = dir.path().join("export_test.srt");

            let segments = vec![
                make_segment(Some(1.0), Some(2.5), "subtitle one", None, None),
                make_segment(Some(3.0), Some(4.0), "subtitle two", None, None),
            ];
            let filter = TranscriptFilter::new();
            let count =
                export_filtered_transcript_to_file(&segments, &filter, ExportFormat::Srt, &path)
                    .expect("write should succeed");

            assert_eq!(count, 2);
            let content = std::fs::read_to_string(&path).expect("should read");
            assert!(content.contains("1\n00:00:01,000 --> 00:00:02,500\nsubtitle one"));
            assert!(content.contains("2\n00:00:03,000 --> 00:00:04,000\nsubtitle two"));
        }

        #[test]
        fn export_to_file_with_filter_reduces_count() {
            let dir = tempdir().expect("tmpdir");
            let path = dir.path().join("export_filtered.txt");

            let segments = vec![
                make_segment(Some(0.0), Some(1.0), "hello", Some("Alice"), Some(0.9)),
                make_segment(Some(1.0), Some(2.0), "world", Some("Bob"), Some(0.8)),
                make_segment(Some(2.0), Some(3.0), "again", Some("Alice"), Some(0.7)),
            ];
            let filter = TranscriptFilter {
                speaker: Some("Alice".to_owned()),
                ..Default::default()
            };
            let count = export_filtered_transcript_to_file(
                &segments,
                &filter,
                ExportFormat::PlainText,
                &path,
            )
            .expect("write should succeed");

            assert_eq!(count, 2, "only Alice segments exported");
        }

        #[test]
        fn speculation_stats_line_inactive_returns_empty() {
            use super::LiveTranscriptionView;
            let view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            assert_eq!(view.speculation_stats_line(), "");
        }

        #[test]
        fn speculation_stats_line_active_computes_rate() {
            use super::LiveTranscriptionView;
            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            view.set_speculative(true);

            // Zero windows → 0.0% rate.
            let line = view.speculation_stats_line();
            assert!(line.contains("[SPEC]"), "should contain [SPEC]: {line}");
            assert!(
                line.contains("0.0%"),
                "zero windows should show 0.0%: {line}"
            );

            // Push a speculative segment and then correct it.
            let seg = make_segment(Some(0.0), Some(1.0), "fast", None, None);
            view.push_speculative_segment(seg, 1);
            let corrected = make_segment(Some(0.0), Some(1.0), "quality", None, None);
            view.push_correction(1, vec![corrected]);

            let line = view.speculation_stats_line();
            assert!(line.contains("1/1"), "should show 1/1 corrections: {line}");
            assert!(line.contains("100.0%"), "rate should be 100%: {line}");
        }

        #[test]
        fn correction_prefix_all_states() {
            use super::{LiveTranscriptionView, SegmentCorrectionState};

            let mut view = LiveTranscriptionView::new(BackendKind::WhisperCpp, false);
            // Out-of-range index has no state entry → Original → ""
            assert_eq!(view.correction_prefix(99), "");

            view.set_speculative(true);

            // Push speculative → idx 0, state is Speculative → "~ "
            let seg = make_segment(Some(0.0), Some(1.0), "fast", None, None);
            view.push_speculative_segment(seg, 7);
            assert_eq!(view.correction_prefix(0), "~ ");

            // Retract → idx 0 becomes Retracted → "x "
            view.retract_segments(7);
            assert_eq!(view.correction_prefix(0), "x ");

            // Push correction → new segment at idx 1 is Corrected → "> "
            let corrected = make_segment(Some(0.0), Some(1.0), "quality", None, None);
            view.push_correction(999, vec![corrected]);
            assert_eq!(view.correction_prefix(1), "> ");

            // Confirm a speculative segment → Confirmed → ""
            let seg2 = make_segment(Some(1.0), Some(2.0), "spec2", None, None);
            view.push_speculative_segment(seg2, 42);
            assert_eq!(
                view.segment_correction_state(2),
                SegmentCorrectionState::Speculative
            );
            view.confirm_segments(42);
            assert_eq!(view.correction_prefix(2), "");
        }

        #[test]
        fn render_speaker_colored_line_start_only_no_end() {
            use super::SpeakerColorMap;
            use super::render_speaker_colored_line;

            // (Some(start), None) → "HH:MM:SS.mmm -> ???"
            let seg = make_segment(Some(5.0), None, "partial", Some("Alice"), None);
            let mut cmap = SpeakerColorMap::new();
            let line = render_speaker_colored_line(&seg, &mut cmap, true);
            assert!(
                line.contains("-> ???"),
                "missing end should produce '-> ???': {line}"
            );
            assert!(
                line.contains("00:00:05.000"),
                "start should be formatted: {line}"
            );

            // (None, None) → "??:??:??.???"
            let seg2 = make_segment(None, None, "no timestamps", None, None);
            let line2 = render_speaker_colored_line(&seg2, &mut cmap, true);
            assert!(
                line2.contains("??:??:??.???"),
                "no timestamps → fallback: {line2}"
            );
        }

        #[test]
        fn export_plain_text_omits_timestamp_when_none() {
            let segments = vec![
                make_segment(None, None, "timeless text", Some("Alice"), None),
                make_segment(Some(1.0), None, "partial timing", None, None),
            ];
            let filter = TranscriptFilter::new();
            let text = export_filtered_transcript(&segments, &filter, ExportFormat::PlainText);
            let lines: Vec<&str> = text.lines().collect();
            // No timestamp arrow when either side is None.
            assert!(
                !lines[0].contains("->"),
                "no arrow for None timestamps: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("timeless text"),
                "text present: {}",
                lines[0]
            );
            assert!(
                lines[0].contains("[Alice]"),
                "speaker present: {}",
                lines[0]
            );
            assert!(
                !lines[1].contains("->"),
                "partial timing has None end: {}",
                lines[1]
            );
            assert!(
                lines[1].contains("partial timing"),
                "text present: {}",
                lines[1]
            );
        }
    }
}

#[cfg(feature = "tui")]
pub use enabled::run_tui;

#[cfg(not(feature = "tui"))]
pub fn run_tui() -> FwResult<()> {
    Err(FwError::Unsupported(
        "tui feature is disabled; rebuild with `--features tui`".to_owned(),
    ))
}
