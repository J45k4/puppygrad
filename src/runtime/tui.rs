//! Interactive front end to the same model contract used by `puppygrad llm`.
use super::{
    catalog::{self, Entry, Status},
    llm,
    llm_ffi::{Generation, Model},
};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
};
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    DefaultTerminal, Frame,
};
use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
use unicode_segmentation::UnicodeSegmentation;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Default, clap::Args)]
pub struct Options {
    /// Device for inference; automatically prefers CUDA, then HIP, then CPU.
    #[arg(long)]
    pub device: Option<String>,
    /// Directory containing custom *.model.json manifests and .pup programs.
    #[arg(long)]
    pub catalog: Option<PathBuf>,
    /// Cache for downloaded assets, bundled programs and compiled kernels.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Load and warm the selected downloaded model only after a prompt is sent.
    #[arg(long)]
    pub no_warmup: bool,
    #[arg(skip)]
    pub max_memory: Option<super::memory_limit::MemoryLimit>,
}

enum Request {
    Download(usize),
    Warmup {
        index: usize,
        device: String,
        limit: Option<u64>,
    },
    Generate {
        index: usize,
        device: String,
        prompt: String,
        temperature: f32,
        limit: Option<u64>,
    },
    Shutdown,
}
enum Update {
    Status(String),
    Warmed {
        error: Option<String>,
        elapsed: Duration,
    },
    Progress {
        file: String,
        bytes: u64,
        total: Option<u64>,
    },
    Chunk(String),
    Done {
        tokens: usize,
        reason: u32,
        elapsed: Duration,
        first_token: Option<Duration>,
    },
    Downloaded,
    Error(String),
}

struct App {
    entries: Vec<Entry>,
    cache: PathBuf,
    selected: usize,
    cursor: ListState,
    browser: bool,
    input: Composer,
    transcript: String,
    status: String,
    device: String,
    temperature: f32,
    limit: Option<u64>,
    busy: bool,
    warming: bool,
    auto_warmup: bool,
    quitting: bool,
    scroll: u16,
    clock: Instant,
    requests: Sender<Request>,
    cancel: Arc<AtomicBool>,
}

impl App {
    fn new(
        entries: Vec<Entry>,
        cache: PathBuf,
        device: String,
        requests: Sender<Request>,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        let selected = entries
            .iter()
            .position(|e| e.manifest.id == "qwen3-0.6b")
            .unwrap_or(0);
        let mut cursor = ListState::default();
        cursor.select(Some(selected));
        Self {
            entries, cache, selected, cursor,
            browser: false,
            input: Composer::default(),
            transcript: "Welcome to Puppygrad. Type /model to choose or download a model.\n\nEach prompt is independent; the loaded model is reused. Type /help for commands.\n".into(),
            status: "Ready".into(), device,
            temperature: 0.7, limit: None,
            busy: false, warming: false, auto_warmup: true, quitting: false, scroll: 0,
            clock: Instant::now(), requests, cancel,
        }
    }

    fn warmup(&mut self) {
        if !self.auto_warmup
            || self.busy
            || self.entries[self.selected].status(&self.cache) != Status::Downloaded
        {
            return;
        }
        self.cancel.store(false, Ordering::Relaxed);
        if self
            .requests
            .send(Request::Warmup {
                index: self.selected,
                device: self.device.clone(),
                limit: self.limit,
            })
            .is_ok()
        {
            self.busy = true;
            self.warming = true;
            self.status = format!(
                "Warming {} on {}… You can type a prompt.",
                self.entries[self.selected].manifest.name, self.device
            );
        }
    }

    fn download(&mut self, index: usize) {
        if self.busy {
            self.status = "An operation is already running; Esc stops it.".into();
            return;
        }
        if self.entries[index].status(&self.cache) == Status::Downloaded {
            self.status = "Already downloaded. Press Enter to select this model.".into();
            return;
        }
        self.cancel.store(false, Ordering::Relaxed);
        if self.requests.send(Request::Download(index)).is_ok() {
            self.busy = true;
            self.status = format!("Downloading {}…", self.entries[index].manifest.name);
        }
    }

    fn submit(&mut self) {
        let text = self.input.text.trim().to_owned();
        if text.is_empty() {
            self.input.clear();
            return;
        }
        if text.starts_with('/') {
            self.input.clear();
            self.command(&text);
            return;
        }
        if self.busy && (!self.warming || self.cancel.load(Ordering::Relaxed)) {
            self.status = "Wait for completion, or press Esc to stop.".into();
            return;
        }
        if self.entries[self.selected].status(&self.cache) != Status::Downloaded {
            self.cursor.select(Some(self.selected));
            self.browser = true;
            self.status = "Download this model with Enter or d, then submit your prompt.".into();
            return;
        }
        self.input.clear();
        self.transcript.push_str(&format!(
            "\nYou: {text}\n\n{}:\n",
            self.entries[self.selected].manifest.name
        ));
        self.scroll = 0;
        self.cancel.store(false, Ordering::Relaxed);
        let queued = self.warming;
        self.warming = false;
        self.busy = self
            .requests
            .send(Request::Generate {
                index: self.selected,
                device: self.device.clone(),
                prompt: text.into(),
                temperature: self.temperature,
                limit: self.limit,
            })
            .is_ok();
        self.status = if queued {
            "Prompt queued; finishing model warmup…"
        } else {
            "Preparing model…"
        }
        .into();
    }

    fn command(&mut self, text: &str) {
        let mut parts = text.split_whitespace();
        match parts.next().unwrap_or("") {
            "/model" => {
                if let Some(id) = parts.next() {
                    if self.busy { self.status = "Stop the current operation before switching models.".into(); return; }
                    if let Some(index) = self.entries.iter().position(|e| e.manifest.id == id) {
                        self.selected = index;
                        self.cursor.select(Some(index));
                        self.status = format!("Selected {}", self.entries[index].manifest.name);
                        self.warmup();
                    } else { self.status = format!("Unknown model {id}; use /model to browse."); }
                } else { self.cursor.select(Some(self.selected)); self.browser = true; }
            }
            "/download" => {
                let index = match parts.next() {
                    Some(id) => self.entries.iter().position(|e| e.manifest.id == id),
                    None => Some(self.selected),
                };
                if let Some(index) = index { self.download(index); }
                else { self.status = "Unknown model. Use /model to browse.".into(); }
            }
            "/device" => {
                if self.busy { self.status = "Stop the current operation before changing device.".into(); return; }
                if let Some(value) = parts.next() {
                    if value == "cpu" || crate::compiler::gpu::device(value).is_ok() {
                        self.device = value.into();
                        self.status = format!("Device: {value}");
                        self.warmup();
                    } else { self.status = "Use /device cpu, cuda:0 or hip:0.".into(); }
                } else { self.status = format!("Device: {}", self.device); }
            }
            "/temperature" => {
                if let Some(value) = parts.next().and_then(|s| s.parse::<f32>().ok()).filter(|v| v.is_finite() && *v >= 0.) {
                    self.temperature = value;
                    self.status = format!("Temperature: {value}");
                } else { self.status = "Use /temperature followed by a finite number >= 0.".into(); }
            }
            "/tokens" => {
                match parts.next() {
                    Some("auto") | Some("none") => {
                        self.limit = None; self.status = "Output length: automatic".into();
                    }
                    Some(value) if value.parse::<u64>().is_ok_and(|n| n > 0 && n <= 1_000_000) => {
                        let value = value.parse().unwrap();
                        self.limit = Some(value); self.status = format!("Token limit: {value}");
                    }
                    _ => self.status = "Use /tokens auto or /tokens N (1–1000000).".into(),
                }
            }
            "/clear" => self.transcript.clear(),
            "/quit" | "/exit" => self.quit(),
            "/help" => self.transcript.push_str("\n/model [id] — browse or select models\n/download [id] — download missing assets\n/device cpu|cuda:0|hip:0 — choose device\n/temperature N — sampling temperature\n/tokens auto|N — automatic output length (default) or a response cap\n/clear — clear display\n/quit — stop and exit\n\nEnter submits. Shift+Enter inserts a newline. Ctrl+A selects the whole prompt; Shift+arrows select text.\nCtrl+Left/Right moves by word; add Shift to select words.\nCtrl+C copies selected text (otherwise quits); Ctrl+X cuts; Ctrl+V pastes.\nTerminal paste with Ctrl+Shift+V also works.\nBackspace/Delete removes selected text; typing or pasting replaces it.\nEsc stops an operation or closes the browser.\nPageUp/PageDown scroll. Prompts are independent.\n"),
            _ => self.status = "Unknown command. Type /help.".into(),
        }
    }

    fn quit(&mut self) {
        self.quitting = true;
        if self.busy {
            self.cancel.store(true, Ordering::Relaxed);
            self.status = "Stopping; waiting for the current operation to return…".into();
        }
    }

    fn copy_selection(&mut self, cut: bool) {
        let Some(range) = self.input.selection() else {
            return;
        };
        match clipboard_write(&self.input.text[range]) {
            Ok(()) => {
                if cut {
                    self.input.delete();
                }
                self.status = if cut {
                    "Cut to clipboard"
                } else {
                    "Copied to clipboard"
                }
                .into();
            }
            Err(error) => self.status = format!("Clipboard: {error}"),
        }
    }

    fn paste_clipboard(&mut self) {
        match clipboard_read() {
            Ok(text) => {
                self.input
                    .insert(&text.replace("\r\n", "\n").replace('\r', "\n"));
                self.status = "Pasted from clipboard".into();
            }
            Err(error) => self.status = format!("Clipboard: {error}"),
        }
    }

    fn update(&mut self, update: Update) {
        match update {
            Update::Status(status) => self.status = status,
            Update::Warmed { error, elapsed } => {
                // A submitted prompt already queued behind warmup owns busy now.
                if self.warming {
                    self.warming = false;
                    self.busy = false;
                    self.status = error.map_or_else(
                        || format!("Ready · warmed in {:.2}s", elapsed.as_secs_f64()),
                        |error| format!("Warmup: {error}"),
                    );
                }
            }
            Update::Progress { file, bytes, total } => {
                self.status = match total {
                    Some(total) => format!(
                        "Downloading {file}: {:.1} / {:.1} MiB ({:.0}%)",
                        bytes as f64 / 1048576.,
                        total as f64 / 1048576.,
                        bytes as f64 * 100. / total.max(1) as f64
                    ),
                    None => format!("Downloading {file}: {:.1} MiB", bytes as f64 / 1048576.),
                }
            }
            Update::Chunk(text) => {
                self.transcript.push_str(&text);
                self.scroll = 0;
            }
            Update::Done {
                tokens,
                reason,
                elapsed,
                first_token,
            } => {
                self.busy = false;
                self.transcript.push('\n');
                let stop = match reason {
                    crate::runtime::llm_ffi::DONE_LIMIT => "Token limit reached · ",
                    crate::runtime::llm_ffi::DONE_CONTEXT => "Context limit reached · ",
                    crate::runtime::llm_ffi::DONE_MEMORY => "Memory limit reached · ",
                    _ => "Complete · ",
                };
                self.status = format!(
                    "{stop}{tokens} tokens · {:.2}s · {:.1} tokens/s · first text {}",
                    elapsed.as_secs_f64(),
                    tokens as f64 / elapsed.as_secs_f64().max(0.000001),
                    first_token.map_or_else(|| "—".into(), |d| format!("{:.2}s", d.as_secs_f64()))
                );
            }
            Update::Downloaded => {
                self.busy = false;
                self.status = "Download complete. Press Enter to select the model.".into();
                self.warmup();
            }
            Update::Error(error) => {
                self.busy = false;
                self.status = error.clone();
                self.transcript.push_str(&format!("\n{error}\n"));
            }
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let layout = self
            .input
            .layout(frame.area().width.saturating_sub(1).max(1) as usize);
        let composer_height = layout
            .rows
            .len()
            .min(frame.area().height.saturating_sub(5).clamp(1, 6) as usize)
            as u16;
        let [header, body, input, status] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(composer_height + 1),
            Constraint::Length(1),
        ])
        .areas(frame.area());
        let title = format!(
            "Puppygrad · {} · {} · temperature {} · {}",
            self.entries[self.selected].manifest.name,
            self.device,
            self.temperature,
            self.limit
                .map_or_else(|| "output auto".into(), |n| format!("max {n} tokens"))
        );
        frame.render_widget(
            Paragraph::new(title).block(Block::new().borders(Borders::BOTTOM)),
            header,
        );
        if self.browser {
            let items: Vec<_> = self
                .entries
                .iter()
                .enumerate()
                .map(|(i, entry)| {
                    let selected = if i == self.selected {
                        " [selected]"
                    } else {
                        ""
                    };
                    ListItem::new(vec![
                        Line::from(format!(
                            "{}{} — {}",
                            entry.manifest.name,
                            selected,
                            entry.status(&self.cache)
                        )),
                        Line::from(format!(
                            "  {} · {}",
                            entry.manifest.id, entry.manifest.description
                        )),
                    ])
                })
                .collect();
            frame.render_stateful_widget(
                List::new(items)
                    .block(Block::new().title(" Models "))
                    .highlight_style(Style::default().bg(Color::DarkGray))
                    .highlight_symbol("› "),
                body,
                &mut self.cursor,
            );
        } else {
            let paragraph = Paragraph::new(self.transcript.as_str()).wrap(Wrap { trim: false });
            let height = paragraph
                .line_count(body.width.max(1))
                .saturating_sub(body.height as usize)
                .min(u16::MAX as usize) as u16;
            frame.render_widget(
                paragraph.scroll((height.saturating_sub(self.scroll), 0)),
                body,
            );
        }
        let status_text = if self.busy {
            let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            format!(
                "{} {}",
                spinner[(self.clock.elapsed().as_millis() / 100 % 10) as usize],
                self.status
            )
        } else {
            self.status.clone()
        };
        let hints = if self.browser {
            " · ↑↓ browse · Enter select/download · d download · Esc close"
        } else {
            " · Enter send · Shift+Enter newline · /model · /help · Ctrl-C quit"
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    status_text,
                    Style::default().fg(if self.busy {
                        Color::Yellow
                    } else {
                        Color::Cyan
                    }),
                ),
                Span::styled(hints, Style::default().fg(Color::DarkGray)),
            ])),
            status,
        );
        let title = if self.browser {
            " Close model browser to enter a prompt "
        } else {
            " Prompt "
        };
        let block = Block::new().borders(Borders::TOP).title(title);
        let content = block.inner(input);
        self.input.scroll = self
            .input
            .scroll
            .min(layout.rows.len().saturating_sub(content.height as usize));
        if layout.cursor_row < self.input.scroll {
            self.input.scroll = layout.cursor_row;
        } else if layout.cursor_row >= self.input.scroll + content.height as usize {
            self.input.scroll = (layout.cursor_row + 1).saturating_sub(content.height as usize);
        }
        let first = self.input.scroll;
        let last = (first + content.height as usize).min(layout.rows.len());
        let selection = self.input.selection();
        let selected_style = Style::default().add_modifier(Modifier::REVERSED);
        let visible = Text::from(
            layout.rows[first..last]
                .iter()
                .map(|range| {
                    let Some(selection) = &selection else {
                        return Line::from(&self.input.text[range.clone()]);
                    };
                    let start = selection.start.clamp(range.start, range.end);
                    let end = selection.end.clamp(range.start, range.end);
                    let mut spans = vec![
                        Span::raw(&self.input.text[range.start..start]),
                        Span::styled(&self.input.text[start..end], selected_style),
                        Span::raw(&self.input.text[end..range.end]),
                    ];
                    if self.input.text.as_bytes().get(range.end) == Some(&b'\n')
                        && selection.contains(&range.end)
                    {
                        spans.push(Span::styled(" ", selected_style));
                    }
                    Line::from(spans)
                })
                .collect::<Vec<_>>(),
        );
        frame.render_widget(Paragraph::new(visible).block(block), input);
        if !self.browser && content.height > 0 && content.width > 0 {
            frame.set_cursor_position((
                content.x + (layout.cursor_column as u16).min(content.width - 1),
                content.y + (layout.cursor_row - first) as u16,
            ));
        }
    }

    fn events(
        &mut self,
        updates: &Receiver<Update>,
        terminal: &mut DefaultTerminal,
    ) -> io::Result<()> {
        loop {
            loop {
                match updates.try_recv() {
                    Ok(update) => self.update(update),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(io::Error::other("model worker exited"))
                    }
                }
            }
            if self.quitting && !self.busy {
                break;
            }
            terminal.draw(|frame| self.render(frame))?;
            if !event::poll(Duration::from_millis(50))? {
                continue;
            }
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => {
                    if !self.browser && key.modifiers.contains(KeyModifiers::CONTROL) {
                        match key.code {
                            KeyCode::Char('c') if self.input.selection().is_some() => {
                                self.copy_selection(false);
                                continue;
                            }
                            KeyCode::Char('x') => {
                                self.copy_selection(true);
                                continue;
                            }
                            KeyCode::Char('v') => {
                                self.paste_clipboard();
                                continue;
                            }
                            _ => {}
                        }
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
                    {
                        self.quit();
                        continue;
                    }
                    if key.code == KeyCode::Esc {
                        if self.busy {
                            self.cancel.store(true, Ordering::Relaxed);
                            self.status = "Stopping after the current step…".into();
                        } else if self.browser {
                            self.browser = false;
                        } else if self.input.selection().is_some() {
                            self.input.selection_anchor = None;
                        } else {
                            self.input.clear();
                        }
                        continue;
                    }
                    if self.browser {
                        let index = self.cursor.selected().unwrap_or(self.selected);
                        match key.code {
                            KeyCode::Up => self.cursor.select(Some(
                                (index + self.entries.len() - 1) % self.entries.len(),
                            )),
                            KeyCode::Down => {
                                self.cursor.select(Some((index + 1) % self.entries.len()))
                            }
                            KeyCode::Enter if !self.busy => {
                                if self.entries[index].status(&self.cache) == Status::Downloaded {
                                    self.selected = index;
                                    self.browser = false;
                                    self.status =
                                        format!("Selected {}", self.entries[index].manifest.name);
                                    self.warmup();
                                } else {
                                    self.download(index);
                                }
                            }
                            KeyCode::Char('d') => self.download(index),
                            _ => {}
                        }
                    } else {
                        match key.code {
                            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                self.input.insert("\n");
                            }
                            KeyCode::Enter => self.submit(),
                            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                self.input.insert("\n");
                            }
                            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                self.input.select_all();
                            }
                            KeyCode::Backspace => {
                                self.input.backspace();
                            }
                            KeyCode::Delete => self.input.delete(),
                            KeyCode::Left | KeyCode::Right => self.input.horizontal(
                                key.code == KeyCode::Right,
                                key.modifiers.contains(KeyModifiers::SHIFT),
                                key.modifiers.contains(KeyModifiers::CONTROL),
                            ),
                            KeyCode::Up | KeyCode::Down | KeyCode::Home | KeyCode::End => {
                                let columns =
                                    terminal.size()?.width.saturating_sub(1).max(1) as usize;
                                match key.code {
                                    KeyCode::Up | KeyCode::Down => self.input.vertical(
                                        key.code == KeyCode::Down,
                                        columns,
                                        key.modifiers.contains(KeyModifiers::SHIFT),
                                    ),
                                    KeyCode::Home | KeyCode::End => self.input.line_edge(
                                        key.code == KeyCode::End,
                                        key.modifiers.contains(KeyModifiers::CONTROL),
                                        columns,
                                        key.modifiers.contains(KeyModifiers::SHIFT),
                                    ),
                                    _ => unreachable!(),
                                }
                            }
                            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                self.input.clear()
                            }
                            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                                self.input.insert(&c.to_string())
                            }
                            KeyCode::PageUp => self.scroll = self.scroll.saturating_add(10),
                            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
                            _ => {}
                        }
                    }
                }
                Event::Paste(text) if !self.browser => self
                    .input
                    .insert(&text.replace("\r\n", "\n").replace('\r', "\n")),
                Event::Resize(_, _) => self.input.reset_column(),
                _ => {}
            }
        }
        Ok(())
    }
}

fn clipboard_command(copy: bool) -> io::Result<Command> {
    let (program, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        (if copy { "pbcopy" } else { "pbpaste" }, &[])
    } else if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        if copy {
            ("wl-copy", &["--type", "text/plain;charset=utf-8"])
        } else {
            ("wl-paste", &["--no-newline", "--type", "text"])
        }
    } else if std::env::var_os("DISPLAY").is_some() {
        if copy {
            ("xclip", &["-selection", "clipboard", "-in"])
        } else {
            ("xclip", &["-selection", "clipboard", "-out"])
        }
    } else {
        return Err(io::Error::other(
            "no desktop clipboard; terminal paste with Ctrl+Shift+V is still available",
        ));
    };
    let mut command = Command::new(program);
    command.args(args);
    Ok(command)
}

fn clipboard_write(text: &str) -> io::Result<()> {
    let mut command = clipboard_command(true)?;
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| {
            io::Error::other(format!(
                "{}: {error}",
                command.get_program().to_string_lossy()
            ))
        })?;
    let written = child.stdin.take().unwrap().write_all(text.as_bytes());
    let status = child.wait()?;
    written?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("could not write to the desktop clipboard"))
    }
}

fn clipboard_read() -> io::Result<String> {
    let mut command = clipboard_command(false)?;
    let output = command.stdin(Stdio::null()).output().map_err(|error| {
        io::Error::other(format!(
            "{}: {error}",
            command.get_program().to_string_lossy()
        ))
    })?;
    if !output.status.success() {
        return Err(io::Error::other(
            "could not read text from the desktop clipboard",
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[derive(Default)]
struct Composer {
    text: String,
    // Byte offset, always at a Unicode grapheme boundary.
    cursor: usize,
    preferred_column: Option<usize>,
    // A wrap boundary belongs to either adjacent row. Vertical movement and
    // End keep the caret on the row the user selected.
    row_hint: Option<usize>,
    scroll: usize,
    selection_anchor: Option<usize>,
}

struct ComposerLayout {
    rows: Vec<std::ops::Range<usize>>,
    cursor_row: usize,
    cursor_column: usize,
}

impl Composer {
    fn clear(&mut self) {
        *self = Self::default();
    }
    fn select_all(&mut self) {
        self.selection_anchor = (!self.text.is_empty()).then_some(0);
        self.cursor = self.text.len();
        self.reset_column();
    }
    fn selection(&self) -> Option<std::ops::Range<usize>> {
        self.selection_anchor
            .filter(|&anchor| anchor != self.cursor)
            .map(|anchor| anchor.min(self.cursor)..anchor.max(self.cursor))
    }
    fn prepare_selection(&mut self, selecting: bool) {
        if selecting {
            self.selection_anchor.get_or_insert(self.cursor);
        } else {
            self.selection_anchor = None;
        }
    }
    fn delete_selection(&mut self) -> bool {
        let selection = self.selection();
        self.selection_anchor = None;
        if let Some(range) = selection {
            self.cursor = range.start;
            self.text.replace_range(range, "");
            self.reset_column();
            return true;
        }
        false
    }
    fn reset_column(&mut self) {
        self.preferred_column = None;
        self.row_hint = None;
    }
    fn normalize_cursor(&mut self) {
        self.cursor = self
            .text
            .grapheme_indices(true)
            .map(|(offset, _)| offset)
            .find(|&offset| offset >= self.cursor)
            .unwrap_or(self.text.len());
        self.reset_column();
    }
    fn insert(&mut self, text: &str) {
        self.delete_selection();
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.normalize_cursor();
    }
    fn previous(&self) -> usize {
        self.text
            .grapheme_indices(true)
            .map(|(offset, _)| offset)
            .take_while(|&offset| offset < self.cursor)
            .last()
            .unwrap_or(0)
    }
    fn next(&self) -> usize {
        self.text
            .grapheme_indices(true)
            .map(|(offset, _)| offset)
            .find(|&offset| offset > self.cursor)
            .unwrap_or(self.text.len())
    }
    fn horizontal(&mut self, right: bool, selecting: bool, by_word: bool) {
        if !selecting {
            if let Some(range) = self.selection() {
                self.cursor = if right { range.end } else { range.start };
                self.selection_anchor = None;
                self.reset_column();
                return;
            }
        }
        self.prepare_selection(selecting);
        self.cursor = if by_word {
            if right {
                self.text
                    .unicode_word_indices()
                    .map(|(offset, word)| offset + word.len())
                    .find(|&end| end > self.cursor)
                    .unwrap_or(self.text.len())
            } else {
                self.text
                    .unicode_word_indices()
                    .flat_map(|(offset, word)| [offset, offset + word.len()])
                    .take_while(|&boundary| boundary < self.cursor)
                    .last()
                    .unwrap_or(0)
            }
        } else if right {
            self.next()
        } else {
            self.previous()
        };
        self.reset_column();
    }
    fn backspace(&mut self) {
        if self.delete_selection() {
            self.normalize_cursor();
            return;
        }
        let previous = self.previous();
        self.text.replace_range(previous..self.cursor, "");
        self.cursor = previous;
        self.normalize_cursor();
    }
    fn delete(&mut self) {
        if self.delete_selection() {
            self.normalize_cursor();
            return;
        }
        self.text.replace_range(self.cursor..self.next(), "");
        self.normalize_cursor();
    }
    fn layout(&self, columns: usize) -> ComposerLayout {
        let mut rows = vec![];
        let (mut start, mut width) = (0, 0);
        for (offset, grapheme) in self.text.grapheme_indices(true) {
            if grapheme == "\n" {
                rows.push(start..offset);
                start = offset + 1;
                width = 0;
                continue;
            }
            let next = Line::from(grapheme).width();
            if width > 0 && width + next > columns {
                rows.push(start..offset);
                start = offset;
                width = 0;
            }
            width += next;
        }
        rows.push(start..self.text.len());
        let cursor_row = self
            .row_hint
            .filter(|&row| {
                rows.get(row)
                    .is_some_and(|r| r.start <= self.cursor && self.cursor <= r.end)
            })
            .unwrap_or_else(|| {
                rows.iter()
                    .rposition(|r| r.start <= self.cursor)
                    .unwrap_or(0)
            });
        let cursor_column = Line::from(&self.text[rows[cursor_row].start..self.cursor]).width();
        ComposerLayout {
            rows,
            cursor_row,
            cursor_column,
        }
    }
    fn vertical(&mut self, down: bool, columns: usize, selecting: bool) {
        self.prepare_selection(selecting);
        let layout = self.layout(columns);
        let goal = *self.preferred_column.get_or_insert(layout.cursor_column);
        let row = if down {
            (layout.cursor_row + 1).min(layout.rows.len() - 1)
        } else {
            layout.cursor_row.saturating_sub(1)
        };
        let range = &layout.rows[row];
        let mut width = 0;
        self.cursor = range.start;
        for (offset, grapheme) in self.text[range.clone()].grapheme_indices(true) {
            let next = Line::from(grapheme).width();
            if width + next > goal {
                break;
            }
            width += next;
            self.cursor = range.start + offset + grapheme.len();
        }
        self.row_hint = Some(row);
    }
    fn line_edge(&mut self, end: bool, whole_text: bool, columns: usize, selecting: bool) {
        self.prepare_selection(selecting);
        let layout = self.layout(columns);
        self.reset_column();
        if whole_text {
            self.cursor = if end { self.text.len() } else { 0 };
        } else {
            let range = &layout.rows[layout.cursor_row];
            self.cursor = if end { range.end } else { range.start };
            self.row_hint = Some(layout.cursor_row);
        }
    }
}

struct TokenWriter<'a> {
    updates: &'a Sender<Update>,
    cancel: &'a AtomicBool,
    started: Instant,
    first: Option<Duration>,
}
impl Write for TokenWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Generation stopped",
            ));
        }
        if !bytes.is_empty() {
            self.first.get_or_insert_with(|| self.started.elapsed());
            let text = std::str::from_utf8(bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            self.updates
                .send(Update::Chunk(text.into()))
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "UI closed"))?;
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Generation stopped",
            ))
        } else {
            Ok(())
        }
    }
}

fn worker(
    entries: Vec<Entry>,
    cache: PathBuf,
    requests: Receiver<Request>,
    updates: Sender<Update>,
    cancel: Arc<AtomicBool>,
    max_memory: Option<super::memory_limit::MemoryLimit>,
) {
    // ABI handles are deliberately created and freed on this one owner thread.
    let mut loaded: Option<(usize, String, Model)> = None;
    let mut tokenizer: Option<(usize, tokenizers::Tokenizer)> = None;
    for request in requests {
        let shutdown = matches!(&request, Request::Shutdown);
        let warming = matches!(&request, Request::Warmup { .. });
        let warmup_started = Instant::now();
        let result: Result<()> = (|| {
            match request {
                Request::Shutdown => return Ok(()),
                Request::Download(index) => {
                    entries[index].download(&cache, &mut |p| {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(io::Error::new(
                                io::ErrorKind::Interrupted,
                                "Download stopped",
                            ));
                        }
                        updates
                            .send(Update::Progress {
                                file: p.filename.into(),
                                bytes: p.bytes,
                                total: p.total,
                            })
                            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "UI closed"))
                    })?;
                    let _ = updates.send(Update::Downloaded);
                }
                request @ (Request::Generate { .. } | Request::Warmup { .. }) => {
                    let (index, device, prompt, temperature, limit) = match request {
                        Request::Generate {
                            index,
                            device,
                            prompt,
                            temperature,
                            limit,
                        } => (index, device, prompt, temperature, limit),
                        Request::Warmup {
                            index,
                            device,
                            limit,
                        } => (index, device, String::new(), 0., limit),
                        _ => unreachable!(),
                    };
                    if cancel.load(Ordering::Relaxed) {
                        return Err("Operation stopped".into());
                    }
                    let started = Instant::now();
                    let entry = &entries[index];
                    let dir = entry.model_dir(&cache);
                    if tokenizer
                        .as_ref()
                        .is_none_or(|(old_index, _)| *old_index != index)
                    {
                        let _ = updates.send(Update::Status("Loading tokenizer…".into()));
                        tokenizer = Some((
                            index,
                            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                                .map_err(|e| e.to_string())?,
                        ));
                    }
                    let tokenizer = &tokenizer.as_ref().unwrap().1;
                    let input = if warming {
                        vec![0]
                    } else {
                        llm::tokenize_with(tokenizer, &dir, &prompt)?
                    };
                    // Only the prompt must fit initially. Output grows the retained
                    // allocation on demand, independently of the optional response cap.
                    let required = input.len();
                    let capacity = required
                        .max(512)
                        .checked_next_power_of_two()
                        .ok_or("context capacity overflow")?;
                    if loaded
                        .as_ref()
                        .is_none_or(|(old_index, old_device, model)| {
                            *old_index != index
                                || *old_device != device
                                || required as u64 > model.info.context_length
                        })
                    {
                        // Release the old weights before loading the next model.
                        loaded = None;
                        let _ = updates.send(Update::Status(format!(
                            "Loading {} and planning context on {device}…",
                            entry.manifest.name
                        )));
                        let program = entry.materialize_program(&cache)?;
                        let model = llm::load_model_with_policy(
                            &program,
                            &dir,
                            &device,
                            None,
                            false,
                            crate::compiler::cpu::CpuTarget::Generic,
                            llm::LoadPolicy {
                                grow_context: true,
                                cache_dir: Some(&cache.join("compiled")),
                                context_request: Some(super::llm_capacity::ContextRequest {
                                    capacity,
                                    prompt_tokens: input.len(),
                                    minimum_capacity: Some(required),
                                }),
                                max_memory: max_memory.map(|m| m.0),
                            },
                        )?;
                        llm::validate_tokenizer(&model, tokenizer)?;
                        loaded = Some((index, device.clone(), model));
                    }
                    if cancel.load(Ordering::Relaxed) {
                        return Err("Generation stopped".into());
                    }
                    let model = &mut loaded.as_mut().unwrap().2;
                    if warming {
                        let _ = updates.send(Update::Status(
                            "Warming decode and prefill kernels… You can type a prompt.".into(),
                        ));
                        let generation = Generation {
                            max_new_tokens: 1,
                            temperature: 0.,
                            seed: 0,
                            reserved: 0,
                        };
                        let mut on_tokens = |_: &[u32]| -> super::llm_ffi::Result<()> {
                            if cancel.load(Ordering::Relaxed) {
                                Err("Warmup stopped".into())
                            } else {
                                Ok(())
                            }
                        };
                        model
                            .infer(&[0], generation, Some(&mut on_tokens))
                            .map_err(io::Error::other)?;
                        if cancel.load(Ordering::Relaxed) {
                            return Err("Warmup stopped".into());
                        }
                        let prefill = vec![0; (model.info.context_length as usize).min(8)];
                        if prefill.len() > 1 {
                            // Prefill warming is optional: a larger dummy shape
                            // may not fit even though decode and a real short prompt do.
                            if let Err(error) =
                                model.infer(&prefill, generation, Some(&mut on_tokens))
                            {
                                if cancel.load(Ordering::Relaxed) {
                                    return Err("Warmup stopped".into());
                                }
                                eprintln!("optional prefill warmup skipped: {error}");
                            }
                        }
                        if cancel.load(Ordering::Relaxed) {
                            return Err("Warmup stopped".into());
                        }
                        let _ = updates.send(Update::Warmed {
                            error: None,
                            elapsed: warmup_started.elapsed(),
                        });
                        return Ok(());
                    }
                    let _ = updates.send(Update::Status(format!(
                        "Processing {} input tokens; compiling kernels if needed…",
                        input.len()
                    )));
                    let remaining = model
                        .info
                        .context_length
                        .checked_sub(input.len() as u64)
                        .ok_or("prompt exceeds model context length")?
                        + 1;
                    let context_bound = limit.is_none_or(|n| n >= remaining);
                    let generation = Generation {
                        max_new_tokens: limit.unwrap_or(remaining).min(remaining),
                        temperature,
                        seed: 299_792_458,
                        reserved: 0,
                    };
                    let mut writer = TokenWriter {
                        updates: &updates,
                        cancel: &cancel,
                        started,
                        first: None,
                    };
                    let output = llm::generate_output_displayed(
                        model,
                        tokenizer,
                        &input,
                        generation,
                        true,
                        false,
                        &mut writer,
                    )
                    .map_err(io::Error::other)?;
                    let _ = updates.send(Update::Done {
                        tokens: output.tokens.len(),
                        reason: if context_bound
                            && output.reason == crate::runtime::llm_ffi::DONE_LIMIT
                        {
                            crate::runtime::llm_ffi::DONE_CONTEXT
                        } else {
                            output.reason
                        },
                        elapsed: started.elapsed(),
                        first_token: writer.first,
                    });
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            if warming {
                loaded = None;
                let _ = updates.send(Update::Warmed {
                    error: Some(error.to_string()),
                    elapsed: warmup_started.elapsed(),
                });
            } else {
                let _ = updates.send(Update::Error(error.to_string()));
            }
        }
        if shutdown {
            break;
        }
    }
}

/// Keep existing compiler diagnostics from overwriting the alternate-screen UI.
#[cfg(unix)]
struct Diagnostics(std::os::fd::OwnedFd);
#[cfg(unix)]
impl Diagnostics {
    fn capture(path: &std::path::Path) -> io::Result<Self> {
        use std::os::fd::{AsRawFd, FromRawFd};
        let log = fs::File::create(path)?;
        io::stderr().flush()?;
        let saved = unsafe { libc::dup(libc::STDERR_FILENO) };
        if saved < 0 {
            return Err(io::Error::last_os_error());
        }
        let saved = unsafe { std::os::fd::OwnedFd::from_raw_fd(saved) };
        if unsafe { libc::dup2(log.as_raw_fd(), libc::STDERR_FILENO) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(saved))
    }
}
#[cfg(unix)]
impl Drop for Diagnostics {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        let _ = io::stderr().flush();
        unsafe {
            libc::dup2(self.0.as_raw_fd(), libc::STDERR_FILENO);
        }
    }
}

pub fn run(options: Options) -> Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err("The TUI needs an interactive terminal; use puppygrad llm for scripts.".into());
    }
    let entries = catalog::load(options.catalog.as_deref())?;
    let cache = options.cache_dir.unwrap_or_else(catalog::default_cache_dir);
    fs::create_dir_all(&cache)?;
    #[cfg(unix)]
    let _diagnostics = Diagnostics::capture(&cache.join("tui.log"))?;
    let device = options.device.unwrap_or_else(|| {
        for (name, backend) in [
            ("cuda:0", crate::compiler::gpu::Backend::Cuda),
            ("hip:0", crate::compiler::gpu::Backend::Hip),
        ] {
            if crate::compiler::gpu::Runtime::new(backend, 0).is_ok() {
                return name.into();
            }
        }
        "cpu".into()
    });
    let (requests, received) = mpsc::channel();
    let (send_updates, updates) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_entries = entries.clone();
    let worker_cache = cache.clone();
    let worker_cancel = cancel.clone();
    let handle = thread::spawn(move || {
        worker(
            worker_entries,
            worker_cache,
            received,
            send_updates,
            worker_cancel,
            options.max_memory,
        )
    });
    let mut app = App::new(entries, cache, device, requests.clone(), cancel.clone());
    app.auto_warmup = !options.no_warmup;
    let mut enhanced_keyboard = false;
    let outcome = (|| {
        let mut terminal = ratatui::try_init()?;
        execute!(io::stdout(), event::EnableBracketedPaste)?;
        execute!(
            io::stdout(),
            event::PushKeyboardEnhancementFlags(
                event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )?;
        enhanced_keyboard = true;
        app.warmup();
        app.events(&updates, &mut terminal)
    })();
    if enhanced_keyboard {
        let _ = execute!(io::stdout(), event::PopKeyboardEnhancementFlags);
    }
    let _ = execute!(io::stdout(), event::DisableBracketedPaste);
    ratatui::restore();
    cancel.store(true, Ordering::Relaxed);
    let _ = requests.send(Request::Shutdown);
    // An ordinary quit waits for cancellation before restoring the terminal.
    // On terminal I/O failure, let process exit stop a still-running worker.
    if outcome.is_ok() {
        handle.join().map_err(|_| "model worker panicked")?;
    }
    outcome?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    struct WarmupFixture {
        root: PathBuf,
        cache: PathBuf,
        entries: Vec<Entry>,
    }
    impl WarmupFixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = format!(
                "tui-warm-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let root = std::env::temp_dir().join(&id);
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join("tiny.pup"), "output weight(\"scores\")\n").unwrap();
            fs::write(root.join("tiny.model.json"), serde_json::to_vec(&serde_json::json!({
                "id":id, "name":"Tiny", "description":"Tiny warmup test", "program":"tiny.pup",
                "checkpoint":{"repo":"test/tiny","revision":"test","files":["config.json","tokenizer.json","model.safetensors"]}
            })).unwrap()).unwrap();
            let entries = catalog::load(Some(&root)).unwrap();
            let cache = root.join("cache");
            let dir = entries[0].model_dir(&cache);
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("config.json"),
                r#"{"vocab_size":6,"n_positions":1024,"eos_token_id":5}"#,
            )
            .unwrap();
            fs::write(dir.join("tokenizer.json"), r###"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":{"type":"WordPiece","prefix":"##","cleanup":false},"model":{"type":"WordLevel","vocab":{"prompt":0,"hello":1,"world":2,"foo":3,"bar":4,"[UNK]":5},"unk_token":"[UNK]"}}"###).unwrap();
            let bytes = [-100.0f32, 1., 0., 0., 0., -100.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![6], &bytes)
                    .unwrap();
            fs::write(
                dir.join("model.safetensors"),
                safetensors::tensor::serialize([("scores", view)], None).unwrap(),
            )
            .unwrap();
            Self {
                root,
                cache,
                entries,
            }
        }
    }
    impl Drop for WarmupFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn warmup_allows_queued_prompt_and_respects_opt_out_and_missing_assets() {
        let f = WarmupFixture::new();
        let (tx, rx) = mpsc::channel();
        let mut app = App::new(
            f.entries.clone(),
            f.cache.clone(),
            "cpu".into(),
            tx,
            Arc::new(AtomicBool::new(false)),
        );
        app.auto_warmup = false;
        app.warmup();
        assert!(rx.try_recv().is_err());
        app.auto_warmup = true;
        app.warmup();
        assert!(app.busy && app.warming);
        assert!(matches!(
            rx.try_recv().unwrap(),
            Request::Warmup { index: 0, .. }
        ));
        app.input.insert("prompt");
        app.submit();
        assert!(
            matches!(rx.try_recv().unwrap(), Request::Generate { prompt, .. } if prompt == "prompt")
        );
        let transcript = app.transcript.clone();
        app.update(Update::Warmed {
            error: None,
            elapsed: Duration::from_millis(10),
        });
        assert!(app.busy && !app.warming);
        assert_eq!(app.transcript, transcript);
        app.update(Update::Done {
            tokens: 128,
            reason: crate::runtime::llm_ffi::DONE_LIMIT,
            elapsed: Duration::from_secs(1),
            first_token: None,
        });
        assert!(!app.busy && app.status.starts_with("Token limit reached"));
        fs::remove_file(f.entries[0].model_dir(&f.cache).join("model.safetensors")).unwrap();
        app.warmup();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn cancelled_warmup_keeps_unsent_prompt_and_emits_no_conversation_output() {
        let f = WarmupFixture::new();
        let (requests, rx) = mpsc::channel();
        let (updates, received) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut app = App::new(
            f.entries.clone(),
            f.cache.clone(),
            "cpu".into(),
            requests,
            cancel.clone(),
        );
        app.warmup();
        cancel.store(true, Ordering::Relaxed);
        let transcript = app.transcript.clone();
        app.input.insert("unsent");
        app.submit();
        assert_eq!(app.input.text, "unsent");
        app.requests.send(Request::Shutdown).unwrap();
        worker(
            f.entries.clone(),
            f.cache.clone(),
            rx,
            updates,
            cancel,
            None,
        );
        let update = received.try_recv().unwrap();
        assert!(matches!(&update, Update::Warmed { error: Some(_), .. }));
        app.update(update);
        assert!(!app.busy && !app.warming);
        assert_eq!(app.transcript, transcript);
        assert!(!f.cache.join("compiled").exists());
        assert!(received.try_recv().is_err());
    }

    #[test]
    fn worker_warms_before_prompt_without_output_and_reuses_loaded_weights() {
        let mut f = WarmupFixture::new();
        // Decode fits this retained buffer, but the optional eight-token dummy
        // prefill fails its bounds check. Keep the successful warm provider.
        fs::write(f.root.join("tiny.pup"), "tokens = input(\"tokens\")\ncache = state(\"cache\", i32, [2])\nwrite = store(index(cache, arange(dim(tokens, 0))), tokens)\noutput after(weight(\"scores\"), write)\n").unwrap();
        f.entries = catalog::load(Some(&f.root)).unwrap();

        let (requests, rx) = mpsc::channel();
        let (updates, received) = mpsc::channel();
        let entries = f.entries.clone();
        let cache = f.cache.clone();
        let handle = thread::spawn(move || {
            worker(
                entries,
                cache,
                rx,
                updates,
                Arc::new(AtomicBool::new(false)),
                None,
            )
        });
        requests
            .send(Request::Warmup {
                index: 0,
                device: "cpu".into(),
                limit: None,
            })
            .unwrap();
        loop {
            match received.recv_timeout(Duration::from_secs(15)).unwrap() {
                Update::Status(_) => {}
                Update::Warmed { error, .. } => {
                    assert!(error.is_none(), "{error:?}");
                    break;
                }
                _ => panic!("warmup emitted conversation output"),
            }
        }
        assert!(fs::read_dir(f.cache.join("compiled/cpu"))
            .unwrap()
            .any(|p| p.unwrap().path().extension().is_some_and(|e| e == "so")));
        // Inference must use the warm provider rather than reopening the checkpoint.
        fs::remove_file(f.entries[0].model_dir(&f.cache).join("model.safetensors")).unwrap();
        requests
            .send(Request::Generate {
                index: 0,
                device: "cpu".into(),
                prompt: "prompt".into(),
                temperature: 0.,
                limit: Some(1),
            })
            .unwrap();
        requests.send(Request::Shutdown).unwrap();
        let mut text = String::new();
        let mut done = false;
        while let Ok(update) = received.recv_timeout(Duration::from_secs(15)) {
            match update {
                Update::Chunk(chunk) => text.push_str(&chunk),
                Update::Done { tokens, reason, .. } => {
                    assert_eq!(tokens, 1);
                    assert_eq!(reason, crate::runtime::llm_ffi::DONE_LIMIT);
                    done = true;
                }
                Update::Error(error) => panic!("{error}"),
                _ => {}
            }
        }
        handle.join().unwrap();
        assert!(done);
        assert_eq!(text, "hello");
    }

    #[test]
    fn automatic_output_runs_past_128_tokens_and_reports_context_exhaustion() {
        let mut f = WarmupFixture::new();
        // A tiny retained program keeps one position counter and constant scores.
        // Initial context is 512; a 600-position model must grow during this response.
        fs::write(f.root.join("tiny.pup"), "tokens = input(\"tokens\")\nposition = state(\"position\", i32, [1])\nadvance = store(index(position, cast(0, i32)), load(index(position, cast(0, i32))) + dim(tokens, 0))\noutput after(weight(\"scores\"), advance)\n").unwrap();
        f.entries = catalog::load(Some(&f.root)).unwrap();
        fs::write(
            f.entries[0].model_dir(&f.cache).join("config.json"),
            r#"{"vocab_size":6,"n_positions":600,"eos_token_id":5}"#,
        )
        .unwrap();
        let (requests, rx) = mpsc::channel();
        let (updates, received) = mpsc::channel();
        for limit in [None, Some(1_000_000)] {
            requests
                .send(Request::Generate {
                    index: 0,
                    device: "cpu".into(),
                    prompt: "prompt".into(),
                    temperature: 0.,
                    limit,
                })
                .unwrap();
        }
        requests.send(Request::Shutdown).unwrap();
        worker(
            f.entries.clone(),
            f.cache.clone(),
            rx,
            updates,
            Arc::new(AtomicBool::new(false)),
            None,
        );
        let mut done = 0;
        for update in received.try_iter() {
            match update {
                Update::Done { tokens, reason, .. } => {
                    assert_eq!(tokens, 600);
                    assert_eq!(reason, crate::runtime::llm_ffi::DONE_CONTEXT);
                    done += 1;
                }
                Update::Error(error) => panic!("{error}"),
                _ => {}
            }
        }
        assert_eq!(done, 2);
        let (tx, _) = mpsc::channel();
        let mut app = App::new(
            f.entries.clone(),
            f.cache.clone(),
            "cpu".into(),
            tx,
            Arc::new(AtomicBool::new(false)),
        );
        assert_eq!(app.limit, None);
        app.command("/tokens 256");
        assert_eq!(app.limit, Some(256));
        app.command("/tokens auto");
        assert_eq!(app.limit, None);
        let mut terminal = Terminal::new(TestBackend::new(100, 20)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let screen = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(screen.contains("output auto"));
        for (reason, label) in [
            (crate::runtime::llm_ffi::DONE_MEMORY, "Memory limit reached"),
            (
                crate::runtime::llm_ffi::DONE_CONTEXT,
                "Context limit reached",
            ),
            (crate::runtime::llm_ffi::DONE_EOS, "Complete"),
        ] {
            app.update(Update::Done {
                tokens: 10,
                reason,
                elapsed: Duration::from_secs(1),
                first_token: None,
            });
            assert!(app.status.starts_with(label));
        }
    }

    #[test]
    fn composer_selection_reverses_and_replaces_whole_unicode_graphemes() {
        let mut input = Composer::default();
        input.insert("e\u{301}👩‍💻xyz");
        input.line_edge(false, true, 20, false);
        input.horizontal(true, true, false);
        input.horizontal(true, true, false);
        input.horizontal(false, true, false);
        assert_eq!(&input.text[input.selection().unwrap()], "e\u{301}");
        input.horizontal(false, true, false);
        assert!(input.selection().is_none());
        input.horizontal(true, true, false);
        input.horizontal(true, true, false);
        input.insert("Q");
        assert_eq!(input.text, "Qxyz");
        input.horizontal(true, true, false);
        input.horizontal(true, true, false);
        input.insert("A\nB");
        assert_eq!(input.text, "QA\nBz");
        input.select_all();
        input.delete();
        assert!(input.text.is_empty());
    }

    #[test]
    fn composer_selects_across_short_and_wrapped_lines_and_moves_by_word() {
        let mut input = Composer::default();
        input.insert("abcd\nx\nabcd");
        input.vertical(false, 20, true);
        input.vertical(false, 20, true);
        input.insert("Z");
        assert_eq!(input.text, "abcdZ");
        input.clear();
        input.insert("abcdefg");
        input.vertical(false, 3, true);
        input.backspace();
        assert_eq!(input.text, "abcd");
        input.clear();
        input.insert("one,  café! three");
        input.horizontal(false, false, true);
        input.insert("X");
        assert_eq!(input.text, "one,  café! Xthree");
        input.horizontal(false, true, true);
        input.horizontal(false, true, true);
        assert_eq!(&input.text[input.selection().unwrap()], "! X");
        input.horizontal(false, true, true);
        assert_eq!(&input.text[input.selection().unwrap()], "café! X");
        input.insert("two ");
        assert_eq!(input.text, "one,  two three");
        input.line_edge(false, true, 20, false);
        input.horizontal(true, true, true);
        assert_eq!(&input.text[input.selection().unwrap()], "one");
        input.delete();
        assert_eq!(input.text, ",  two three");
    }

    #[test]
    fn composer_keeps_column_across_short_lines_and_edits_at_cursor() {
        let mut input = Composer::default();
        input.insert("abcd\nx\nabcd");
        input.vertical(false, 20, false);
        assert_eq!(input.cursor, 6);
        input.vertical(false, 20, false);
        assert_eq!(input.cursor, 4);
        input.vertical(true, 20, false);
        assert_eq!(input.cursor, 6);
        input.vertical(true, 20, false);
        assert_eq!(input.cursor, input.text.len());
        input.vertical(false, 20, false);
        input.horizontal(false, false, false);
        input.insert("Y");
        assert_eq!(input.text, "abcd\nYx\nabcd");
        input.backspace();
        input.delete();
        assert_eq!(input.text, "abcd\n\nabcd");
        input.backspace();
        assert_eq!(input.text, "abcd\nabcd");
    }

    #[test]
    fn composer_edits_unicode_graphemes_without_splitting_them() {
        let mut input = Composer::default();
        input.insert("e\u{301}你👩‍💻");
        input.backspace();
        assert_eq!(input.text, "e\u{301}你");
        input.horizontal(false, false, false);
        input.delete();
        assert_eq!(input.text, "e\u{301}");
        input.horizontal(false, false, false);
        input.delete();
        assert_eq!(input.text, "");
        assert_eq!(input.cursor, 0);
    }

    #[test]
    fn composer_navigates_wrapped_rows_and_keeps_end_on_selected_row() {
        let mut input = Composer::default();
        input.insert("abcdefg");
        input.vertical(false, 3, false);
        input.insert("X");
        assert_eq!(input.text, "abcdXefg");
        input.line_edge(false, true, 3, false);
        input.line_edge(true, false, 3, false);
        assert_eq!(input.cursor, 3);
        assert_eq!(input.layout(3).cursor_row, 0);
        input.vertical(true, 3, false);
        assert_eq!(input.cursor, 6);
        assert_eq!(input.layout(3).cursor_row, 1);
        input.vertical(true, 3, false);
        assert_eq!(input.cursor, 8);
        input.vertical(false, 3, false);
        assert_eq!(input.cursor, 6);
        input.line_edge(false, true, 3, false);
        assert_eq!(input.cursor, 0);
        input.line_edge(true, true, 3, false);
        assert_eq!(input.cursor, input.text.len());
    }

    #[test]
    fn model_browser_renders_download_states_and_queues_download() {
        let mut entries = catalog::load(None).unwrap();
        let cache = std::env::temp_dir().join(format!("pup-tui-ui-{}", std::process::id()));
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.manifest.id = format!("tui-test-{}-{i}", std::process::id());
        }
        let dir = entries[0].model_dir(&cache);
        fs::create_dir_all(&dir).unwrap();
        for name in &entries[0].manifest.checkpoint.files {
            fs::write(dir.join(name), "data").unwrap();
        }
        let (tx, rx) = mpsc::channel();
        let mut app = App::new(
            entries,
            cache.clone(),
            "cpu".into(),
            tx,
            Arc::new(AtomicBool::new(false)),
        );
        app.command("/model");
        assert!(app.browser);
        let mut terminal = Terminal::new(TestBackend::new(100, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let screen: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(screen.contains("GPT-2 small"));
        assert!(screen.contains("Qwen3-0.6B"));
        assert!(screen.contains("downloaded"));
        assert!(screen.contains("not downloaded"));
        app.download(1);
        assert!(matches!(rx.try_recv().unwrap(), Request::Download(1)));
        assert!(app.busy);
        fs::remove_dir_all(cache).unwrap();
    }

    #[test]
    fn invalid_controls_do_not_change_settings_and_unicode_input_is_preserved() {
        let (tx, _) = mpsc::channel();
        let mut app = App::new(
            catalog::load(None).unwrap(),
            PathBuf::from("/tmp/unused-tui-cache"),
            "cpu".into(),
            tx,
            Arc::new(AtomicBool::new(false)),
        );
        app.command("/temperature NaN");
        assert_eq!(app.temperature, 0.7);
        app.command("/tokens 0");
        assert_eq!(app.limit, None);
        app.command("/device nonsense");
        assert_eq!(app.device, "cpu");
        app.input.insert("你好 🐶");
        let mut terminal = Terminal::new(TestBackend::new(20, 12)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        assert_eq!(app.input.text, "你好 🐶");
    }
}
