//! SQLite chat sessions with a bounded, chronological active window.
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Turn {
    pub user: String,
    pub assistant: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum LegacyRecord {
    Turn(Turn),
    Reset,
}

#[derive(Clone, Debug)]
pub(super) struct Session {
    pub id: String,
    pub title: String,
    pub model: Option<String>,
    pub updated: String,
    pub turns: usize,
}

pub(super) struct Conversation {
    pub path: PathBuf,
    db: Connection,
    pub session_id: String,
    pub title: String,
    pub model: Option<String>,
    count: usize,
    start: usize,
    pub turns: Vec<Turn>,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
fn title(user: &str) -> String {
    user.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect()
}
fn id(db: &Connection) -> rusqlite::Result<String> {
    db.query_row("SELECT lower(hex(randomblob(16)))", [], |row| row.get(0))
}

impl Conversation {
    pub fn open(path: &Path) -> Result<Self> {
        // Keep the old CLI path usable, but never overwrite a JSONL archive.
        let legacy = path.with_extension("jsonl");
        let path = if path.extension().is_some_and(|e| e == "jsonl") {
            path.with_extension("sqlite3")
        } else {
            path.to_owned()
        };
        let db = crate::database::open(&path)?;
        let mut chat = Self {
            session_id: id(&db)?,
            title: "New chat".into(),
            model: None,
            path,
            db,
            count: 0,
            start: 0,
            turns: Vec::new(),
        };
        if legacy.is_file() {
            chat.import_legacy(&legacy)?;
        }
        Ok(chat)
    }

    pub fn import_database(&mut self, path: &Path) -> Result<()> {
        crate::database::import(&mut self.db, path, &self.path)
    }

    /// Atomic, one-time import; every reset starts another resumable session.
    pub fn import_legacy(&mut self, path: &Path) -> Result<()> {
        let source = path.canonicalize()?.to_string_lossy().into_owned();
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if tx
            .query_row(
                "SELECT 1 FROM legacy_imports WHERE path=?1",
                [&source],
                |_| Ok(()),
            )
            .optional()?
            .is_some()
        {
            return Ok(());
        }
        let file = fs::File::open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } != 0 {
                return Err(
                    "Legacy conversation is in use; close the old TUI before importing it".into(),
                );
            }
        }
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut session = id(&tx)?;
        let mut ordinal = 0;
        let timestamp = now();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            if !line.ends_with('\n') {
                return Err("Incomplete legacy conversation record; import rolled back, original file unchanged".into());
            }
            match serde_json::from_str::<LegacyRecord>(&line)? {
                LegacyRecord::Reset => {
                    session = id(&tx)?;
                    ordinal = 0;
                }
                LegacyRecord::Turn(turn) => {
                    if ordinal == 0 {
                        tx.execute("INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?1,?2,?3,?3)", params![session, title(&turn.user), timestamp])?;
                    }
                    tx.execute(
                        "INSERT INTO turns VALUES(?1,?2,?3,?4)",
                        params![session, ordinal, turn.user, turn.assistant],
                    )?;
                    ordinal += 1;
                }
            }
        }
        tx.execute("INSERT INTO legacy_imports VALUES(?1)", [source])?;
        tx.commit()?;
        Ok(())
    }

    pub fn sessions(&self) -> Result<Vec<Session>> {
        let mut query = self.db.prepare(
            "SELECT s.id,s.title,s.model,
            strftime('%Y-%m-%d %H:%M',s.updated_at/1000,'unixepoch','localtime'),
            (SELECT count(*) FROM turns WHERE session_id=s.id)
            FROM sessions s ORDER BY s.updated_at DESC,s.rowid DESC",
        )?;
        let sessions = query
            .query_map([], |r| {
                Ok(Session {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    model: r.get(2)?,
                    updated: r.get(3)?,
                    turns: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(sessions)
    }

    pub fn current(&self) -> Session {
        Session {
            id: self.session_id.clone(),
            title: self.title.clone(),
            model: self.model.clone(),
            updated: String::new(),
            turns: self.count,
        }
    }

    pub fn resume(&mut self, requested: &str) -> Result<Session> {
        // Prefix matching avoids typing a full ID; reject ambiguous prefixes.
        let sessions = self.sessions()?;
        let session = if requested == "latest" {
            sessions.first().cloned().ok_or("No saved sessions yet")?
        } else {
            let mut matches = sessions.into_iter().filter(|s| s.id.starts_with(requested));
            let session = matches
                .next()
                .ok_or("Session not found; use /sessions to browse")?;
            if requested.is_empty() || matches.next().is_some() {
                return Err("Session ID is ambiguous; use a longer ID or /sessions".into());
            }
            session
        };
        let start = session.turns.saturating_sub(8);
        let turns = Self::read_session(&self.db, &session.id, start, session.turns)?;
        self.session_id = session.id.clone();
        self.title = session.title.clone();
        self.model = session.model.clone();
        self.count = session.turns;
        self.start = start;
        self.turns = turns;
        Ok(session)
    }

    fn read_session(db: &Connection, session: &str, start: usize, end: usize) -> Result<Vec<Turn>> {
        let mut query = db.prepare("SELECT user,assistant FROM turns WHERE session_id=?1 AND ordinal>=?2 AND ordinal<?3 ORDER BY ordinal")?;
        let turns = query
            .query_map(params![session, start, end], |r| {
                Ok(Turn {
                    user: r.get(0)?,
                    assistant: r.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if turns.len() != end - start {
            return Err("Session changed while loading; resume it again before continuing".into());
        }
        Ok(turns)
    }
    fn read_turns(&mut self, start: usize, end: usize) -> Result<Vec<Turn>> {
        Self::read_session(&self.db, &self.session_id, start, end)
    }

    pub fn append(&mut self, user: String, assistant: String) -> Result<()> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let count: usize = tx.query_row(
            "SELECT count(*) FROM turns WHERE session_id=?1",
            [&self.session_id],
            |r| r.get(0),
        )?;
        if count != self.count {
            return Err("This session was updated in another TUI; use /resume to reload it before continuing".into());
        }
        let timestamp = now();
        let first_title = if self.count == 0 {
            title(&user)
        } else {
            self.title.clone()
        };
        tx.execute(
            "INSERT INTO sessions(id,title,model,created_at,updated_at) VALUES(?1,?2,?3,?4,?4)
            ON CONFLICT(id) DO UPDATE SET model=excluded.model,updated_at=excluded.updated_at",
            params![self.session_id, first_title, self.model, timestamp],
        )?;
        tx.execute(
            "INSERT INTO turns VALUES(?1,?2,?3,?4)",
            params![self.session_id, self.count, user, assistant],
        )?;
        tx.commit()?;
        self.title = first_title;
        self.count += 1;
        self.turns.push(Turn { user, assistant });
        Ok(())
    }

    /// Allocate a fresh session; persist it only when its first turn is saved.
    pub fn reset(&mut self) -> Result<()> {
        self.session_id = id(&self.db)?;
        self.title = "New chat".into();
        self.model = None;
        self.count = 0;
        self.start = 0;
        self.turns.clear();
        Ok(())
    }

    pub fn older_messages(&self) -> usize {
        self.start * 2
    }
    /// Reserve roughly 20% for a reply; never truncate the current user message.
    pub fn prepare(
        &mut self,
        context: usize,
        encode: impl Fn(&[Turn], usize) -> Result<Vec<u32>>,
    ) -> Result<(Vec<u32>, usize, usize)> {
        let single = encode(&[], self.count * 2)?;
        if single.is_empty() || single.len() > context {
            return Err(
                "The current message exceeds the model context; shorten it and try again".into(),
            );
        }
        let budget = (context - context / 5).max(single.len());
        let mut dropped = 0;
        let mut input = encode(&self.turns, self.older_messages())?;
        while input.len() > budget && dropped < self.turns.len() {
            dropped += 1;
            input = encode(&self.turns[dropped..], (self.start + dropped) * 2)?;
        }
        self.turns.drain(..dropped);
        self.start += dropped;
        Ok((input, dropped, budget))
    }

    pub fn fetch(
        &mut self,
        count: usize,
        budget: usize,
        encode: impl Fn(&[Turn], usize, &str) -> Result<Vec<u32>>,
    ) -> Result<(Vec<u32>, String)> {
        if count == 0 || count > 1024 {
            return Ok((
                Vec::new(),
                "Use a positive message count no greater than 1024.".into(),
            ));
        }
        if self.start == 0 {
            return Ok((
                Vec::new(),
                "No older messages are available in this conversation.".into(),
            ));
        }
        let start = self.start.saturating_sub(count.div_ceil(2));
        let mut turns = self.read_turns(start, self.start)?;
        let added = turns.len() * 2;
        turns.extend_from_slice(&self.turns);
        let feedback = format!("Added {added} older messages before the recent conversation.");
        let input = encode(&turns, start * 2, &feedback)?;
        if input.len() > budget {
            return Ok((Vec::new(), "Those older messages do not fit alongside the recent messages and current question. Request fewer messages or answer with the available context.".into()));
        }
        self.start = start;
        self.turns = turns;
        Ok((input, feedback))
    }

    pub fn undo_fetch(&mut self, previous_turns: usize) {
        let added = self.turns.len().saturating_sub(previous_turns);
        self.turns.drain(..added);
        self.start += added;
    }
}

pub(super) fn fetch_count(text: &str) -> Option<usize> {
    let mut words = text.split_whitespace();
    if words.next()? != "FETCH_OLDER" {
        return None;
    }
    let count = words.next()?;
    if !count.bytes().all(|b| b.is_ascii_digit()) || words.next().is_some() {
        return None;
    }
    Some(count.parse().unwrap_or(usize::MAX))
}

/// Hold only a possible control response; ordinary text streams immediately.
pub(super) fn fetch_prefix(text: &str) -> bool {
    let text = text.trim_start();
    if "FETCH_OLDER".starts_with(text) {
        return true;
    }
    let Some(tail) = text.strip_prefix("FETCH_OLDER") else {
        return false;
    };
    tail.starts_with(char::is_whitespace)
        && tail.trim().bytes().all(|b| b.is_ascii_digit())
        && text.len() <= 40
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "puppygrad-sessions-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            Self(root)
        }
        fn path(&self) -> PathBuf {
            self.0.join("conversation.sqlite3")
        }
        fn chat(&self) -> Conversation {
            Conversation::open(&self.path()).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn encode(turns: &[Turn], _: usize) -> Result<Vec<u32>> {
        Ok(vec![
            0;
            5 + turns
                .iter()
                .map(|t| t.user.len() + t.assistant.len())
                .sum::<usize>()
        ])
    }
    fn encode_fetch(turns: &[Turn], older: usize, _: &str) -> Result<Vec<u32>> {
        encode(turns, older)
    }
    #[test]
    fn sessions_are_isolated_and_resume_recent_turns_then_fetch_their_own_archive() {
        let f = Fixture::new();
        let mut chat = f.chat();
        let first = chat.session_id.clone();
        chat.model = Some("qwen3-0.6b".into());
        for i in 0..10 {
            chat.append(format!("u{i}"), format!("a{i}")).unwrap();
        }
        chat.reset().unwrap();
        chat.append("different chat".into(), "different answer".into())
            .unwrap();
        let second = chat.session_id.clone();
        assert_ne!(first, second);
        drop(chat);
        let mut chat = f.chat();
        assert!(chat.turns.is_empty()); // Fresh startup; no empty row is saved.
        assert_eq!(chat.sessions().unwrap().len(), 2);
        let resumed = chat.resume(&first[..8]).unwrap();
        assert_eq!(resumed.model.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(chat.turns.len(), 8);
        assert_eq!(chat.older_messages(), 4);
        assert_eq!(chat.turns[0].user, "u2");
        chat.fetch(2, 100, encode_fetch).unwrap();
        assert_eq!(chat.turns[0].user, "u1");
        assert!(!chat.turns.iter().any(|t| t.user.contains("different")));
        chat.resume(&second).unwrap();
        assert_eq!(chat.turns[0].user, "different chat");
        assert_eq!(chat.older_messages(), 0);
        let before = chat.session_id.clone();
        assert!(chat.resume("nonexistent").is_err());
        assert_eq!(chat.session_id, before);
    }
    #[test]
    fn compaction_refusal_and_undo_preserve_recent_turns() {
        let f = Fixture::new();
        let mut chat = f.chat();
        for i in 0..5 {
            chat.append(format!("u{i}"), format!("a{i}")).unwrap();
        }
        let (input, dropped, budget) = chat.prepare(20, encode).unwrap();
        assert_eq!((input.len(), dropped, budget), (13, 3, 16));
        assert_eq!(chat.older_messages(), 6);
        assert!(chat
            .fetch(2, budget, encode_fetch)
            .unwrap()
            .1
            .contains("do not fit"));
        assert_eq!(chat.older_messages(), 6);
        assert_eq!(chat.fetch(3, 32, encode_fetch).unwrap().0.len(), 21);
        assert_eq!(
            chat.turns
                .iter()
                .map(|t| t.user.as_str())
                .collect::<Vec<_>>(),
            ["u1", "u2", "u3", "u4"]
        );
        chat.undo_fetch(2);
        assert_eq!(chat.older_messages(), 6);
        assert_eq!(chat.turns[0].user, "u3");
        assert!(chat.prepare(3, encode).is_err());
        assert_eq!(chat.turns.len(), 2);
        chat.resume("latest").unwrap();
        assert_eq!(chat.turns.len(), 5);
    }
    #[test]
    fn legacy_import_is_atomic_idempotent_and_preserves_reset_separated_chats() {
        let f = Fixture::new();
        let source = f.path().with_extension("jsonl");
        let data = concat!(
            "{\"type\":\"turn\",\"user\":\"first\",\"assistant\":\"reply1\"}\n",
            "{\"type\":\"reset\"}\n{\"type\":\"reset\"}\n",
            "{\"type\":\"turn\",\"user\":\"second\",\"assistant\":\"reply2\"}\n",
            "{\"type\":\"reset\"}\n"
        );
        fs::write(&source, data).unwrap();
        let mut chat = Conversation::open(&source).unwrap();
        assert_eq!(chat.path, f.path());
        assert_eq!(chat.sessions().unwrap().len(), 2);
        assert!(chat.turns.is_empty());
        chat.resume("latest").unwrap();
        assert_eq!(chat.turns[0].user, "second");
        drop(chat);
        assert_eq!(f.chat().sessions().unwrap().len(), 2);
        assert_eq!(fs::read_to_string(&source).unwrap(), data);
        let bad = Fixture::new();
        let bad_source = bad.path().with_extension("jsonl");
        let corrupt = format!("{data}{{\"type\":\"turn\"");
        fs::write(&bad_source, &corrupt).unwrap();
        assert!(Conversation::open(&bad.path()).is_err());
        let db = Connection::open(bad.path()).unwrap();
        assert_eq!(
            db.query_row("SELECT count(*) FROM sessions", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM legacy_imports", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(fs::read_to_string(&bad_source).unwrap(), corrupt);
        fs::write(&bad_source, data).unwrap();
        assert_eq!(bad.chat().sessions().unwrap().len(), 2);
    }
    #[test]
    fn concurrent_new_sessions_work_but_stale_resumes_cannot_overwrite_turns() {
        let f = Fixture::new();
        let mut a = f.chat();
        let mut b = f.chat();
        a.append("a".into(), "one".into()).unwrap();
        b.append("b".into(), "two".into()).unwrap();
        let session = a.session_id.clone();
        b.resume(&session).unwrap();
        a.append("a2".into(), "three".into()).unwrap();
        assert!(b
            .append("stale".into(), "lost".into())
            .unwrap_err()
            .to_string()
            .contains("another TUI"));
        b.resume(&session).unwrap();
        assert_eq!(b.turns.len(), 2);
        assert_eq!(b.turns[1].user, "a2");
        b.append("a3".into(), "four".into()).unwrap();
        assert_eq!(b.sessions().unwrap().len(), 2);
    }
    #[cfg(unix)]
    #[test]
    fn database_has_private_permissions_and_legacy_writer_blocks_import() {
        use std::os::{fd::AsRawFd, unix::fs::PermissionsExt};
        let f = Fixture::new();
        let chat = f.chat();
        assert_eq!(
            fs::metadata(chat.path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let f = Fixture::new();
        let source = f.path().with_extension("jsonl");
        fs::write(&source, b"").unwrap();
        let file = fs::File::open(&source).unwrap();
        assert_eq!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert!(Conversation::open(&f.path())
            .err()
            .unwrap()
            .to_string()
            .contains("in use"));
        drop(file);
        assert!(Conversation::open(&f.path()).is_ok());
    }
    #[test]
    fn retrieval_requires_an_entire_control_response() {
        assert_eq!(fetch_count("\nFETCH_OLDER 10\n"), Some(10));
        for text in [
            "FETCH_OLDER",
            "FETCH_OLDER ten",
            "FETCH_OLDER -2",
            "FETCH_OLDER 2 then answer",
            "Use FETCH_OLDER 2",
        ] {
            assert_eq!(fetch_count(text), None);
        }
        for text in ["", "FET", "FETCH_OLDER", "FETCH_OLDER ", "FETCH_OLDER 12\n"] {
            assert!(fetch_prefix(text));
        }
        for text in [
            "Hello",
            "FETCH_OLDERx",
            "FETCH_OLDER 2 is an example",
            "FETCH_OLDER 1\n2",
        ] {
            assert!(!fetch_prefix(text));
        }
    }
}
