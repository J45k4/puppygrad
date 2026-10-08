//! Shared persistent database location and schema for sessions and GPU metadata.
use rusqlite::{
    params_from_iter, types::Value, Connection, OpenFlags, OptionalExtension, TransactionBehavior,
};
use std::{
    cell::RefCell,
    ffi::OsStr,
    marker::PhantomData,
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
thread_local! { static OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) }; }

fn resolve(explicit: Option<&Path>, environment: Option<&OsStr>, cwd: &Path) -> PathBuf {
    let selected = explicit
        .map(Path::to_owned)
        .or_else(|| environment.filter(|p| !p.is_empty()).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("puppygrad.db"));
    if selected.is_absolute() {
        selected
    } else {
        cwd.join(selected)
    }
}

/// The owner thread's explicit override, then PUPPYGRAD_DB, then ./puppygrad.db.
pub fn path() -> Result<PathBuf> {
    let explicit = OVERRIDE.with(|value| value.borrow().clone());
    Ok(resolve(
        explicit.as_deref(),
        std::env::var_os("PUPPYGRAD_DB").as_deref(),
        &std::env::current_dir()?,
    ))
}

/// Keep one worker/library caller's explicit database path scoped to its thread.
/// This also avoids changing process environment for concurrent callers.
pub fn use_path(path: &Path) -> Result<DatabasePathGuard> {
    if path.as_os_str().is_empty() {
        return Err("Database path must not be empty".into());
    }
    let selected = resolve(Some(path), None, &std::env::current_dir()?);
    let previous = OVERRIDE.with(|value| value.replace(Some(selected)));
    Ok(DatabasePathGuard {
        previous,
        _local: PhantomData,
    })
}
pub struct DatabasePathGuard {
    previous: Option<PathBuf>,
    _local: PhantomData<Rc<()>>,
}
impl Drop for DatabasePathGuard {
    fn drop(&mut self) {
        OVERRIDE.with(|value| value.replace(self.previous.take()));
    }
}

pub(crate) fn open(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    drop(options.open(path)?);
    let mut db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(5))?;
    let version: i32 = db.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version > 4 {
        return Err(
            format!("Database version {version} is newer than this application supports").into(),
        );
    }
    db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
    if version < 4 {
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let old_kernel_table =
            has_table(&tx, "kernel_modules")? && !has_column(&tx, "kernel_modules", "cache_dir")?;
        if old_kernel_table {
            tx.execute_batch("ALTER TABLE kernel_modules RENAME TO old_kernel_modules;")?;
        }
        tx.execute_batch(SCHEMA)?;
        if !has_column(&tx, "turns", "created_at")? {
            tx.execute_batch("ALTER TABLE turns ADD COLUMN created_at INTEGER;")?;
        }
        if old_kernel_table {
            // Explicitly reopening an old kernel DB upgrades it in place.
            let directory = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let root = directory.canonicalize()?.to_string_lossy().into_owned();
            tx.execute(&format!("INSERT INTO kernel_modules(cache_dir,{KERNEL_COLUMNS}) SELECT ?1,{KERNEL_COLUMNS} FROM old_kernel_modules"),[root])?;
            tx.execute_batch("DROP TABLE old_kernel_modules; CREATE INDEX IF NOT EXISTS kernel_modules_last_used ON kernel_modules(last_used_at);")?;
        }
        tx.execute_batch("PRAGMA user_version=4;")?;
        tx.commit()?;
    }
    Ok(db)
}

fn has_table(db: &Connection, table: &str) -> rusqlite::Result<bool> {
    Ok(db
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
            [table],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}
fn has_column(db: &Connection, table: &str, column: &str) -> rusqlite::Result<bool> {
    let mut query = db.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = query
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names.iter().any(|name| name == column))
}
const KERNEL_COLUMNS: &str = "cache_key,backend,architecture,compiler_version,compiler_options,source_path,binary_path,source_sha256,binary_sha256,source_bytes,binary_bytes,kernel_count,gemm_count,created_at,last_used_at,load_count,hit_count,compile_count";

/// Copy known tables once from a consistent read-only snapshot. Keep originals.
pub(crate) fn import(db: &mut Connection, source: &Path, destination: &Path) -> Result<()> {
    if !source.is_file() {
        return Ok(());
    }
    let source = source.canonicalize()?;
    if source == destination.canonicalize()? {
        return Ok(());
    }
    let label = source.to_string_lossy().into_owned();
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    if tx
        .query_row(
            "SELECT 1 FROM database_imports WHERE path=?1",
            [&label],
            |_| Ok(()),
        )
        .optional()?
        .is_some()
    {
        return Ok(());
    }
    let mut legacy = Connection::open_with_flags(&source, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    legacy.busy_timeout(Duration::from_secs(5))?;
    let snapshot = legacy.transaction()?;
    for (table, columns) in [
        ("sessions", "id,title,model,created_at,updated_at"),
        ("turns", "session_id,ordinal,user,assistant,created_at"),
        ("legacy_imports", "path"),
    ] {
        if !has_table(&snapshot, table)? {
            continue;
        }
        let select_columns = if table == "turns" && !has_column(&snapshot, "turns", "created_at")? {
            "session_id,ordinal,user,assistant,NULL AS created_at"
        } else {
            columns
        };
        let mut select = snapshot.prepare(&format!("SELECT {select_columns} FROM {table}"))?;
        let count = select.column_count();
        let placeholders = vec!["?"; count].join(",");
        let mut insert = tx.prepare(&format!(
            "INSERT OR IGNORE INTO {table}({columns}) VALUES({placeholders})"
        ))?;
        let mut rows = select.query([])?;
        while let Some(row) = rows.next()? {
            let values = (0..count)
                .map(|i| row.get::<_, Value>(i))
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let added = insert.execute(params_from_iter(&values))?;
            if table == "turns" && added == 0 {
                let same: bool = tx.query_row(
                    "SELECT user=?3 AND assistant=?4 FROM turns WHERE session_id=?1 AND ordinal=?2",
                    params_from_iter(values.iter().take(4)),
                    |r| r.get(0),
                )?;
                if !same {
                    return Err(
                        "Conflicting session IDs in legacy database; import rolled back".into(),
                    );
                }
                tx.execute("UPDATE turns SET created_at=?3 WHERE session_id=?1 AND ordinal=?2 AND created_at IS NULL",
                    rusqlite::params![values[0], values[1], values[4]])?;
            }
        }
    }
    if has_table(&snapshot, "kernel_modules")? {
        let select_columns = if has_column(&snapshot, "kernel_modules", "cache_dir")? {
            format!("cache_dir,{KERNEL_COLUMNS}")
        } else {
            format!("?1 AS cache_dir,{KERNEL_COLUMNS}")
        };
        let root = source.parent().unwrap().to_string_lossy().into_owned();
        let mut select =
            snapshot.prepare(&format!("SELECT {select_columns} FROM kernel_modules"))?;
        let count = select.column_count();
        let placeholders = vec!["?"; count].join(",");
        let mut insert = tx.prepare(&format!("INSERT OR IGNORE INTO kernel_modules(cache_dir,{KERNEL_COLUMNS}) VALUES({placeholders})"))?;
        let mut rows = if select.parameter_count() == 0 {
            select.query([])?
        } else {
            select.query([root])?
        };
        while let Some(row) = rows.next()? {
            let values = (0..count)
                .map(|i| row.get::<_, Value>(i))
                .collect::<rusqlite::Result<Vec<_>>>()?;
            insert.execute(params_from_iter(values))?;
        }
    }
    tx.execute("INSERT INTO database_imports VALUES(?1)", [label])?;
    tx.commit()?;
    Ok(())
}

const SCHEMA: &str = r#"CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT,
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS turns (
            session_id TEXT NOT NULL REFERENCES sessions(id), ordinal INTEGER NOT NULL,
            user TEXT NOT NULL, assistant TEXT NOT NULL, created_at INTEGER,
            PRIMARY KEY(session_id, ordinal)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC);
        CREATE TABLE IF NOT EXISTS legacy_imports (path TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS kernel_modules (
            cache_dir TEXT NOT NULL, cache_key TEXT NOT NULL,
            backend TEXT NOT NULL, architecture TEXT NOT NULL,
            compiler_version TEXT NOT NULL, compiler_options TEXT NOT NULL,
            source_path TEXT NOT NULL, binary_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL, binary_sha256 TEXT NOT NULL,
            source_bytes INTEGER NOT NULL, binary_bytes INTEGER NOT NULL,
            kernel_count INTEGER NOT NULL, gemm_count INTEGER NOT NULL,
            created_at INTEGER NOT NULL, last_used_at INTEGER NOT NULL,
            load_count INTEGER NOT NULL, hit_count INTEGER NOT NULL,
            compile_count INTEGER NOT NULL,
            PRIMARY KEY(cache_dir, cache_key)
        );
        CREATE INDEX IF NOT EXISTS kernel_modules_last_used ON kernel_modules(last_used_at);
        CREATE TABLE IF NOT EXISTS database_imports (path TEXT PRIMARY KEY);
"#;

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "puppygrad-database-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    const OLD_SESSIONS: &str = "CREATE TABLE sessions(id TEXT PRIMARY KEY,title TEXT NOT NULL,model TEXT,created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL);
        CREATE TABLE turns(session_id TEXT NOT NULL REFERENCES sessions(id),ordinal INTEGER NOT NULL,user TEXT NOT NULL,assistant TEXT NOT NULL,PRIMARY KEY(session_id,ordinal)) WITHOUT ROWID;
        CREATE TABLE legacy_imports(path TEXT PRIMARY KEY); PRAGMA user_version=1;
        INSERT INTO sessions VALUES('saved','Earlier chat','qwen3',10,20);
        INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('saved',0,'Remember puppy','OK puppy');
        INSERT INTO legacy_imports VALUES('/old/chat.jsonl');";

    fn old_kernels(path: &Path) -> Connection {
        let db = Connection::open(path).unwrap();
        // The previous per-directory index had no cache_dir column.
        let schema = SCHEMA
            .replace(
                "cache_dir TEXT NOT NULL, cache_key TEXT NOT NULL",
                "cache_key TEXT NOT NULL",
            )
            .replace(
                "PRIMARY KEY(cache_dir, cache_key)",
                "PRIMARY KEY(cache_key)",
            );
        db.execute_batch(&schema).unwrap();
        db.execute_batch("PRAGMA user_version=1; INSERT INTO kernel_modules VALUES('module','hip','gfx1201','9.0','[]','module.hip','module.hsaco','source-hash','binary-hash',10,20,3,1,100,200,7,5,2);").unwrap();
        db
    }

    #[test]
    fn location_precedence_and_relative_paths_use_the_calling_directory() {
        let cwd = Path::new("/project");
        assert_eq!(resolve(None, None, cwd), cwd.join("puppygrad.db"));
        assert_eq!(
            resolve(None, Some(OsStr::new("")), cwd),
            cwd.join("puppygrad.db")
        );
        assert_eq!(
            resolve(None, Some(OsStr::new("data/chat.db")), cwd),
            cwd.join("data/chat.db")
        );
        assert_eq!(
            resolve(None, Some(OsStr::new("/shared/all.db")), cwd),
            Path::new("/shared/all.db")
        );
        assert_eq!(
            resolve(
                Some(Path::new("chosen.db")),
                Some(OsStr::new("ignored.db")),
                cwd
            ),
            cwd.join("chosen.db")
        );
    }

    #[test]
    fn scoped_overrides_restore_and_do_not_affect_other_threads() {
        let before = path().unwrap();
        let f = Fixture::new();
        let chosen = f.0.join("explicit.db");
        let outer = use_path(&chosen).unwrap();
        assert_eq!(path().unwrap(), chosen);
        let inner = use_path(&f.0.join("inner.db")).unwrap();
        assert_eq!(path().unwrap(), f.0.join("inner.db"));
        drop(inner);
        assert_eq!(path().unwrap(), chosen);
        let other = std::thread::spawn(|| path().unwrap()).join().unwrap();
        assert_eq!(other, before);
        drop(outer);
        assert_eq!(path().unwrap(), before);
    }

    #[test]
    fn session_schema_upgrade_preserves_messages_and_enforces_foreign_keys() {
        let f = Fixture::new();
        let target = f.0.join("puppygrad.db");
        let old = Connection::open(&target).unwrap();
        old.execute_batch(OLD_SESSIONS).unwrap();
        drop(old);
        let db = open(&target).unwrap();
        assert!(has_table(&db, "kernel_modules").unwrap());
        assert_eq!(
            db.query_row("SELECT assistant FROM turns", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "OK puppy"
        );
        assert!(db
            .execute("INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('absent',0,'user','answer')", [])
            .is_err());
        assert_eq!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
                .unwrap(),
            4
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let fresh = f.0.join("fresh.db");
            drop(open(&fresh).unwrap());
            assert_eq!(
                std::fs::metadata(fresh).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn version_two_upgrade_adds_settings_without_changing_saved_chats() {
        let f = Fixture::new();
        let path = f.0.join("puppygrad.db");
        let db = open(&path).unwrap();
        db.execute_batch(
            "DROP TABLE app_settings; ALTER TABLE turns DROP COLUMN created_at; PRAGMA user_version=2;
            INSERT INTO sessions VALUES('chat','Saved chat','qwen3-1.7b',1,2);
            INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('chat',0,'hello','hi');",
        )
        .unwrap();
        drop(db);
        let db = open(&path).unwrap();
        assert!(has_table(&db, "app_settings").unwrap());
        assert!(has_table(&db, "kernel_modules").unwrap());
        assert_eq!(db.query_row("SELECT model,user,assistant FROM sessions JOIN turns ON sessions.id=turns.session_id", [], |r| Ok((r.get::<_,String>(0)?, r.get::<_,String>(1)?, r.get::<_,String>(2)?))).unwrap(),
            ("qwen3-1.7b".into(), "hello".into(), "hi".into()));
        assert_eq!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
                .unwrap(),
            4
        );
    }

    #[test]
    fn version_three_upgrade_keeps_old_turn_times_unknown_and_preserves_preferences() {
        let f = Fixture::new();
        let path = f.0.join("puppygrad.db");
        let db = open(&path).unwrap();
        db.execute_batch(
            "ALTER TABLE turns DROP COLUMN created_at; PRAGMA user_version=3;
            INSERT INTO sessions VALUES('chat','Saved chat','qwen3-1.7b',1,2);
            INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('chat',0,'hello','hi');
            INSERT INTO app_settings VALUES('last_model','qwen3-1.7b');",
        )
        .unwrap();
        drop(db);
        let db = open(&path).unwrap();
        assert!(has_column(&db, "turns", "created_at").unwrap());
        assert_eq!(
            db.query_row("SELECT user,assistant,created_at FROM turns", [], |r| Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<i64>>(2)?
            )))
            .unwrap(),
            ("hello".into(), "hi".into(), None)
        );
        assert_eq!(
            db.query_row(
                "SELECT value FROM app_settings WHERE key='last_model'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
            "qwen3-1.7b"
        );
        assert_eq!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
                .unwrap(),
            4
        );
    }

    #[test]
    fn database_import_preserves_known_timestamps_and_fills_unknown_duplicate_times() {
        let f = Fixture::new();
        let target = f.0.join("puppygrad.db");
        let archive = f.0.join("archive.db");
        let original = open(&archive).unwrap();
        original.execute_batch("INSERT INTO sessions VALUES('chat','Chat',NULL,123,123);
            INSERT INTO turns(session_id,ordinal,user,assistant,created_at) VALUES('chat',0,'hello','hi',123);
            INSERT INTO turns(session_id,ordinal,user,assistant,created_at) VALUES('chat',1,'second','reply',456);").unwrap();
        let mut db = open(&target).unwrap();
        db.execute_batch(
            "INSERT INTO sessions VALUES('chat','Chat',NULL,123,123);
            INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('chat',0,'hello','hi');",
        )
        .unwrap();
        import(&mut db, &archive, &target).unwrap();
        let timestamps = db
            .prepare("SELECT created_at FROM turns ORDER BY ordinal")
            .unwrap()
            .query_map([], |r| r.get::<_, Option<i64>>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(timestamps, [Some(123), Some(456)]);
        assert_eq!(
            original
                .query_row("SELECT created_at FROM turns WHERE ordinal=0", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            123
        );
        import(&mut db, &archive, &target).unwrap();
        assert_eq!(
            db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn kernel_schema_upgrade_preserves_counts_and_recreates_the_index() {
        let f = Fixture::new();
        let target = f.0.join("cache.sqlite3");
        drop(old_kernels(&target));
        let db = open(&target).unwrap();
        let row = db
            .query_row(
                "SELECT cache_dir,load_count,hit_count,compile_count FROM kernel_modules",
                [],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(row, (f.0.to_string_lossy().into_owned(), 7, 5, 2));
        assert!(has_table(&db, "sessions").unwrap());
        let table: String = db
            .query_row(
                "SELECT tbl_name FROM sqlite_master WHERE name='kernel_modules_last_used'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(table, "kernel_modules");
    }

    #[test]
    fn separate_session_and_kernel_databases_import_once_and_leave_originals_intact() {
        let f = Fixture::new();
        let target = f.0.join("puppygrad.db");
        let sessions = f.0.join("conversation.sqlite3");
        let source = Connection::open(&sessions).unwrap();
        source.execute_batch(OLD_SESSIONS).unwrap();
        let kernel_dir = f.0.join("hip");
        std::fs::create_dir_all(&kernel_dir).unwrap();
        let kernels = kernel_dir.join("cache.sqlite3");
        let kernel_source = old_kernels(&kernels);
        let mut db = open(&target).unwrap();
        for _ in 0..2 {
            import(&mut db, &sessions, &target).unwrap();
            import(&mut db, &kernels, &target).unwrap();
        }
        assert_eq!(
            db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM database_imports", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            db.query_row("SELECT path FROM legacy_imports", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            "/old/chat.jsonl"
        );
        assert_eq!(
            db.query_row("SELECT cache_dir FROM kernel_modules", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            kernel_dir.to_string_lossy()
        );
        assert_eq!(
            db.query_row("SELECT load_count FROM kernel_modules", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            7
        );
        for original in [source, kernel_source] {
            assert_eq!(
                original
                    .query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
                    .unwrap(),
                1
            );
        }
        // Importing the current file into itself is harmless.
        import(&mut db, &target, &target).unwrap();
    }

    #[test]
    fn conflicting_archive_turns_roll_back_the_entire_import() {
        let f = Fixture::new();
        let target = f.0.join("puppygrad.db");
        let archive = f.0.join("old.sqlite3");
        let source = Connection::open(&archive).unwrap();
        source.execute_batch(OLD_SESSIONS).unwrap();
        source.execute_batch("INSERT INTO sessions VALUES('another','Another chat',NULL,1,2); INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('another',0,'question','answer');").unwrap();
        let mut db = open(&target).unwrap();
        db.execute_batch("INSERT INTO sessions VALUES('saved','Current chat',NULL,1,2); INSERT INTO turns(session_id,ordinal,user,assistant) VALUES('saved',0,'Different question','Different answer');").unwrap();
        assert!(import(&mut db, &archive, &target).is_err());
        assert_eq!(
            db.query_row("SELECT count(*) FROM sessions", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            db.query_row("SELECT assistant FROM turns", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "Different answer"
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM database_imports", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}
