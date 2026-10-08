//! Rebuildable SQLite bookkeeping; generated GPU source and binaries stay files.
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(super) struct Identity<'a> {
    pub key: &'a str,
    pub backend: &'a str,
    pub architecture: &'a str,
    pub compiler_version: &'a str,
    pub compiler_options: &'a str,
}

pub(super) struct Artifacts {
    pub source_sha256: String,
    pub binary_sha256: String,
    pub source_bytes: usize,
    pub binary_bytes: usize,
}
impl Artifacts {
    pub fn new(source: &[u8], binary: &[u8]) -> Self {
        Self {
            source_sha256: format!("{:x}", Sha256::digest(source)),
            binary_sha256: format!("{:x}", Sha256::digest(binary)),
            source_bytes: source.len(),
            binary_bytes: binary.len(),
        }
    }
}

pub(super) struct Index(Connection, String);
impl Index {
    pub fn open(directory: &Path) -> Result<Self> {
        let path = crate::database::path()?;
        let mut db = crate::database::open(&path)?;
        crate::database::import(&mut db, &directory.join("cache.sqlite3"), &path)?;
        Ok(Self(
            db,
            directory.canonicalize()?.to_string_lossy().into_owned(),
        ))
    }

    /// No row means legacy files or a rebuilt index: normal source verification
    /// and GPU module loading still apply before such files can be registered.
    pub fn allows(&self, identity: &Identity<'_>, artifacts: &Artifacts) -> Result<bool> {
        let matches = self
            .0
            .query_row(
                "SELECT
            backend=?2 AND architecture=?3 AND compiler_version=?4 AND compiler_options=?5
            AND source_sha256=?6 AND binary_sha256=?7 AND source_bytes=?8 AND binary_bytes=?9
            FROM kernel_modules WHERE cache_key=?1 AND cache_dir=?10",
                params![
                    identity.key,
                    identity.backend,
                    identity.architecture,
                    identity.compiler_version,
                    identity.compiler_options,
                    artifacts.source_sha256,
                    artifacts.binary_sha256,
                    artifacts.source_bytes,
                    artifacts.binary_bytes,
                    self.1
                ],
                |r| r.get::<_, bool>(0),
            )
            .optional()?;
        Ok(matches.unwrap_or(true))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        identity: &Identity<'_>,
        artifacts: &Artifacts,
        source_path: &str,
        binary_path: &str,
        kernels: usize,
        gemms: usize,
        hit: bool,
    ) -> Result<()> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as i64;
        self.0.execute("INSERT INTO kernel_modules (
            cache_dir,cache_key,backend,architecture,compiler_version,compiler_options,source_path,binary_path,
            source_sha256,binary_sha256,source_bytes,binary_bytes,kernel_count,gemm_count,
            created_at,last_used_at,load_count,hit_count,compile_count
        ) VALUES(?17,?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?14,1,?15,?16)
        ON CONFLICT(cache_dir,cache_key) DO UPDATE SET
            backend=excluded.backend,architecture=excluded.architecture,
            compiler_version=excluded.compiler_version,compiler_options=excluded.compiler_options,
            source_path=excluded.source_path,binary_path=excluded.binary_path,
            source_sha256=excluded.source_sha256,binary_sha256=excluded.binary_sha256,
            source_bytes=excluded.source_bytes,binary_bytes=excluded.binary_bytes,
            kernel_count=excluded.kernel_count,gemm_count=excluded.gemm_count,
            last_used_at=MAX(kernel_modules.last_used_at,excluded.last_used_at),
            load_count=kernel_modules.load_count+1,hit_count=kernel_modules.hit_count+excluded.hit_count,
            compile_count=kernel_modules.compile_count+excluded.compile_count",
            params![identity.key, identity.backend, identity.architecture, identity.compiler_version,
                identity.compiler_options, source_path, binary_path, artifacts.source_sha256,
                artifacts.binary_sha256, artifacts.source_bytes, artifacts.binary_bytes,
                kernels, gemms, now, i64::from(hit), i64::from(!hit), self.1])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "puppygrad-kernel-index-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn index(&self) -> Index {
            let _scope = crate::database::use_path(&self.0.join("puppygrad.db")).unwrap();
            Index::open(&self.0).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn identity() -> Identity<'static> {
        Identity {
            key: "test-module",
            backend: "hip",
            architecture: "gfx1201",
            compiler_version: "7.2",
            compiler_options: "[\"-ffp-contract=off\"]",
        }
    }
    #[test]
    fn legacy_files_register_once_and_counts_survive_reopen() {
        let f = Fixture::new();
        let index = f.index();
        let spec = identity();
        let artifacts = Artifacts::new(b"source", b"binary");
        assert!(index.allows(&spec, &artifacts).unwrap());
        index
            .record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, true)
            .unwrap();
        drop(index);
        let index = f.index();
        assert!(index.allows(&spec, &artifacts).unwrap());
        index
            .record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, true)
            .unwrap();
        index
            .record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, false)
            .unwrap();
        let row = index.0.query_row("SELECT count(*),load_count,hit_count,compile_count,source_path,binary_path,source_bytes,binary_bytes FROM kernel_modules",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,String>(4)?,r.get::<_,String>(5)?,r.get::<_,i64>(6)?,r.get::<_,i64>(7)?))).unwrap();
        assert_eq!(
            row,
            (1, 3, 2, 1, "test.hip".into(), "test.hsaco".into(), 6, 6)
        );
    }
    #[test]
    fn incompatible_compilers_architectures_flags_and_same_size_corruption_are_rejected() {
        let f = Fixture::new();
        let index = f.index();
        let spec = identity();
        let artifacts = Artifacts::new(b"source", b"binary");
        index
            .record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, false)
            .unwrap();
        assert!(!index
            .allows(&spec, &Artifacts::new(b"source", b"broken"))
            .unwrap());
        assert!(!index
            .allows(&spec, &Artifacts::new(b"SOURCE", b"binary"))
            .unwrap());
        for changed in [
            Identity {
                architecture: "gfx1100",
                ..identity()
            },
            Identity {
                backend: "cuda",
                ..identity()
            },
            Identity {
                compiler_version: "8.0",
                ..identity()
            },
            Identity {
                compiler_options: "[]",
                ..identity()
            },
        ] {
            assert!(!index.allows(&changed, &artifacts).unwrap());
        }
    }
    #[test]
    fn separate_connections_update_counts_without_losing_hits_and_index_can_be_rebuilt() {
        let f = Fixture::new();
        let a = f.index();
        let b = f.index();
        let spec = identity();
        let artifacts = Artifacts::new(b"source", b"binary");
        a.record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, false)
            .unwrap();
        b.record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, true)
            .unwrap();
        a.record(&spec, &artifacts, "test.hip", "test.hsaco", 8, 3, true)
            .unwrap();
        assert_eq!(
            a.0.query_row("SELECT load_count FROM kernel_modules", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            3
        );
        a.0.execute("INSERT INTO sessions VALUES('saved','Chat',NULL,1,1)", [])
            .unwrap();
        a.0.execute("DELETE FROM kernel_modules", []).unwrap();
        drop(a);
        drop(b);
        let rebuilt = f.index();
        assert!(rebuilt.allows(&spec, &artifacts).unwrap());
        assert_eq!(
            rebuilt
                .0
                .query_row("SELECT count(*) FROM sessions", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
    #[test]
    fn one_database_distinguishes_the_same_key_in_two_cache_directories() {
        let f = Fixture::new();
        let _scope = crate::database::use_path(&f.0.join("puppygrad.db")).unwrap();
        let a_dir = f.0.join("a");
        let b_dir = f.0.join("b");
        std::fs::create_dir_all(&a_dir).unwrap();
        std::fs::create_dir_all(&b_dir).unwrap();
        let a = Index::open(&a_dir).unwrap();
        let b = Index::open(&b_dir).unwrap();
        let spec = identity();
        let first = Artifacts::new(b"source", b"first");
        let second = Artifacts::new(b"source", b"second");
        a.record(&spec, &first, "test.hip", "test.hsaco", 8, 3, false)
            .unwrap();
        b.record(&spec, &second, "test.hip", "test.hsaco", 8, 3, true)
            .unwrap();
        assert!(a.allows(&spec, &first).unwrap());
        assert!(b.allows(&spec, &second).unwrap());
        assert!(!a.allows(&spec, &second).unwrap());
        assert_eq!(
            a.0.query_row("SELECT count(*) FROM kernel_modules", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
    }
    #[test]
    fn future_schema_versions_are_not_rewritten() {
        let f = Fixture::new();
        let db = f.index();
        db.0.execute_batch("PRAGMA user_version=5").unwrap();
        drop(db);
        let _scope = crate::database::use_path(&f.0.join("puppygrad.db")).unwrap();
        assert!(Index::open(&f.0).is_err());
        let db = Connection::open(f.0.join("puppygrad.db")).unwrap();
        assert_eq!(
            db.query_row("PRAGMA user_version", [], |r| r.get::<_, i32>(0))
                .unwrap(),
            5
        );
    }
}
