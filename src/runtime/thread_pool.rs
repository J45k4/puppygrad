use std::sync::mpsc;
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadPool {
    threads: usize,
}

impl ThreadPool {
    pub fn new(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn parallel_chunks<T, F>(&self, len: usize, chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, usize) -> T + Sync,
    {
        if len == 0 {
            return Vec::new();
        }

        let chunk_size = chunk_size.max(1);
        let chunks = len.div_ceil(chunk_size);
        if self.threads == 1 || chunks == 1 {
            return (0..chunks)
                .map(|chunk| {
                    let start = chunk * chunk_size;
                    let end = (start + chunk_size).min(len);
                    f(start, end)
                })
                .collect();
        }

        let workers = self.threads.min(chunks);
        let mut results = Vec::with_capacity(chunks);

        thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            let f = &f;

            for worker in 0..workers {
                let tx = tx.clone();
                scope.spawn(move || {
                    let mut chunk = worker;
                    while chunk < chunks {
                        let start = chunk * chunk_size;
                        let end = (start + chunk_size).min(len);
                        tx.send((chunk, f(start, end)))
                            .expect("parallel chunk receiver dropped");
                        chunk += workers;
                    }
                });
            }

            drop(tx);
            results.extend(rx);
        });

        results.sort_by_key(|(chunk, _)| *chunk);
        results.into_iter().map(|(_, value)| value).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_chunks_preserves_order() {
        let pool = ThreadPool::new(4);
        let chunks = pool.parallel_chunks(10, 3, |start, end| (start..end).collect::<Vec<_>>());
        let values: Vec<usize> = chunks.into_iter().flatten().collect();

        assert_eq!(values, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn single_thread_chunks_preserve_order() {
        let pool = ThreadPool::new(1);
        let chunks = pool.parallel_chunks(7, 2, |start, end| (start..end).sum::<usize>());

        assert_eq!(chunks, vec![1, 5, 9, 6]);
    }
}
