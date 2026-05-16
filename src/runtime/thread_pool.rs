use std::fmt;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    Run(Job),
    Shutdown,
}

pub struct ThreadPool {
    inner: Arc<ThreadPoolInner>,
}

struct ThreadPoolInner {
    threads: usize,
    sender: mpsc::Sender<Message>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl ThreadPool {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let (sender, receiver) = mpsc::channel::<Message>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(threads);

        for _ in 0..threads {
            let receiver = Arc::clone(&receiver);
            workers.push(thread::spawn(move || loop {
                let message = receiver
                    .lock()
                    .expect("thread pool receiver lock poisoned")
                    .recv();
                match message {
                    Ok(Message::Run(job)) => job(),
                    Ok(Message::Shutdown) | Err(_) => break,
                }
            }));
        }

        Self {
            inner: Arc::new(ThreadPoolInner {
                threads,
                sender,
                workers: Mutex::new(workers),
            }),
        }
    }

    pub fn threads(&self) -> usize {
        self.inner.threads
    }

    pub fn parallel_chunks<T, F>(&self, len: usize, chunk_size: usize, f: F) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn(usize, usize) -> T + Send + Sync + 'static,
    {
        if len == 0 {
            return Vec::new();
        }

        let chunk_size = chunk_size.max(1);
        let chunks = len.div_ceil(chunk_size);
        if self.threads() == 1 || chunks == 1 {
            return (0..chunks)
                .map(|chunk| {
                    let start = chunk * chunk_size;
                    let end = (start + chunk_size).min(len);
                    f(start, end)
                })
                .collect();
        }

        let workers = self.threads().min(chunks);
        let (tx, rx) = mpsc::channel();
        let f = Arc::new(f);
        for worker in 0..workers {
            let tx = tx.clone();
            let f = Arc::clone(&f);
            self.inner
                .sender
                .send(Message::Run(Box::new(move || {
                    let mut worker_results = Vec::new();
                    let mut chunk = worker;
                    while chunk < chunks {
                        let start = chunk * chunk_size;
                        let end = (start + chunk_size).min(len);
                        worker_results.push((chunk, f(start, end)));
                        chunk += workers;
                    }
                    tx.send(worker_results)
                        .expect("parallel chunk receiver dropped");
                })))
                .expect("thread pool worker channel closed");
        }
        drop(tx);

        let mut results: Vec<_> = rx.into_iter().flatten().collect();
        results.sort_by_key(|(chunk, _)| *chunk);
        results.into_iter().map(|(_, value)| value).collect()
    }
}

impl Clone for ThreadPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadPool")
            .field("threads", &self.threads())
            .finish_non_exhaustive()
    }
}

impl PartialEq for ThreadPool {
    fn eq(&self, other: &Self) -> bool {
        self.threads() == other.threads()
    }
}

impl Eq for ThreadPool {}

impl Drop for ThreadPoolInner {
    fn drop(&mut self) {
        for _ in 0..self.threads {
            let _ = self.sender.send(Message::Shutdown);
        }

        if let Ok(workers) = self.workers.get_mut() {
            for worker in workers.drain(..) {
                let _ = worker.join();
            }
        }
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

    #[test]
    fn cloned_pool_reuses_workers() {
        let pool = ThreadPool::new(2);
        let cloned = pool.clone();

        let chunks = cloned.parallel_chunks(4, 1, |start, end| end - start);

        assert_eq!(chunks, vec![1, 1, 1, 1]);
        assert_eq!(pool, cloned);
    }
}
