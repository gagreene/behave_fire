//! Allocation diagnostics compiled only under the private allocation-probe feature.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const OUTSIDE: usize = 0;
pub const KERNEL: usize = 1;
pub const ENCODER: usize = 2;

/// Observe the same kernel/write scopes in configured model bindings and tests.
#[derive(Default)]
pub struct Observer;
impl behave_fire::output::RunObserver for Observer {
    fn observe<T>(&self, phase: behave_fire::output::RunPhase, work: impl FnOnce() -> T) -> T {
        let scope = match phase {
            behave_fire::output::RunPhase::Kernel => KERNEL,
            behave_fire::output::RunPhase::Encoder => ENCODER,
        };
        let _guard = Scope::enter(scope);
        work()
    }
}
static GENERATION: AtomicUsize = AtomicUsize::new(0);
static WORKERS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
thread_local! { static VISITED: Cell<(usize, usize)> = const { Cell::new((0, 0)) }; }
static ACTIVE: AtomicBool = AtomicBool::new(false);
static REQUESTS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
static BYTES: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
static CALLS: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
thread_local! { static SCOPE: Cell<usize> = const { Cell::new(OUTSIDE) }; }

pub struct Counter;

#[global_allocator]
static ALLOCATOR: Counter = Counter;

fn count(bytes: usize) {
    if ACTIVE.load(Ordering::Relaxed) {
        let scope = SCOPE.try_with(Cell::get).unwrap_or(OUTSIDE);
        REQUESTS[scope].fetch_add(1, Ordering::Relaxed);
        BYTES[scope].fetch_add(bytes, Ordering::Relaxed);
    }
}

// SAFETY: allocation pointers and layouts are forwarded unchanged to System.
// Counters and constant-initialized TLS do not allocate or dereference pointers.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

/// Reset counters before one isolated diagnostic call, with all workers joined.
pub fn start() {
    ACTIVE.store(false, Ordering::SeqCst);
    GENERATION.fetch_add(1, Ordering::Relaxed);
    for counters in [&REQUESTS, &BYTES, &CALLS, &WORKERS] {
        for counter in counters {
            counter.store(0, Ordering::Relaxed);
        }
    }
    ACTIVE.store(true, Ordering::SeqCst);
}

/// Stop counting after workers join; report request, byte, call, and worker counts for each scope.
pub fn stop() -> [(usize, usize, usize, usize); 3] {
    ACTIVE.store(false, Ordering::SeqCst);
    std::array::from_fn(|i| {
        (
            REQUESTS[i].load(Ordering::Relaxed),
            BYTES[i].load(Ordering::Relaxed),
            CALLS[i].load(Ordering::Relaxed),
            WORKERS[i].load(Ordering::Relaxed),
        )
    })
}

/// Restore the worker's previous scope even when a calculation returns an error.
pub struct Scope {
    previous: usize,
}
impl Scope {
    pub fn enter(scope: usize) -> Self {
        assert!(scope == KERNEL || scope == ENCODER);
        let previous = SCOPE.with(|current| current.replace(scope));
        if ACTIVE.load(Ordering::Relaxed) {
            CALLS[scope].fetch_add(1, Ordering::Relaxed);
            VISITED.with(|visited| {
                let generation = GENERATION.load(Ordering::Relaxed);
                let (previous, mask) = visited.get();
                let mask = if previous == generation { mask } else { 0 };
                if mask & (1 << scope) == 0 {
                    WORKERS[scope].fetch_add(1, Ordering::Relaxed);
                }
                visited.set((generation, mask | (1 << scope)));
            });
        }
        Self { previous }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        SCOPE.with(|current| current.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn distinguishes_scopes_and_restores_after_nested_exit() {
        start();
        black_box(vec![0_u8; 11]);
        {
            let _kernel = Scope::enter(KERNEL);
            black_box(vec![0_u8; 13]);
            {
                let _encoder = Scope::enter(ENCODER);
                black_box(vec![0_u8; 17]);
            }
            black_box(vec![0_u8; 19]);
        }
        let counts = stop();
        assert_eq!(counts, [(1, 11, 0, 0), (2, 32, 1, 1), (1, 17, 1, 1)]);
    }

    #[test]
    fn disabled_probe_does_not_count_or_reuse_old_samples() {
        start();
        {
            let _scope = Scope::enter(KERNEL);
            black_box(vec![0_u8; 23]);
        }
        let first = stop();
        {
            let _scope = Scope::enter(KERNEL);
            black_box(vec![0_u8; 23]);
        }
        assert_eq!(first, stop());
        start();
        assert_eq!(stop(), [(0, 0, 0, 0); 3]);
    }

    #[test]
    fn worker_scopes_do_not_count_thread_setup_as_kernel_work() {
        start();
        std::thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    let _scope = Scope::enter(KERNEL);
                    black_box(vec![0_u8; 29]);
                });
            }
        });
        let counts = stop();
        assert_eq!(counts[1], (4, 116, 4, 4));
        assert_eq!(counts[2], (0, 0, 0, 0));
        assert!(counts[0].0 > 0);
    }
}
