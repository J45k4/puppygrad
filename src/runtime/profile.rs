//! Generic aggregation of the compiler's C profiling ABI, independent of model contracts.
use serde_json::{json, Value};
use std::time::Duration;

pub struct Profile {
    metadata: Value,
    totals: Vec<u64>,
    wall_ms: Vec<f64>,
}
impl Profile {
    pub fn new(metadata: &Value) -> Self {
        Self {
            metadata: metadata.clone(),
            totals: vec![0; metadata["stats_length"].as_u64().unwrap() as usize],
            wall_ms: Vec::new(),
        }
    }
    pub fn record(&mut self, counters: &[u64], wall: Duration) {
        assert_eq!(self.totals.len(), counters.len());
        for (total, &value) in self.totals.iter_mut().zip(counters) {
            *total += value;
        }
        self.wall_ms.push(wall.as_secs_f64() * 1000.);
    }
    pub fn report(&self) -> Value {
        let mut kernels = self.metadata["kernels"].as_array().unwrap().clone();
        for kernel in &mut kernels {
            let i = kernel["stats_offset"].as_u64().unwrap() as usize;
            let calls = self.totals[i];
            kernel["calls"] = json!(calls);
            kernel["elapsed_ms"] = json!(self.totals[i + 1] as f64 / 1e6);
            kernel["packing_ms"] = json!(self.totals[i + 2] as f64 / 1e6);
            kernel["compute_ms"] = json!(self.totals[i + 3] as f64 / 1e6);
            kernel["packed_bytes_total"] = json!(calls * kernel["packed_bytes"].as_u64().unwrap());
            kernel["matmul_flops_total"] = json!(calls * kernel["matmul_flops"].as_u64().unwrap());
        }
        kernels.sort_by(|a, b| {
            b["elapsed_ms"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["elapsed_ms"].as_f64().unwrap())
        });
        let mut times = self.wall_ms.clone();
        times.sort_by(f64::total_cmp);
        let percentile = |p: f64| -> Option<f64> {
            if times.is_empty() {
                return None;
            }
            let index = p * (times.len() - 1) as f64;
            let lo = index.floor() as usize;
            let hi = index.ceil() as usize;
            Some(times[lo] + (times[hi] - times[lo]) * (index - lo as f64))
        };
        json!({"invocations":times.len(),"host_call_median_ms":percentile(0.5),"host_call_p95_ms":percentile(0.95),
            "invocation_ms":self.totals[0] as f64/1e6,"setup_ms":self.totals[1] as f64/1e6,
            "output_copy_ms":self.totals[2] as f64/1e6,"cleanup_ms":self.totals[3] as f64/1e6,
            "workspace_bytes":self.metadata["workspace_bytes"],
            "packed_workspace_bytes":self.metadata["packed_workspace_bytes"],
            "memory_plan":self.metadata["memory_plan"],
            "matmul_stack_scratch_bytes_per_worker":self.metadata["matmul_stack_scratch_bytes_per_worker"],
            "reachable_pops":self.metadata["reachable_pops"],
            "kernel_count":kernels.len(),"kernels":kernels})
    }
}
