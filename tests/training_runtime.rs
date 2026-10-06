use serde_json::json;
use std::{fs, process::Command};
#[test]
fn generic_harness_trains_scalar_regression_from_csv_without_mnist() {
    let dir = std::env::temp_dir().join(format!(
        "pup-training-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&dir).unwrap();
    fs::write(dir.join("samples.csv"), "x,y\n1,2\n2,4\n3,6\n").unwrap();
    fs::write(
        dir.join("regression.pup"),
        r#"
x = input("x")
y = input("y")
w = input("weight")
valid = input("valid")
rate = input("rate")
count = reduce(valid, add, 1)
error = reshape(x * w - y, [2])
loss = reduce(error * error * valid, add, 1) / count
gradient = 2.0 * reduce(error * reshape(x, [2]) * valid, add, 1) / count
updated = w - rate * gradient
output updated, loss
"#,
    )
    .unwrap();
    let schema = json!({"columns":[{"name":"x","fields":["x"],"dtype":"f32","shape":[1]},{"name":"y","fields":["y"],"dtype":"f32","shape":[1]}]});
    let file = |tensor: &str| json!({"path":"samples.csv","tensor":tensor,"csv":schema});
    let config = json!({"version":1,"batch_size":2,"datasets":{"x":{"train":file("x"),"test":file("x")},"y":{"train":file("y"),"test":file("y")}},
        "state":[{"input":"weight","output":"updated","shape":[],"init":{"kind":"zeros"}}],"learning_rate":"rate","valid":"valid","loss":"loss"});
    fs::write(
        dir.join("regression.train.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .current_dir(&dir)
        .args([
            "train",
            "regression.pup",
            "--cpu-target",
            "native",
            "--epochs",
            "15",
            "--learning-rate",
            "0.02",
            "--output-dir",
            "result",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("result/metrics.json")).unwrap()).unwrap();
    assert!(report["final_test"]["loss"].as_f64().unwrap() < 0.001);
    assert_eq!(report["profile"]["invocations"], 30); // includes padded final batch every epoch
    assert!(report["final_test"]["accuracy"].is_null());
    assert_eq!(report["build"]["cpu_target"], "native");
    assert!(report["build"]["flags"]
        .as_array()
        .unwrap()
        .contains(&json!("-march=native")));
    assert!(report["build"]["native_fingerprint"].is_string());
    let loaded =
        puppygrad::runtime::data::load(&dir.join("result/state.safetensors"), &Default::default())
            .unwrap();
    let value = f32::from_le_bytes(loaded["weight"].bytes[..4].try_into().unwrap());
    assert!((value - 2.).abs() < 0.02, "weight={value}");
    // Refuse a bad state/output mapping before doing a training run.
    let mut bad = config;
    bad["state"][0]["shape"] = json!([3]);
    fs::write(dir.join("bad.json"), serde_json::to_vec(&bad).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .current_dir(&dir)
        .args([
            "train",
            "regression.pup",
            "--config",
            "bad.json",
            "--output-dir",
            "bad-output",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!dir.join("bad-output").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn mnist_gradients_match_finite_differences_and_padding_is_ignored() {
    use puppygrad::compiler::{
        cpu::{self, Tensor},
        pop::DType,
        source::{self, Context, TensorSpec},
    };
    let mut context = Context::default();
    let shapes = [
        vec![128, 28, 28],
        vec![128],
        vec![784, 64],
        vec![64],
        vec![64, 10],
        vec![10],
        vec![],
        vec![128],
    ];
    for (slot, name) in [
        "images",
        "labels",
        "w1",
        "b1",
        "w2",
        "b2",
        "learning_rate",
        "valid",
    ]
    .iter()
    .enumerate()
    {
        context.tensors.insert(
            (*name).into(),
            TensorSpec {
                slot,
                dtype: if slot < 2 { DType::U8 } else { DType::F32 },
                shape: shapes[slot].clone(),
            },
        );
    }
    let program =
        source::parse_with_context(include_str!("../examples/mnist.pup"), &context).unwrap();
    let exe = cpu::compile_profiled(&program, std::path::Path::new(".cache/pup/tests")).unwrap();
    let metadata = exe.profile_metadata.as_ref().unwrap();
    assert!(
        metadata["workspace_bytes"].as_u64().unwrap() <= 70 * 1024,
        "MNIST workspace exceeded the pinned tinygrad baseline: {metadata}"
    );
    assert!(metadata["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .all(|k| k["packed_bytes"] == 0));
    let f = |x: Vec<f32>| Tensor::F32(x.into());
    let mut images = vec![0u8; 128 * 784];
    for (i, v) in images[..5 * 784].iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let mut labels = vec![0u8; 128];
    labels[..5].copy_from_slice(&[1, 3, 5, 7, 9]);
    let mut valid = vec![0.; 128];
    valid[..5].fill(1.);
    let mut inputs = vec![
        Tensor::U8(images.into()),
        Tensor::U8(labels.into()),
        f(vec![0.002; 784 * 64]),
        f(vec![0.1; 64]),
        f((0..640).map(|i| (i % 19) as f32 * 0.01 - 0.09).collect()),
        f(vec![0.; 10]),
        f(vec![1.]),
        f(valid),
    ];
    let initial = exe.run_profiled(&inputs, 1).unwrap().outputs;
    // Alter all padding pixels; valid masks must remove their loss and gradient contributions.
    if let Tensor::U8(x) = &mut inputs[0] {
        std::sync::Arc::make_mut(x)[5 * 784..].fill(255);
    }
    let changed = exe.run_profiled(&inputs, 1).unwrap().outputs;
    for i in 0..5 {
        assert_eq!(initial[i].f32().unwrap(), changed[i].f32().unwrap());
    }
    for (slot, index, output) in [
        (2, 100 * 64 + 4, 0),
        (3, 3, 1),
        (4, 4 * 10 + 2, 2),
        (5, 0, 3),
    ] {
        let original = inputs[slot].f32().unwrap()[index];
        let gradient = original - initial[output].f32().unwrap()[index];
        let set = |inputs: &mut Vec<Tensor>, value| {
            if let Tensor::F32(x) = &mut inputs[slot] {
                std::sync::Arc::make_mut(x)[index] = value;
            }
        };
        set(&mut inputs, original + 0.002);
        let plus = exe.run_profiled(&inputs, 1).unwrap().outputs[4]
            .f32()
            .unwrap()[0];
        set(&mut inputs, original - 0.002);
        let minus = exe.run_profiled(&inputs, 1).unwrap().outputs[4]
            .f32()
            .unwrap()[0];
        set(&mut inputs, original);
        assert!(
            (gradient - (plus - minus) / 0.004).abs() < 0.001,
            "slot {slot}: analytic {gradient}, numerical {}",
            (plus - minus) / 0.004
        );
    }
}
