use puppygrad::runtime::data::{self, idx, DataType, Format, LoadOptions};
use std::{fs, io::Write};
fn temp() -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "pup-loaders-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    path
}
#[test]
fn idx_preserves_dtype_shape_endianness_and_rejects_bad_payloads() {
    let mut bytes = vec![0, 0, 0x0c, 2];
    bytes.extend(2u32.to_be_bytes());
    bytes.extend(2u32.to_be_bytes());
    for x in [-1i32, 2, 300, -400] {
        bytes.extend(x.to_be_bytes());
    }
    let t = idx::read(&bytes).unwrap();
    assert_eq!(t.dtype, DataType::I32);
    assert_eq!(t.shape, [2, 2]);
    let puppygrad::compiler::cpu::Tensor::I32(batch) = t.batch(&[1, 0], 3).unwrap() else {
        panic!()
    };
    assert_eq!(&*batch, &[300, -400, -1, 2, 0, 0]);
    assert!(t.batch(&[2], 1).is_err());
    assert!(idx::read(&bytes[..bytes.len() - 1]).is_err());
    bytes.push(0);
    assert!(idx::read(&bytes).is_err());
    assert!(idx::read(&[0, 0, 0x08, 0]).is_err());
    let mut overflow = vec![0, 0, 0x0e, 3];
    for _ in 0..3 {
        overflow.extend(u32::MAX.to_be_bytes());
    }
    assert!(idx::read(&overflow).is_err());
}
#[test]
fn auto_gzip_idx_and_explicit_format_override() {
    let dir = temp();
    let path = dir.join("train-images-idx3-ubyte.gz");
    let mut bytes = vec![0, 0, 8, 2];
    bytes.extend(2u32.to_be_bytes());
    bytes.extend(3u32.to_be_bytes());
    bytes.extend([0, 1, 255, 3, 4, 5]);
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(&bytes).unwrap();
    fs::write(&path, gzip.finish().unwrap()).unwrap();
    let loaded = data::load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(loaded["tensor"].shape, [2, 3]);
    assert_eq!(loaded["tensor"].bytes, [0, 1, 255, 3, 4, 5]);
    let odd = dir.join("named.csv");
    fs::write(&odd, &bytes).unwrap();
    assert!(data::load(&odd, &LoadOptions::default()).is_err());
    assert!(data::load(
        &odd,
        &LoadOptions {
            format: Some(Format::Idx),
            csv: None
        }
    )
    .is_ok());
    fs::write(&path, &bytes).unwrap();
    assert!(data::load(&path, &LoadOptions::default()).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn csv_groups_numeric_columns_and_rejects_missing_or_malformed_values() {
    let schema=serde_json::from_str(r#"{"columns":[{"name":"x","fields":["p,0","p1"],"dtype":"f32","shape":[2]},{"name":"label","fields":["class"],"dtype":"i32"}]}"#).unwrap();
    let rows = data::csv::read(b"\"p,0\",p1,class\r\n1.5,2,3\r\n4,5,6\r\n", &schema).unwrap();
    assert_eq!(rows["x"].shape, [2, 2]);
    assert_eq!(rows["label"].shape, [2]);
    assert_eq!(rows["x"].batch(&[1], 1).unwrap().f32().unwrap(), &[4., 5.]);
    for invalid in [
        "\"p,0\",p1,class\n,2,3\n",
        "\"p,0\",p1,class\nNaN,2,3\n",
        "\"p,0\",p1,class\n1,2\n",
        "\"p,0\",p1,class\n\"1\"oops,2,3\n",
    ] {
        assert!(data::csv::read(invalid.as_bytes(), &schema).is_err());
    }
}
#[test]
fn safetensors_returns_named_tensors_without_conversion() {
    use safetensors::tensor::{serialize, Dtype, TensorView};
    let dir = temp();
    let path = dir.join("weights.safetensors");
    let values = [1.25f32, -3.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let byte_values = [255u8, 2];
    fs::write(
        &path,
        serialize(
            [
                ("w", TensorView::new(Dtype::F32, vec![2], &values).unwrap()),
                (
                    "labels",
                    TensorView::new(Dtype::U8, vec![2], &byte_values).unwrap(),
                ),
            ],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let loaded = data::load(&path, &LoadOptions::default()).unwrap();
    assert_eq!(loaded["w"].bytes, values);
    assert_eq!(loaded["labels"].bytes, byte_values);
    assert_eq!(loaded["labels"].class_label(0).unwrap(), 255);
    fs::remove_dir_all(dir).unwrap();
}
