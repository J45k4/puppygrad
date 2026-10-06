use puppygrad::runtime::image_ffi::{self as ffi, Api, Callbacks, ErrorBuffer, Info, Model};
use std::{
    ffi::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};
static FREED: AtomicUsize = AtomicUsize::new(0);
unsafe extern "C" fn build(_: *const u8, _: usize, _: *mut ErrorBuffer) -> *mut c_void {
    Box::into_raw(Box::new(0u8)).cast()
}
unsafe extern "C" fn infer(
    state: *mut c_void,
    request: *const u8,
    len: usize,
    cb: *const Callbacks,
    _: *mut ErrorBuffer,
) -> i32 {
    let mode = std::slice::from_raw_parts(request, len)[0];
    *state.cast::<u8>() = mode;
    let c = &*cb;
    c.on_progress.unwrap()(c.user, 0, 2);
    if mode == 1 {
        c.on_progress.unwrap()(c.user, 2, 2);
        c.on_progress.unwrap()(c.user, 1, 2);
    } else {
        c.on_progress.unwrap()(c.user, 2, 2);
    }
    if mode != 2 {
        c.on_done.unwrap()(c.user, 0);
    }
    if mode == 3 {
        c.on_done.unwrap()(c.user, 0);
    }
    0
}
unsafe extern "C" fn read(
    state: *mut c_void,
    dst: *mut u8,
    cap: usize,
    info: *mut Info,
    _: *mut ErrorBuffer,
) -> i32 {
    let mode = *state.cast::<u8>();
    *info = Info {
        width: 2,
        height: 1,
        format: ffi::RGB8,
        reserved: 0,
        stride: 6,
        bytes: if mode == 4 { usize::MAX } else { 6 },
    };
    if !dst.is_null() {
        if mode == 5 {
            (*info).height = 2;
        }
        if cap < 6 {
            return 1;
        }
        std::ptr::copy_nonoverlapping([255u8, 0, 0, 0, 255, 0].as_ptr(), dst, 6);
    }
    0
}
unsafe extern "C" fn free(state: *mut c_void) {
    drop(Box::from_raw(state.cast::<u8>()));
    FREED.fetch_add(1, Ordering::SeqCst);
}
fn api() -> Api {
    Api {
        abi_version: ffi::ABI_VERSION,
        struct_size: std::mem::size_of::<Api>() as u32,
        build_model: Some(build),
        infer: Some(infer),
        read_output: Some(read),
        free_model: Some(free),
    }
}
#[test]
fn image_contract_checks_completion_metadata_and_lifecycle() {
    let before = FREED.load(Ordering::SeqCst);
    {
        let mut model = unsafe { Model::from_api(api(), b"{}").unwrap() };
        let mut updates = vec![];
        let mut progress = |step, total| {
            updates.push((step, total));
            Ok(())
        };
        let out = model.infer(&[0], Some(&mut progress)).unwrap();
        assert_eq!(out.pixels, [255, 0, 0, 0, 255, 0]);
        assert_eq!(updates, [(0, 2), (2, 2)]);
        for (mode, message) in [
            (1, "invalid image progress"),
            (2, "without matching on_done"),
            (3, "more than once"),
            (4, "invalid RGB8"),
            (5, "metadata changed"),
        ] {
            assert!(model.infer(&[mode], None).err().unwrap().contains(message));
        }
        let mut bad_progress = |_, _| -> ffi::Result<()> { panic!("callback test") };
        assert!(model
            .infer(&[0], Some(&mut bad_progress))
            .err()
            .unwrap()
            .contains("panicked"));
        assert!(model.infer(&[0], None).is_ok());
    }
    assert_eq!(FREED.load(Ordering::SeqCst), before + 1);
    let mut bad = api();
    bad.struct_size = 8;
    assert!(unsafe { Model::from_api(bad, b"{}") }.err().is_some());
    let mut bad = api();
    bad.infer = None;
    assert!(unsafe { Model::from_api(bad, b"{}") }.err().is_some());
}
#[test]
fn c_header_can_supply_a_shared_image_provider() {
    let dir = std::path::Path::new(".cache/image-ffi-test");
    std::fs::create_dir_all(dir).unwrap();
    let source = r#"
#include "puppygrad_image.h"
#include <stdlib.h>
#include <string.h>
static void *build(const uint8_t *c,size_t n,PupImageError *e) { (void)c;(void)n;(void)e;return malloc(1); }
static int32_t infer(void *s,const uint8_t *r,size_t n,const PupImageCallbacks *c,PupImageError *e) { (void)s;(void)r;(void)n;(void)e;if(c&&c->on_progress)c->on_progress(c->user,1,1);if(c&&c->on_done)c->on_done(c->user,0);return 0; }
static int32_t read(void *s,uint8_t *p,size_t n,PupImageInfo *i,PupImageError *e) { (void)s;(void)e;*i=(PupImageInfo){1,1,1,0,3,3};if(p){if(n<3)return 1;memcpy(p,"abc",3);}return 0; }
static PupImageApi api={1,sizeof(PupImageApi),build,infer,read,free};
const PupImageApi *get_image_api(void) { return &api; }
"#;
    std::fs::write(dir.join("provider.c"), source).unwrap();
    let status = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-Iinclude"])
        .arg(dir.join("provider.c"))
        .arg("-o")
        .arg(dir.join("provider.so"))
        .status()
        .unwrap();
    assert!(status.success());
    let mut model = unsafe { Model::load(&dir.join("provider.so"), b"{}").unwrap() };
    assert_eq!(model.infer(b"{}", None).unwrap().pixels, b"abc");
    let file = dir.join("output.png");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .arg("image")
        .arg(dir.join("provider.so"))
        .args(["--prompt", "test", "--output"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image = image::open(&file).unwrap().to_rgb8();
    assert_eq!(image.dimensions(), (1, 1));
    assert_eq!(image.get_pixel(0, 0).0, *b"abc");
}

#[test]
fn image_cli_rejects_unavailable_device_before_loading_assets() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_puppygrad"))
        .args([
            "image",
            "examples/image.pup",
            "--prompt",
            "test",
            "--device",
            "cuda:0",
            "--model-dir",
            "missing-assets",
            "--download",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--device cpu"));
}

#[test]
fn image_cli_rejects_invalid_requests_before_loading_assets() {
    for args in [
        ["--steps", "0"],
        ["--guidance-scale", "NaN"],
        ["--width", "0"],
    ] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_puppygrad"))
            .args([
                "image",
                "examples/image.pup",
                "--prompt",
                "test",
                "--model-dir",
                "missing-assets",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("must be positive"));
    }
}
