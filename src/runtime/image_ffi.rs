//! Synchronous text-to-image ABI. Providers own tokenization and model state.
pub use super::llm_ffi::{ErrorBuffer, Result};
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};
pub const ABI_VERSION: u32 = 1;
pub const RGB8: u32 = 1;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Info {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub reserved: u32,
    pub stride: usize,
    pub bytes: usize,
}
impl Info {
    pub fn validate(self) -> Result<()> {
        let row = (self.width as usize)
            .checked_mul(3)
            .ok_or("image row overflow")?;
        let size = self
            .stride
            .checked_mul(self.height as usize)
            .ok_or("image size overflow")?;
        if self.width == 0
            || self.height == 0
            || self.width as u64 * self.height as u64 > 1024 * 1024
            || self.format != RGB8
            || self.reserved != 0
            || self.stride != row
            || self.bytes != size
        {
            return Err("invalid RGB8 image metadata or dimensions exceed 1 megapixel".into());
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Callbacks {
    pub user: *mut c_void,
    pub on_progress: Option<unsafe extern "C" fn(*mut c_void, u32, u32)>,
    pub on_done: Option<unsafe extern "C" fn(*mut c_void, i32)>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Api {
    pub abi_version: u32,
    pub struct_size: u32,
    pub build_model:
        Option<unsafe extern "C" fn(*const u8, usize, *mut ErrorBuffer) -> *mut c_void>,
    pub infer: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const u8,
            usize,
            *const Callbacks,
            *mut ErrorBuffer,
        ) -> i32,
    >,
    pub read_output: Option<
        unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut Info, *mut ErrorBuffer) -> i32,
    >,
    pub free_model: Option<unsafe extern "C" fn(*mut c_void)>,
}
impl Api {
    fn validate(self) -> Result<()> {
        if self.abi_version != ABI_VERSION
            || (self.struct_size as usize) < std::mem::size_of::<Self>()
        {
            return Err("unsupported image ABI version or truncated API table".into());
        }
        if self.build_model.is_none()
            || self.infer.is_none()
            || self.read_output.is_none()
            || self.free_model.is_none()
        {
            return Err("image API missing a required function".into());
        }
        Ok(())
    }
}
pub struct Output {
    pub info: Info,
    pub pixels: Vec<u8>,
}
pub type Progress<'a> = dyn FnMut(u32, u32) -> Result<()> + 'a;
struct CallbackState<'a> {
    progress: Option<&'a mut Progress<'a>>,
    last: Option<(u32, u32)>,
    done: Option<i32>,
    error: Option<String>,
}
unsafe extern "C" fn progress(user: *mut c_void, step: u32, total: u32) {
    let s = &mut *user.cast::<CallbackState<'_>>();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        if s.done.is_some() {
            return Err("image progress after completion".into());
        }
        if total == 0 || step > total || s.last.is_some_and(|(old, n)| old > step || n != total) {
            return Err("invalid image progress".into());
        }
        s.last = Some((step, total));
        if let Some(callback) = &mut s.progress {
            callback(step, total)?;
        }
        Ok(())
    }));
    match r {
        Ok(Ok(())) => (),
        Ok(Err(e)) => s.error = Some(e),
        Err(_) => s.error = Some("image progress callback panicked".into()),
    }
}
unsafe extern "C" fn done(user: *mut c_void, status: i32) {
    let s = &mut *user.cast::<CallbackState<'_>>();
    if s.done.replace(status).is_some() {
        s.error = Some("image completed more than once".into());
    }
}
pub struct Model {
    state: NonNull<c_void>,
    api: Api,
    _library: Option<libloading::Library>,
    _thread: PhantomData<Rc<()>>,
}
impl Model {
    /// # Safety
    /// The provider must implement the documented synchronous image ABI.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self> {
        let library = libloading::Library::new(path).map_err(|e| e.to_string())?;
        let get = library
            .get::<unsafe extern "C" fn() -> *const Api>(b"get_image_api\0")
            .map_err(|e| e.to_string())?;
        let pointer = get();
        if pointer.is_null() {
            return Err("get_image_api returned null".into());
        }
        // Read only the fixed header before trusting the advertised table size.
        let version = std::ptr::read(pointer.cast::<u32>());
        let size = std::ptr::read(pointer.cast::<u32>().add(1));
        if version != ABI_VERSION || (size as usize) < std::mem::size_of::<Api>() {
            return Err("unsupported image ABI version or truncated API table".into());
        }
        let mut model = Self::from_api(*pointer, config)?;
        model._library = Some(library);
        Ok(model)
    }
    /// # Safety
    /// Functions must obey ABI layouts, pointers and state lifetime requirements.
    pub unsafe fn from_api(api: Api, config: &[u8]) -> Result<Self> {
        api.validate()?;
        let mut bytes = [0u8; 4096];
        let mut error = ErrorBuffer {
            data: bytes.as_mut_ptr(),
            capacity: bytes.len(),
            length: 0,
        };
        let state = api.build_model.unwrap()(config.as_ptr(), config.len(), &mut error);
        let state = NonNull::new(state)
            .ok_or_else(|| message(&bytes, &error, "image build_model failed"))?;
        Ok(Self {
            state,
            api,
            _library: None,
            _thread: PhantomData,
        })
    }
    pub fn infer<'a>(
        &mut self,
        request: &[u8],
        callback: Option<&'a mut Progress<'a>>,
    ) -> Result<Output> {
        let mut bytes = [0u8; 4096];
        let mut error = ErrorBuffer {
            data: bytes.as_mut_ptr(),
            capacity: bytes.len(),
            length: 0,
        };
        let mut state = CallbackState {
            progress: callback,
            last: None,
            done: None,
            error: None,
        };
        let callbacks = Callbacks {
            user: (&mut state as *mut CallbackState<'_>).cast(),
            on_progress: Some(progress),
            on_done: Some(done),
        };
        let status = unsafe {
            self.api.infer.unwrap()(
                self.state.as_ptr(),
                request.as_ptr(),
                request.len(),
                &callbacks,
                &mut error,
            )
        };
        if let Some(e) = state.error {
            return Err(e);
        }
        if state.done != Some(status) {
            return Err("image infer returned without matching on_done".into());
        }
        if status != 0 {
            return Err(message(&bytes, &error, "image inference failed"));
        }
        if state.last.is_some_and(|(step, total)| step != total) {
            return Err("image completed with incomplete progress".into());
        }
        let mut info = Info::default();
        let status = unsafe {
            self.api.read_output.unwrap()(
                self.state.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut info,
                &mut error,
            )
        };
        if status != 0 {
            return Err(message(&bytes, &error, "image output query failed"));
        }
        info.validate()?;
        let mut pixels = vec![0u8; info.bytes];
        let mut copied = Info::default();
        let status = unsafe {
            self.api.read_output.unwrap()(
                self.state.as_ptr(),
                pixels.as_mut_ptr(),
                pixels.len(),
                &mut copied,
                &mut error,
            )
        };
        if status != 0 {
            return Err(message(&bytes, &error, "image output copy failed"));
        }
        if copied != info {
            return Err("image output metadata changed between reads".into());
        }
        Ok(Output { info, pixels })
    }
}
fn message(bytes: &[u8], error: &ErrorBuffer, fallback: &str) -> String {
    if error.length == 0 {
        fallback.into()
    } else {
        String::from_utf8_lossy(&bytes[..error.length.min(bytes.len())]).into_owned()
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { self.api.free_model.unwrap()(self.state.as_ptr()) };
    }
}
