//! Versioned LLM application ABI. Keep layouts in sync with include/puppygrad_llm.h.
use std::{ffi::c_void, marker::PhantomData, path::Path, ptr::NonNull, rc::Rc};
pub const ABI_VERSION: u32 = 1;
pub const NO_EOS: u32 = u32::MAX;
pub const DONE_LIMIT: u32 = 0;
pub const DONE_EOS: u32 = 1;
pub const DONE_ERROR: u32 = 2;
pub type Result<T> = std::result::Result<T, String>;
#[repr(C)]
pub struct ErrorBuffer {
    pub data: *mut u8,
    pub capacity: usize,
    pub length: usize,
}
impl ErrorBuffer {
    /// # Safety
    /// `data` must point to `capacity` writable bytes when capacity is nonzero.
    pub unsafe fn set(&mut self, text: &str) {
        self.length = text.len().min(self.capacity);
        if self.length > 0 {
            std::ptr::copy_nonoverlapping(text.as_ptr(), self.data, self.length);
        }
    }
}
struct ErrorStorage {
    bytes: [u8; 4096],
}
impl ErrorStorage {
    fn new() -> Self {
        Self { bytes: [0; 4096] }
    }
    fn buffer(&mut self) -> ErrorBuffer {
        ErrorBuffer {
            data: self.bytes.as_mut_ptr(),
            capacity: self.bytes.len(),
            length: 0,
        }
    }
    fn message(&self, error: &ErrorBuffer, fallback: &str) -> String {
        if error.length == 0 {
            fallback.into()
        } else {
            String::from_utf8_lossy(&self.bytes[..error.length.min(self.bytes.len())]).into_owned()
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Info {
    pub vocab_size: u32,
    pub eos_token: u32,
    pub context_length: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Generation {
    pub max_new_tokens: u64,
    pub temperature: f32,
    pub reserved: u32,
    pub seed: u64,
}
impl Generation {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0. {
            return Err("temperature must be finite and >= 0".into());
        }
        if self.reserved != 0 {
            return Err("reserved generation fields must be zero".into());
        }
        usize::try_from(self.max_new_tokens)
            .map_err(|_| "token limit exceeds host address space")?;
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Callbacks {
    pub on_tokens: Option<unsafe extern "C" fn(*mut c_void, *const u32, usize)>,
    pub on_done: Option<unsafe extern "C" fn(*mut c_void, u32)>,
    pub user: *mut c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Api {
    pub abi_version: u32,
    pub struct_size: u32,
    pub build_model:
        Option<unsafe extern "C" fn(*const u8, usize, *mut Info, *mut ErrorBuffer) -> *mut c_void>,
    pub infer: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const u32,
            usize,
            *const Generation,
            *const Callbacks,
            *mut ErrorBuffer,
        ) -> i32,
    >,
    pub read_output: Option<
        unsafe extern "C" fn(*mut c_void, *mut u32, usize, *mut usize, *mut ErrorBuffer) -> i32,
    >,
    pub free_model: Option<unsafe extern "C" fn(*mut c_void)>,
}
impl Api {
    fn validate(&self) -> Result<()> {
        if self.abi_version != ABI_VERSION {
            return Err(format!(
                "unsupported LLM ABI version {} (expected {ABI_VERSION})",
                self.abi_version
            ));
        }
        if (self.struct_size as usize) < std::mem::size_of::<Self>() {
            return Err("LLM API table is too small".into());
        }
        if self.build_model.is_none()
            || self.infer.is_none()
            || self.read_output.is_none()
            || self.free_model.is_none()
        {
            return Err("LLM API table has a missing required function".into());
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct Output {
    pub tokens: Vec<u32>,
    pub reason: u32,
}
pub type TokenCallback<'a> = dyn FnMut(&[u32]) -> Result<()> + 'a;
struct CallbackState<'a> {
    callback: Option<&'a mut TokenCallback<'a>>,
    seen: Vec<u32>,
    limit: usize,
    vocab: u32,
    done: Option<u32>,
    error: Option<String>,
}
unsafe extern "C" fn tokens_callback(user: *mut c_void, tokens: *const u32, count: usize) {
    let state = &mut *user.cast::<CallbackState<'_>>();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        if state.error.is_some() {
            return Ok(());
        }
        if state.done.is_some() {
            return Err("LLM sent tokens after completion".into());
        }
        if count > state.limit.saturating_sub(state.seen.len()) {
            return Err("LLM callback exceeded token limit".into());
        }
        if count == 0 {
            return Ok(());
        }
        if tokens.is_null() {
            return Err("LLM callback returned a null token buffer".into());
        }
        let tokens = std::slice::from_raw_parts(tokens, count);
        if tokens.iter().any(|&id| id >= state.vocab) {
            return Err("LLM callback returned an invalid token ID".into());
        }
        state.seen.extend_from_slice(tokens);
        if let Some(callback) = &mut state.callback {
            callback(tokens)?;
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => (),
        Ok(Err(e)) => state.error = Some(e),
        Err(_) => state.error = Some("token callback panicked".into()),
    }
}
unsafe extern "C" fn done_callback(user: *mut c_void, reason: u32) {
    let state = &mut *user.cast::<CallbackState<'_>>();
    if state.done.replace(reason).is_some() {
        state.error = Some("LLM completed more than once".into());
    }
    if !matches!(reason, DONE_LIMIT | DONE_EOS | DONE_ERROR) {
        state.error = Some("LLM returned an unknown completion reason".into());
    }
}
/// Owns provider state, keeping the provider library loaded until free_model returns.
/// Calls are exclusive and confined to the owner thread in ABI v1.
pub struct Model {
    state: NonNull<c_void>,
    api: Api,
    pub info: Info,
    _library: Option<libloading::Library>,
    _thread: PhantomData<Rc<()>>,
}
impl Model {
    /// # Safety
    /// The library must implement the documented native LLM ABI. Native plugins
    /// execute with the host process's privileges and must provide valid pointers.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self> {
        let library = libloading::Library::new(path).map_err(|e| e.to_string())?;
        let get_api = library
            .get::<unsafe extern "C" fn() -> *const Api>(b"get_llm_api\0")
            .map_err(|e| format!("missing get_llm_api: {e}"))?;
        let ptr = get_api();
        if ptr.is_null() {
            return Err("get_llm_api returned null".into());
        }
        // Read only the fixed header before checking the size of a foreign table.
        let version = std::ptr::read(ptr.cast::<u32>());
        let size = std::ptr::read(ptr.cast::<u32>().add(1));
        if version != ABI_VERSION {
            return Err(format!(
                "unsupported LLM ABI version {version} (expected {ABI_VERSION})"
            ));
        }
        if (size as usize) < std::mem::size_of::<Api>() {
            return Err("LLM API table is too small".into());
        }
        Self::build(*ptr, config, Some(library))
    }
    /// # Safety
    /// Every function in the table must obey the ABI and remain valid for the
    /// returned Model's lifetime. Use this for statically linked providers.
    pub unsafe fn from_api(api: Api, config: &[u8]) -> Result<Self> {
        Self::build(api, config, None)
    }
    unsafe fn build(api: Api, config: &[u8], library: Option<libloading::Library>) -> Result<Self> {
        api.validate()?;
        let mut info = Info::default();
        let mut storage = ErrorStorage::new();
        let mut error = storage.buffer();
        let ptr = api.build_model.unwrap()(config.as_ptr(), config.len(), &mut info, &mut error);
        let state =
            NonNull::new(ptr).ok_or_else(|| storage.message(&error, "build_model failed"))?;
        let model = Self {
            state,
            api,
            info,
            _library: library,
            _thread: PhantomData,
        };
        if info.vocab_size == 0
            || info.context_length == 0
            || info.eos_token != NO_EOS && info.eos_token >= info.vocab_size
        {
            return Err("LLM returned invalid model metadata".into());
        }
        Ok(model)
    }
    pub fn infer<'a>(
        &mut self,
        input: &[u32],
        generation: Generation,
        on_tokens: Option<&'a mut TokenCallback<'a>>,
    ) -> Result<Output> {
        generation.validate()?;
        if input.is_empty() {
            return Err("prompt must contain at least one token".into());
        }
        if input.iter().any(|&id| id >= self.info.vocab_size) {
            return Err("input contains a token outside model vocabulary".into());
        }
        let required = (input.len() as u64)
            .checked_add(generation.max_new_tokens.saturating_sub(1))
            .ok_or("context length overflow")?;
        if required > self.info.context_length {
            return Err(format!(
                "generation would exceed context length {}",
                self.info.context_length
            ));
        }
        let streaming = on_tokens.is_some();
        let mut callback_state = CallbackState {
            callback: on_tokens,
            seen: vec![],
            limit: generation.max_new_tokens as usize,
            vocab: self.info.vocab_size,
            done: None,
            error: None,
        };
        let callbacks = Callbacks {
            on_tokens: if streaming {
                Some(tokens_callback)
            } else {
                None
            },
            on_done: Some(done_callback),
            user: (&mut callback_state as *mut CallbackState<'_>).cast(),
        };
        let mut storage = ErrorStorage::new();
        let mut error = storage.buffer();
        let status = unsafe {
            self.api.infer.unwrap()(
                self.state.as_ptr(),
                input.as_ptr(),
                input.len(),
                &generation,
                &callbacks,
                &mut error,
            )
        };
        if let Some(e) = callback_state.error {
            return Err(e);
        }
        let reason = callback_state
            .done
            .ok_or("LLM infer returned without on_done")?;
        if status != 0 {
            return Err(storage.message(&error, "LLM inference failed"));
        }
        if reason == DONE_ERROR {
            return Err(storage.message(&error, "LLM reported failed completion"));
        }
        let mut count = 0;
        let status = unsafe {
            self.api.read_output.unwrap()(
                self.state.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut count,
                &mut error,
            )
        };
        if status != 0 {
            return Err(storage.message(&error, "read_output failed"));
        }
        if count > generation.max_new_tokens as usize {
            return Err("LLM output exceeded token limit".into());
        }
        let mut tokens = vec![0; count];
        let mut written = 0;
        let status = unsafe {
            self.api.read_output.unwrap()(
                self.state.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len(),
                &mut written,
                &mut error,
            )
        };
        if status != 0 {
            return Err(storage.message(&error, "read_output failed"));
        }
        if written != count {
            return Err("LLM output changed between queries".into());
        }
        if tokens.iter().any(|&id| id >= self.info.vocab_size) {
            return Err("LLM output contains an invalid token ID".into());
        }
        if streaming && tokens != callback_state.seen {
            return Err("LLM streaming tokens differ from retained output".into());
        }
        if reason == DONE_EOS
            && (self.info.eos_token == NO_EOS || tokens.last() != Some(&self.info.eos_token))
        {
            return Err("LLM reported EOS without an EOS token".into());
        }
        if reason == DONE_LIMIT && count != generation.max_new_tokens as usize {
            return Err("LLM completed before the requested token limit without EOS".into());
        }
        Ok(Output { tokens, reason })
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { self.api.free_model.unwrap()(self.state.as_ptr()) };
    }
}
