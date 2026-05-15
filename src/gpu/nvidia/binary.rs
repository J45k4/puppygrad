use super::asm::SmVersion;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NvidiaBinaryFormat {
    RawSassWords,
    NouveauQmd,
    CudaCubin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterUsage {
    pub gprs_per_thread: u16,
    pub predicate_regs: u8,
    pub uniform_regs: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchMetadata {
    pub local_size: [u16; 3],
    pub shared_static_bytes: u32,
    pub shared_dynamic_bytes: u32,
    pub params_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineCode {
    pub sm: SmVersion,
    pub words: Vec<u32>,
    pub asm: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvidiaKernelBinary {
    pub name: String,
    pub format: NvidiaBinaryFormat,
    pub code: MachineCode,
    pub registers: RegisterUsage,
    pub launch: LaunchMetadata,
}

impl NvidiaKernelBinary {
    pub fn code_size_bytes(&self) -> usize {
        self.code.words.len() * std::mem::size_of::<u32>()
    }

    pub fn is_dispatchable_package(&self) -> bool {
        matches!(
            self.format,
            NvidiaBinaryFormat::NouveauQmd | NvidiaBinaryFormat::CudaCubin
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_sass_words_are_not_a_complete_launch_binary() {
        let bin = NvidiaKernelBinary {
            name: "empty".to_string(),
            format: NvidiaBinaryFormat::RawSassWords,
            code: MachineCode {
                sm: SmVersion::new(80).unwrap(),
                words: vec![0, 1, 2, 3],
                asm: None,
            },
            registers: RegisterUsage {
                gprs_per_thread: 4,
                predicate_regs: 1,
                uniform_regs: 0,
            },
            launch: LaunchMetadata {
                local_size: [1, 1, 1],
                shared_static_bytes: 0,
                shared_dynamic_bytes: 0,
                params_size: 0,
            },
        };

        assert_eq!(bin.code_size_bytes(), 16);
        assert!(!bin.is_dispatchable_package());
    }
}
