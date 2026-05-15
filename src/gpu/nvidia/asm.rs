use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmVersion(u8);

impl SmVersion {
    pub fn new(sm: u8) -> Result<Self, AsmError> {
        if sm < 70 {
            return Err(AsmError::UnsupportedSm(sm));
        }
        Ok(Self(sm))
    }

    pub fn value(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AsmError {
    UnsupportedSm(u8),
    InvalidLocalSize([u16; 3]),
}

impl fmt::Display for AsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AsmError::UnsupportedSm(sm) => write!(f, "SM {sm} is not supported yet"),
            AsmError::InvalidLocalSize(size) => write!(f, "invalid CUDA local size {size:?}"),
        }
    }
}

impl std::error::Error for AsmError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegFile {
    Gpr,
    Ugpr,
    Pred,
    Upred,
    Barrier,
}

impl RegFile {
    pub fn prefix(self) -> &'static str {
        match self {
            RegFile::Gpr => "r",
            RegFile::Ugpr => "ur",
            RegFile::Pred => "p",
            RegFile::Upred => "up",
            RegFile::Barrier => "b",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    pub file: RegFile,
    pub index: u16,
}

impl Reg {
    pub const fn new(file: RegFile, index: u16) -> Self {
        Self { file, index }
    }

    pub const fn r(index: u16) -> Self {
        Self::new(RegFile::Gpr, index)
    }

    pub const fn p(index: u16) -> Self {
        Self::new(RegFile::Pred, index)
    }
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file.prefix(), self.index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsaValue(pub u32);

impl fmt::Display for SsaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Pred,
    U32,
    U64,
    S32,
    F32,
    F16x2,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            ScalarType::Pred => "pred",
            ScalarType::U32 => "u32",
            ScalarType::U64 => "u64",
            ScalarType::S32 => "s32",
            ScalarType::F32 => "f32",
            ScalarType::F16x2 => "f16x2",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operand {
    Reg(Reg),
    Ssa(SsaValue),
    ImmU32(u32),
    ImmI32(i32),
    ImmF32(f32),
    Zero,
    True,
    False,
    Param(String),
}

impl From<Reg> for Operand {
    fn from(reg: Reg) -> Self {
        Self::Reg(reg)
    }
}

impl From<SsaValue> for Operand {
    fn from(value: SsaValue) -> Self {
        Self::Ssa(value)
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Reg(reg) => reg.fmt(f),
            Operand::Ssa(value) => value.fmt(f),
            Operand::ImmU32(value) => write!(f, "0x{value:x}"),
            Operand::ImmI32(value) => write!(f, "{value}"),
            Operand::ImmF32(value) => write!(f, "{value:?}"),
            Operand::Zero => f.write_str("rz"),
            Operand::True => f.write_str("pT"),
            Operand::False => f.write_str("!pT"),
            Operand::Param(name) => write!(f, "param.{name}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dst {
    Reg(Reg),
    Ssa(SsaValue),
    None,
}

impl From<Reg> for Dst {
    fn from(reg: Reg) -> Self {
        Self::Reg(reg)
    }
}

impl From<SsaValue> for Dst {
    fn from(value: SsaValue) -> Self {
        Self::Ssa(value)
    }
}

impl fmt::Display for Dst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dst::Reg(reg) => reg.fmt(f),
            Dst::Ssa(value) => value.fmt(f),
            Dst::None => f.write_str("_"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Predicate {
    pub value: Operand,
    pub invert: bool,
}

impl Predicate {
    pub const TRUE: Self = Self {
        value: Operand::True,
        invert: false,
    };
}

impl Default for Predicate {
    fn default() -> Self {
        Self::TRUE
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.invert {
            write!(f, "!")?;
        }
        self.value.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    Shared,
    Local,
    Constant,
    Param,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            AddressSpace::Global => "global",
            AddressSpace::Shared => "shared",
            AddressSpace::Local => "local",
            AddressSpace::Constant => "const",
            AddressSpace::Param => "param",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemRef {
    pub space: AddressSpace,
    pub base: Operand,
    pub offset: i32,
}

impl MemRef {
    pub fn new(space: AddressSpace, base: impl Into<Operand>, offset: i32) -> Self {
        Self {
            space,
            base: base.into(),
            offset,
        }
    }
}

impl fmt::Display for MemRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.offset == 0 {
            write!(f, "{}[{}]", self.space, self.base)
        } else if self.offset > 0 {
            write!(f, "{}[{}+{}]", self.space, self.base, self.offset)
        } else {
            write!(f, "{}[{}{}]", self.space, self.base, self.offset)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialReg {
    ThreadIdxX,
    ThreadIdxY,
    ThreadIdxZ,
    BlockIdxX,
    BlockIdxY,
    BlockIdxZ,
    BlockDimX,
    BlockDimY,
    BlockDimZ,
}

impl fmt::Display for SpecialReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            SpecialReg::ThreadIdxX => "tid.x",
            SpecialReg::ThreadIdxY => "tid.y",
            SpecialReg::ThreadIdxZ => "tid.z",
            SpecialReg::BlockIdxX => "ctaid.x",
            SpecialReg::BlockIdxY => "ctaid.y",
            SpecialReg::BlockIdxZ => "ctaid.z",
            SpecialReg::BlockDimX => "ntid.x",
            SpecialReg::BlockDimY => "ntid.y",
            SpecialReg::BlockDimZ => "ntid.z",
        };
        f.write_str(text)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Label(String),
    Comment(String),
    Mov {
        ty: ScalarType,
        dst: Dst,
        src: Operand,
    },
    FAdd {
        dst: Dst,
        a: Operand,
        b: Operand,
    },
    FMul {
        dst: Dst,
        a: Operand,
        b: Operand,
    },
    FFma {
        dst: Dst,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    IAdd3 {
        dst: Dst,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    IMad {
        dst: Dst,
        a: Operand,
        b: Operand,
        c: Operand,
    },
    Shl {
        dst: Dst,
        value: Operand,
        shift: Operand,
    },
    SetP {
        dst: Dst,
        cmp: IntCmp,
        a: Operand,
        b: Operand,
    },
    Load {
        ty: ScalarType,
        dst: Dst,
        addr: MemRef,
    },
    Store {
        ty: ScalarType,
        addr: MemRef,
        value: Operand,
    },
    S2R {
        dst: Dst,
        special: SpecialReg,
    },
    Bra {
        label: String,
    },
    BarSync {
        barrier: u8,
    },
    Exit,
    Nop,
    Copy {
        dst: Dst,
        src: Operand,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for IntCmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            IntCmp::Eq => "eq",
            IntCmp::Ne => "ne",
            IntCmp::Lt => "lt",
            IntCmp::Le => "le",
            IntCmp::Gt => "gt",
            IntCmp::Ge => "ge",
        };
        f.write_str(text)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Label(label) => write!(f, "{label}:"),
            Op::Comment(comment) => write!(f, "// {comment}"),
            Op::Mov { ty, dst, src } => write!(f, "mov.{ty} {dst}, {src}"),
            Op::FAdd { dst, a, b } => write!(f, "fadd.f32 {dst}, {a}, {b}"),
            Op::FMul { dst, a, b } => write!(f, "fmul.f32 {dst}, {a}, {b}"),
            Op::FFma { dst, a, b, c } => write!(f, "ffma.f32 {dst}, {a}, {b}, {c}"),
            Op::IAdd3 { dst, a, b, c } => write!(f, "iadd3.u32 {dst}, {a}, {b}, {c}"),
            Op::IMad { dst, a, b, c } => write!(f, "imad.u32 {dst}, {a}, {b}, {c}"),
            Op::Shl { dst, value, shift } => write!(f, "shl.u32 {dst}, {value}, {shift}"),
            Op::SetP { dst, cmp, a, b } => write!(f, "setp.{cmp}.u32 {dst}, {a}, {b}"),
            Op::Load { ty, dst, addr } => write!(f, "ld.{ty} {dst}, {addr}"),
            Op::Store { ty, addr, value } => write!(f, "st.{ty} {addr}, {value}"),
            Op::S2R { dst, special } => write!(f, "s2r.u32 {dst}, {special}"),
            Op::Bra { label } => write!(f, "bra {label}"),
            Op::BarSync { barrier } => write!(f, "bar.sync {barrier}"),
            Op::Exit => f.write_str("exit"),
            Op::Nop => f.write_str("nop"),
            Op::Copy { dst, src } => write!(f, "copy {dst}, {src}"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SchedDeps {
    pub delay: u8,
    pub yield_hint: bool,
    pub read_barrier: Option<u8>,
    pub write_barrier: Option<u8>,
    pub wait_barrier_mask: u8,
    pub reuse_mask: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Instr {
    pub pred: Predicate,
    pub op: Op,
    pub deps: SchedDeps,
}

impl Instr {
    pub fn new(op: Op) -> Self {
        Self {
            pred: Predicate::default(),
            op,
            deps: SchedDeps::default(),
        }
    }

    pub fn predicated(pred: Predicate, op: Op) -> Self {
        Self {
            pred,
            op,
            deps: SchedDeps::default(),
        }
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pred == Predicate::TRUE {
            write!(f, "{}", self.op)
        } else {
            write!(f, "@{} {}", self.pred, self.op)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Pointer { mutable: bool },
    Scalar(ScalarType),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelParam {
    pub name: String,
    pub kind: ParamKind,
    pub offset: u32,
    pub size: u32,
    pub align: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlock {
    pub label: String,
    pub instrs: Vec<Instr>,
}

impl BasicBlock {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            instrs: Vec::new(),
        }
    }

    pub fn push(&mut self, op: Op) {
        self.instrs.push(Instr::new(op));
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub sm: SmVersion,
    pub local_size: [u16; 3],
    pub params: Vec<KernelParam>,
    pub shared_static_bytes: u32,
    pub blocks: Vec<BasicBlock>,
}

impl Kernel {
    pub fn new(
        name: impl Into<String>,
        sm: SmVersion,
        local_size: [u16; 3],
    ) -> Result<Self, AsmError> {
        let threads = local_size.iter().try_fold(1u32, |acc, dim| {
            acc.checked_mul(u32::from(*dim))
                .ok_or(AsmError::InvalidLocalSize(local_size))
        })?;
        if threads == 0 || threads > 1024 {
            return Err(AsmError::InvalidLocalSize(local_size));
        }

        Ok(Self {
            name: name.into(),
            sm,
            local_size,
            params: Vec::new(),
            shared_static_bytes: 0,
            blocks: Vec::new(),
        })
    }

    pub fn display_ir(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            ".kernel {} sm_{} local_size=({}, {}, {})",
            self.name,
            self.sm.value(),
            self.local_size[0],
            self.local_size[1],
            self.local_size[2]
        )?;
        for param in &self.params {
            writeln!(
                f,
                ".param {} offset={} size={}",
                param.name, param.offset, param.size
            )?;
        }
        for block in &self.blocks {
            writeln!(f, "{}:", block.label)?;
            for instr in &block.instrs {
                writeln!(f, "    {instr}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_predicated_instruction() {
        let instr = Instr::predicated(
            Predicate {
                value: Reg::p(0).into(),
                invert: true,
            },
            Op::Exit,
        );
        assert_eq!(instr.to_string(), "@!p0 exit");
    }

    #[test]
    fn formats_vector_add_shaped_kernel_ir() {
        let mut block = BasicBlock::new("entry");
        block.push(Op::S2R {
            dst: Reg::r(0).into(),
            special: SpecialReg::ThreadIdxX,
        });
        block.push(Op::Load {
            ty: ScalarType::F32,
            dst: Reg::r(1).into(),
            addr: MemRef::new(AddressSpace::Global, Operand::Param("a".to_string()), 0),
        });
        block.push(Op::Load {
            ty: ScalarType::F32,
            dst: Reg::r(2).into(),
            addr: MemRef::new(AddressSpace::Global, Operand::Param("b".to_string()), 0),
        });
        block.push(Op::FAdd {
            dst: Reg::r(3).into(),
            a: Reg::r(1).into(),
            b: Reg::r(2).into(),
        });
        block.push(Op::Store {
            ty: ScalarType::F32,
            addr: MemRef::new(AddressSpace::Global, Operand::Param("out".to_string()), 0),
            value: Reg::r(3).into(),
        });
        block.push(Op::Exit);

        let mut kernel = Kernel::new("vadd", SmVersion::new(80).unwrap(), [128, 1, 1]).unwrap();
        kernel.blocks.push(block);

        let text = kernel.display_ir();
        assert!(text.contains(".kernel vadd sm_80 local_size=(128, 1, 1)"));
        assert!(text.contains("s2r.u32 r0, tid.x"));
        assert!(text.contains("fadd.f32 r3, r1, r2"));
        assert!(text.contains("st.f32 global[param.out], r3"));
    }
}
