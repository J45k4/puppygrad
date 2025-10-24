mod codegen;
mod graph_ir;
mod parser;

use std::{env, fs, io::ErrorKind, path::Path, process::Command};

use tempfile::tempdir;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = match args.next() {
        Some(command) => command,
        None => return Err(usage()),
    };

    match command.as_str() {
        "emit" => handle_emit(args.collect()),
        "run" => handle_run(args.collect()),
        other => Err(format!("unknown command '{}'\n{}", other, usage())),
    }
}

fn handle_emit(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() {
        return Err("emit command requires a target language".into());
    }

    let mut args_iter = args.into_iter();
    let target = args_iter.next().unwrap();

    match target.as_str() {
        "c" => emit_c(args_iter.collect()),
        other => Err(format!("unsupported emit target '{}'", other)),
    }
}

fn emit_c(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "emit c requires a PROGRAM argument containing a file path or inline source".into(),
        );
    }

    let program_arg = args.join(" ");
    let c_source = compile_program_to_c(&program_arg)?;
    println!("{}", c_source);
    Ok(())
}

fn handle_run(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() {
        return Err(
            "run command requires a PROGRAM argument containing a file path or inline source"
                .into(),
        );
    }

    let program_arg = args.join(" ");
    let c_source = compile_program_to_c(&program_arg)?;

    let temp_dir =
        tempdir().map_err(|err| format!("failed to create temporary directory: {}", err))?;
    let c_path = temp_dir.path().join("program.c");
    fs::write(&c_path, c_source).map_err(|err| {
        format!(
            "failed to write C source to '{}': {}",
            c_path.display(),
            err
        )
    })?;

    let exe_name = if cfg!(windows) {
        "program.exe"
    } else {
        "program"
    };
    let exe_path = temp_dir.path().join(exe_name);
    compile_c_source(&c_path, &exe_path)?;

    let status = Command::new(&exe_path)
        .status()
        .map_err(|err| format!("failed to run compiled program: {}", err))?;

    if !status.success() {
        return Err(format!("program exited with status {}", status));
    }

    Ok(())
}

fn compile_c_source(c_path: &Path, exe_path: &Path) -> Result<(), String> {
    let compilers = ["cc", "gcc", "clang"];

    for compiler in &compilers {
        match Command::new(compiler)
            .arg("-std=c11")
            .arg(c_path)
            .arg("-O2")
            .arg("-o")
            .arg(exe_path)
            .arg("-lm")
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    return Ok(());
                }

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "C compiler '{}' failed with status {}.\nstdout:\n{}\nstderr:\n{}",
                    compiler, output.status, stdout, stderr
                ));
            }
            Err(err) if err.kind() == ErrorKind::NotFound => {
                continue;
            }
            Err(err) => {
                return Err(format!(
                    "failed to invoke C compiler '{}': {}",
                    compiler, err
                ));
            }
        }
    }

    Err(format!(
        "no suitable C compiler found. Tried: {}",
        compilers.join(", ")
    ))
}

fn compile_program_to_c(program_arg: &str) -> Result<String, String> {
    let source = load_program_source(program_arg)?;
    let program = parser::parse(source).map_err(|err| err.to_string())?;
    let module = graph_ir::lower_program(&program);
    Ok(codegen::generate_c_code(&module))
}

fn load_program_source(arg: &str) -> Result<String, String> {
    let path = Path::new(arg);
    if path.exists() {
        if path.is_file() {
            fs::read_to_string(path)
                .map_err(|err| format!("failed to read program file '{}': {}", arg, err))
        } else {
            Err(format!("program path '{}' is not a file", arg))
        }
    } else {
        Ok(arg.to_string())
    }
}

fn usage() -> String {
    "Usage:\n  puppygrad emit c PROGRAM\n  puppygrad run PROGRAM\n\nPROGRAM may be a path to a source file or the source code itself.".to_string()
}
