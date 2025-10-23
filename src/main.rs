mod codegen;
mod graph_ir;
mod parser;

use std::{env, fs, path::Path};

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
    let source = load_program_source(&program_arg)?;
    let program = parser::parse(source).map_err(|err| err.to_string())?;
    let module = graph_ir::lower_program(&program);
    let c_source = codegen::generate_c_code(&module);
    println!("{}", c_source);
    Ok(())
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
    "Usage: puppygrad emit c PROGRAM\n\nPROGRAM may be a path to a source file or the source code itself.".to_string()
}
