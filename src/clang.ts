
import { mkdtempSync, rmSync, writeFileSync } from "fs"
import { tmpdir } from "os"
import path from "path"
import { dlopen, FFIType, suffix, ptr } from "bun:ffi"
import type { DType } from "./tensor"

export interface RunResult {
    exitCode: number
    stdout: string
    stderr: string
    binaryPath?: string
    compiled: boolean
    output?: Float32Array | Float64Array | Int32Array | Uint8Array
}

export interface Program {
    run(): Promise<RunResult>
}

export class ClangProgram implements Program {
    constructor(
        private source: string,
        private opts?: {
            outputLength?: number
            outputDtype?: DType
        }
    ) {}

    public async run(): Promise<RunResult> {
        const dir = mkdtempSync(path.join(tmpdir(), "clang-prog-"))
        const srcPath = path.join(dir, "main.c")
        const libPath = path.join(dir, `libprog.${suffix}`)

        writeFileSync(srcPath, this.source, "utf8")

        try {
            const compile = Bun.spawn(["clang", "-O2", "-std=c11", "-shared", "-fPIC", srcPath, "-o", libPath], {
                stdout: "pipe",
                stderr: "pipe",
            })
            const compileExit = await compile.exited
            const compileStdout = await new Response(compile.stdout).text()
            const compileStderr = await new Response(compile.stderr).text()

            if (compileExit !== 0) {
                return { exitCode: compileExit, stdout: compileStdout, stderr: compileStderr, compiled: false }
            }

            const ffi = dlopen(libPath, {
                entry_fill: { args: [FFIType.ptr], returns: FFIType.void },
                entry_size: { args: [], returns: FFIType.i32 },
            })

            const size = this.opts?.outputLength ?? ((ffi.symbols as any).entry_size?.() as number)
            const dtype = this.opts?.outputDtype ?? "float32"

            let output: Float32Array | Float64Array | Int32Array | Uint8Array | undefined
            if (size && size > 0) {
                switch (dtype) {
                    case "float64":
                        output = new Float64Array(size)
                        break
                    case "int32":
                        output = new Int32Array(size)
                        break
                    case "bool":
                        output = new Uint8Array(size)
                        break
                    default:
                        output = new Float32Array(size)
                }
                if (output) {
                    const fill = (ffi.symbols as any).entry_fill as (p: number) => void
                    fill(ptr(output))
                }
            }

            if (typeof ffi.close === "function") ffi.close()
            return { exitCode: 0, stdout: "", stderr: "", binaryPath: libPath, compiled: true, output }
        } finally {
            try {
                rmSync(dir, { recursive: true, force: true })
            } catch {
                // ignore cleanup errors
            }
        }
    }
}
