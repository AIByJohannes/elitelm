import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

_DLL_HANDLES: list[object] = []
_REGISTERED_DLL_PATHS: set[str] = set()


def _add_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    path = path.resolve()
    path_str = str(path)
    if path_str in _REGISTERED_DLL_PATHS:
        return
    _DLL_HANDLES.append(os.add_dll_directory(path_str))
    current_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{path_str};{current_path}" if current_path else path_str
    _REGISTERED_DLL_PATHS.add(path_str)


def _resolve_qnn_sdk_root(sdk_arg: str | None) -> Path:
    if sdk_arg:
        candidate = Path(sdk_arg).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Provided QNN SDK path does not exist: {candidate}")
        return candidate

    env_sdk = os.environ.get("QNN_SDK_ROOT")
    if env_sdk:
        candidate = Path(env_sdk)
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parent
    qairt_root = repo_root / "qairt"
    if qairt_root.exists():
        versions = sorted((p for p in qairt_root.iterdir() if p.is_dir()), reverse=True)
        if versions:
            return versions[0].resolve()

    raise FileNotFoundError(
        "Unable to locate the QNN SDK. Provide --qnn-sdk or set QNN_SDK_ROOT."
    )


def _default_backend_path(sdk_root: Path) -> tuple[Path, str]:
    candidate_arches = [
        "aarch64-windows-msvc",
        "arm64x-windows-msvc",
        "x86_64-windows-msvc",
    ]
    for arch in candidate_arches:
        candidate = sdk_root / "lib" / arch / "QnnHtp.dll"
        if candidate.exists():
            return candidate, arch
    raise FileNotFoundError("Unable to locate QnnHtp.dll in the QNN SDK lib directory")


def _configure_qnn_provider(model_dir: Path, sdk_arg: str | None, backend_arg: str | None) -> og.Config:
    if not og.is_qnn_available():
        raise RuntimeError("onnxruntime-genai was built without QNN support on this platform")

    config = og.Config(str(model_dir))
    config.clear_providers()
    config.append_provider("QNNExecutionProvider")

    sdk_root = _resolve_qnn_sdk_root(sdk_arg)
    if backend_arg:
        backend_path = Path(backend_arg).expanduser()
        arch = backend_path.parent.name
    else:
        backend_path, arch = _default_backend_path(sdk_root)

    if not backend_path.exists():
        raise FileNotFoundError(f"QNN backend DLL not found at: {backend_path}")

    os.environ.setdefault("QNN_SDK_ROOT", str(sdk_root))
    _add_dll_dir(backend_path.parent)
    _add_dll_dir(sdk_root / "lib" / arch)
    _add_dll_dir(sdk_root / "bin" / arch)

    config.set_provider_option("QNNExecutionProvider", "backend_path", backend_path.name)
    config.set_provider_option("QNNExecutionProvider", "qnn_sdk_root", str(sdk_root))
    return config


def _load_model(args) -> tuple[og.Model, og.Tokenizer, og.TokenizerStream]:
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.device == "qnn":
        config = _configure_qnn_provider(
            model_dir,
            getattr(args, "qnn_sdk", None),
            getattr(args, "qnn_backend", None),
        )
        model = og.Model(config)
    else:
        model = og.Model(str(model_dir))

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    return model, tokenizer, tokenizer_stream


def main(args):
    if args.verbose:
        print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    model, tokenizer, tokenizer_stream = _load_model(args)
    if args.verbose:
        device_info = "QNN" if args.device == "qnn" else "CPU"
        print(f"Model loaded on {device_info}")
        print("Tokenizer created")
        print()

    search_options = {
        name: getattr(args, name)
        for name in [
            "do_sample",
            "max_length",
            "min_length",
            "top_p",
            "top_k",
            "temperature",
            "repetition_penalty",
        ]
        if hasattr(args, name)
    }

    if "max_length" not in search_options:
        search_options["max_length"] = 2048

    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if args.timings:
            started_timestamp = time.time()

        prompt = f'{chat_template.format(input=text)}'
        input_tokens = tokenizer.encode(prompt).astype(np.int32)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        if args.verbose:
            print("Generator created")
            print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print()
        print("Output: ", end="", flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings and first:
                    first_token_timestamp = time.time()
                    first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end="", flush=True)
                if args.timings:
                    new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        del generator

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print(
                "Prompt length: {prompt_len}, New tokens: {new_tokens}, Time to first: {ttf:.2f}s, "
                "Prompt tokens per second: {prompt_tps:.2f} tps, New tokens per second: {gen_tps:.2f} tps".format(
                    prompt_len=len(input_tokens),
                    new_tokens=len(new_tokens),
                    ttf=prompt_time,
                    prompt_tps=len(input_tokens) / prompt_time if prompt_time else 0.0,
                    gen_tps=len(new_tokens) / run_time if run_time else 0.0,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        description="End-to-end AI Question/Answer example for gen-ai",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Onnx model folder path (must contain config.json and model.onnx)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "qnn"],
        default="cpu",
        help="Select inference device. Use 'qnn' to run on the Hexagon NPU.",
    )
    parser.add_argument(
        "--qnn-sdk",
        type=str,
        help="Path to the QNN SDK root. Defaults to QNN_SDK_ROOT or ./qairt/<version>",
    )
    parser.add_argument(
        "--qnn-backend",
        type=str,
        help="Path to QnnHtp.dll. Defaults to the best match in <qnn-sdk>/lib",
    )
    parser.add_argument(
        "-i",
        "--min_length",
        type=int,
        help="Min number of tokens to generate including the prompt",
    )
    parser.add_argument(
        "-l",
        "--max_length",
        type=int,
        help="Max number of tokens to generate including the prompt",
    )
    parser.add_argument(
        "-ds",
        "--do_sample",
        action="store_true",
        default=False,
        help=(
            "Do random sampling. When false, greedy or beam search are used to generate the "
            "output. Defaults to false"
        ),
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=float,
        help="Top p probability to sample with",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        help="Top k tokens to sample from",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="Temperature to sample with",
    )
    parser.add_argument(
        "-r",
        "--repetition_penalty",
        type=float,
        help="Repetition penalty to sample with",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose output and timing information. Defaults to false",
    )
    parser.add_argument(
        "-g",
        "--timings",
        action="store_true",
        default=False,
        help="Print timing information for each generation step. Defaults to false",
    )
    main(parser.parse_args())