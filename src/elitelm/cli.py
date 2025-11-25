import argparse
import sys
import os
import uvicorn
from .session import ChatSession, load_config

def cmd_run(args):
    """Logic to run the interactive chat (formerly chat_cli.py)"""
    config_path = args.config
    config = load_config(config_path)

    verbose = getattr(config, "verbose", False)
    timings_enabled = getattr(config, "timings", False)

    if verbose:
        print(f"Loading model from {config_path}...")
    
    session = ChatSession(config)
    
    if verbose:
        print(f"Model loaded on {session.device_label}")
        print("Ready for input (Ctrl+C to exit).")
        print("-" * 40)

    while True:
        try:
            user_text = input(">>> ")
            if not user_text.strip():
                continue
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        print("Output: ", end="", flush=True)

        def handle_piece(piece: str) -> None:
            print(piece, end="", flush=True)

        result = session.generate(
            user_text,
            timings=timings_enabled,
            on_token=handle_piece,
        )
        print("\n")

def cmd_serve(args):
    """Logic to start the API server"""
    # Pass the config path to the API via environment variable
    os.environ["ELITELM_CONFIG"] = args.config
    print(f"Starting EliteLM server with config: {args.config}")
    uvicorn.run("elitelm.api:app", host=args.host, port=args.port, reload=False)

def main() -> int:
    parser = argparse.ArgumentParser(description="EliteLM Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a model interactively")
    run_parser.add_argument(
        "config", 
        nargs="?", 
        default="llama3-qa.yaml", 
        help="Path to the YAML configuration file"
    )

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "config", 
        nargs="?", 
        default="llama3-qa.yaml", 
        help="Path to the YAML configuration file"
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()

    try:
        if args.command == "run":
            cmd_run(args)
        elif args.command == "serve":
            cmd_serve(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())
