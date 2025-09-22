import argparse
import sys

from elitelm import ChatSession, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive EliteLM chat client",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="llama3-qa.yaml",
        help="Path to the YAML configuration file. Defaults to llama3-qa.yaml in the repo root.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="prompts",
        action="append",
        help=(
            "Prompt text to run the chat client non-interactively. "
            "Provide multiple --prompt arguments to simulate a dialogue."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a file containing newline-delimited prompts for non-interactive runs.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read a single prompt from standard input for a non-interactive run.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    config = load_config(cli_args.config)

    verbose = getattr(config, "verbose", False)
    timings_enabled = getattr(config, "timings", False)

    if verbose:
        print("Loading model...")
    session = ChatSession(config)
    if verbose:
        print(f"Model loaded on {session.device_label}")
        print("Tokenizer created")
        print()

    prompts: list[str] = []

    if cli_args.prompts:
        prompts.extend(text for text in cli_args.prompts if text is not None)

    if cli_args.prompt_file:
        try:
            with open(cli_args.prompt_file, "r", encoding="utf-8") as prompt_file:
                prompts.extend(
                    line.rstrip("\r\n")
                    for line in prompt_file
                    if line.rstrip("\r\n")
                )
        except OSError as exc:
            print(f"Failed to read prompts from {cli_args.prompt_file}: {exc}", file=sys.stderr)
            sys.exit(1)

    if cli_args.stdin:
        stdin_text = sys.stdin.read()
        if stdin_text:
            prompts.append(stdin_text.rstrip("\r\n"))

    def run_prompt(user_text: str, *, echo_input: bool) -> None:
        if not user_text:
            print("Error, input cannot be empty")
            return

        if echo_input:
            print(f"Input: {user_text}")

        if verbose:
            print("Generator created")
            print("Running generation loop ...")

        print()
        print("Output: ", end="", flush=True)

        def handle_piece(piece: str) -> None:
            print(piece, end="", flush=True)

        result = session.generate(
            user_text,
            timings=timings_enabled,
            on_token=handle_piece,
        )

        print()
        print()

        if result.interrupted:
            print("  --control+c pressed, aborting generation--")
            print()

        if timings_enabled and result.stats:
            stats = result.stats
            print(
                "Prompt length: {prompt_len}, New tokens: {new_tokens}, Time to first: {ttf:.2f}s, "
                "Prompt tokens per second: {prompt_tps:.2f} tps, New tokens per second: {gen_tps:.2f} tps".format(
                    prompt_len=stats.prompt_length,
                    new_tokens=stats.new_tokens,
                    ttf=stats.time_to_first_token,
                    prompt_tps=stats.prompt_tokens_per_second,
                    gen_tps=stats.generated_tokens_per_second,
                )
            )

    if prompts:
        for prompt in prompts:
            run_prompt(prompt, echo_input=True)
        return

    while True:
        try:
            user_text = input("Input: ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        run_prompt(user_text, echo_input=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(0)