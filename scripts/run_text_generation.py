#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize text requests, run the C++ generator, and decode responses."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Text file with one request per line")
    parser.add_argument("--output", type=Path, default=Path("./data/responses.txt"))
    parser.add_argument("--prompt", help="Single request without an input file")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--main-binary", type=Path, default=Path("./main"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def load_requests(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [args.prompt]
    if args.input is None:
        raise ValueError("either --input or --prompt is required")
    return args.input.read_text(encoding="utf-8").splitlines()


def load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for text generation wrapper. "
            "Install: pip install transformers tokenizers sentencepiece"
        ) from exc

    return AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=True,
    )


def encode_request(tokenizer, request: str, system_prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    text = request if not system_prompt else f"{system_prompt}\n\n{request}"
    payload = tokenizer(text, add_special_tokens=False)
    return list(payload["input_ids"])


def write_token_batch(path: Path, sequences: list[list[int]], pad_token_id: int) -> None:
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    lengths = [len(seq) for seq in sequences]

    with path.open("wb") as f:
        f.write(struct.pack("<ii", batch_size, max_len))
        f.write(struct.pack(f"<{batch_size}i", *lengths))
        for seq in sequences:
            padded = seq + [pad_token_id] * (max_len - len(seq))
            f.write(struct.pack(f"<{max_len}i", *padded))


def read_token_batch(path: Path) -> list[list[int]]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"invalid token batch file: {path}")

    offset = 0
    batch_size, max_len = struct.unpack_from("<ii", data, offset)
    offset += 8
    lengths = list(struct.unpack_from(f"<{batch_size}i", data, offset))
    offset += 4 * batch_size

    sequences: list[list[int]] = []
    for b in range(batch_size):
        row = list(struct.unpack_from(f"<{max_len}i", data, offset))
        offset += 4 * max_len
        sequences.append(row[: lengths[b]])
    return sequences


def decode_responses(tokenizer, sequences: list[list[int]]) -> list[str]:
    return [
        tokenizer.decode(tokens, skip_special_tokens=True).rstrip("\r\n")
        for tokens in sequences
    ]


def write_responses(path: Path, responses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, response in enumerate(responses, start=1):
            f.write(f"===== RESPONSE {idx} =====\n")
            f.write(response)
            f.write("\n")
            if idx != len(responses):
                f.write("\n")


def run_main(args: argparse.Namespace, prompt_tokens: Path, output_tokens: Path) -> None:
    cmd = [
        str(args.main_binary),
        "-m",
        str(args.model_dir),
        "--token-input",
        str(prompt_tokens),
        "--token-output",
        str(output_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.context_len > 0:
        cmd.extend(["--context-len", str(args.context_len)])
    if args.validate:
        cmd.append("--validate")
    if args.warmup:
        cmd.append("--warmup")

    subprocess.run(cmd, check=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    requests = load_requests(args)
    tokenizer = load_tokenizer(args.model_dir)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer is missing pad/eos/bos token ids")

    sequences = [encode_request(tokenizer, request, args.system) for request in requests]
    if not sequences:
        raise ValueError("no requests were provided")
    if any(len(seq) == 0 for seq in sequences):
        raise ValueError("tokenized request must not be empty")

    with tempfile.TemporaryDirectory(prefix="llama_practice_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        prompt_tokens = tmpdir_path / "prompt_tokens.bin"
        output_tokens = tmpdir_path / "generated_tokens.bin"
        write_token_batch(prompt_tokens, sequences, pad_token_id)
        run_main(args, prompt_tokens, output_tokens)
        responses = decode_responses(tokenizer, read_token_batch(output_tokens))
        write_responses(args.output, responses)

        if args.keep_temp:
            keep_dir = args.output.parent / f"{args.output.stem}.tmp"
            keep_dir.mkdir(parents=True, exist_ok=True)
            keep_prompt = keep_dir / "prompt_tokens.bin"
            keep_output = keep_dir / "generated_tokens.bin"
            keep_prompt.write_bytes(prompt_tokens.read_bytes())
            keep_output.write_bytes(output_tokens.read_bytes())

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_text_generation.py error: {exc}", file=sys.stderr)
        raise
