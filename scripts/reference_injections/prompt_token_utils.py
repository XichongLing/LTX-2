#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LTX_CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"

import sys

if str(LTX_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(LTX_CORE_SRC))

from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer  # noqa: E402


@dataclass(frozen=True)
class PromptToken:
    index: int
    token_id: int
    token: str
    text: str
    active: bool


def find_tokenizer_root(gemma_root: str | Path) -> Path:
    root = Path(gemma_root).expanduser().resolve()
    if root.is_file() and root.name == "tokenizer.model":
        return root.parent
    direct = root / "tokenizer.model"
    if direct.is_file():
        return root
    matches = sorted(root.rglob("tokenizer.model"))
    if not matches:
        raise FileNotFoundError(f"Could not find tokenizer.model under {root}")
    return matches[0].parent


def load_ltx_tokenizer(gemma_root: str | Path, max_length: int = 1024) -> LTXVGemmaTokenizer:
    return LTXVGemmaTokenizer(str(find_tokenizer_root(gemma_root)), max_length=max_length)


def inspect_prompt_tokens(prompt: str, gemma_root: str | Path, max_length: int = 1024) -> list[PromptToken]:
    wrapper = load_ltx_tokenizer(gemma_root, max_length=max_length)
    tokenizer = wrapper.tokenizer
    encoded = tokenizer(
        prompt.strip(),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids[0].tolist()
    attention_mask = encoded.attention_mask[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    result: list[PromptToken] = []
    for idx, (token_id, token, active) in enumerate(zip(input_ids, tokens, attention_mask, strict=True)):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        result.append(
            PromptToken(
                index=idx,
                token_id=int(token_id),
                token=str(token),
                text=str(piece),
                active=bool(active),
            )
        )
    return result


def infer_token_indices(
    prompt: str,
    token_text: str,
    gemma_root: str | Path,
    max_length: int = 1024,
) -> list[int]:
    wrapper = load_ltx_tokenizer(gemma_root, max_length=max_length)
    tokenizer = wrapper.tokenizer
    encoded_prompt = tokenizer(
        prompt.strip(),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_ids = encoded_prompt.input_ids[0].tolist()
    active = encoded_prompt.attention_mask[0].tolist()
    active_positions = [i for i, m in enumerate(active) if m]
    active_ids = [prompt_ids[i] for i in active_positions]

    candidates = [token_text, token_text.strip(), " " + token_text.strip()]
    special_ids = set(tokenizer.all_special_ids)
    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue
        candidate_ids = tokenizer(candidate, add_special_tokens=False).input_ids
        candidate_ids = [i for i in candidate_ids if i not in special_ids]
        if not candidate_ids:
            continue
        for start in range(0, len(active_ids) - len(candidate_ids) + 1):
            if active_ids[start : start + len(candidate_ids)] == candidate_ids:
                return active_positions[start : start + len(candidate_ids)]

    # Fallback for subword splits or tokenizer quirks: choose active tokens whose
    # decoded piece contains the requested text after common SentencePiece cleanup.
    target = token_text.strip().lower()
    matches: list[int] = []
    for pos in active_positions:
        piece = tokenizer.decode([prompt_ids[pos]], skip_special_tokens=True).strip().lower()
        token = tokenizer.convert_ids_to_tokens([prompt_ids[pos]])[0].replace("▁", "").strip().lower()
        if target and (target in piece or target in token):
            matches.append(pos)
    if matches:
        return matches

    raise ValueError(
        f"Could not infer token indices for {token_text!r}. Run prompt_token_utils.py to inspect tokenization."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LTX/Gemma padded prompt token indices.")
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--token-text", default=None, help="Optional word/phrase to locate, e.g. cat.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tokens = inspect_prompt_tokens(args.prompt, args.gemma_root, max_length=args.max_length)
    active_tokens = [t for t in tokens if t.active]
    payload = {
        "prompt": args.prompt,
        "active_token_count": len(active_tokens),
        "tokens": [asdict(t) for t in active_tokens],
    }
    if args.token_text is not None:
        payload["matched_indices"] = infer_token_indices(
            args.prompt,
            args.token_text,
            args.gemma_root,
            max_length=args.max_length,
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Prompt: {args.prompt}")
    print(f"Active tokens: {len(active_tokens)}")
    for item in active_tokens:
        print(f"{item.index:4d}  id={item.token_id:<8d} token={item.token!r:<18} text={item.text!r}")
    if args.token_text is not None:
        print(f"Matched indices for {args.token_text!r}: {payload['matched_indices']}")


if __name__ == "__main__":
    main()
