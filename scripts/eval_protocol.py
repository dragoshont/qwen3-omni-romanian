"""Pure helpers shared by Romanian speech evaluation and report validation."""

from __future__ import annotations

import hashlib
import re
import unicodedata


# Frozen by eval/T4_SEEDED_PROTOCOL.md before issue #1 was rerun.  Reuse the
# same namespace for later paired evaluations so prompt seeds remain comparable.
PROMPT_SEED_NAMESPACE = "t4-seeded-v2"


def prompt_seed(base_seed: int, prompt_id: str) -> int:
    """Return a stable 63-bit seed independent of prompt evaluation order."""

    payload = f"{PROMPT_SEED_NAMESPACE}:{base_seed}:{prompt_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def normalize_romanian_asr(text: str) -> str:
    """Normalize orthographic formatting without erasing Romanian diacritics."""

    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("ş", "ș").replace("ţ", "ț")
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def detect_repetition(text: str) -> bool:
    """Conservative ASR-text loop heuristic; not an acoustic-quality label."""

    words = normalize_romanian_asr(text).split()
    if len(words) >= 6:
        for width in (1, 2, 3):
            ngrams = [" ".join(words[index : index + width]) for index in range(len(words) - width + 1)]
            for index in range(2 * width, len(ngrams)):
                if ngrams[index] == ngrams[index - width] == ngrams[index - 2 * width]:
                    return True
    normalized = " ".join(words)
    if re.search(r"(-[a-zăâîșț]){5,}", normalized):
        return True
    return len(words) >= 15 and len(set(words)) / len(words) < 0.35
