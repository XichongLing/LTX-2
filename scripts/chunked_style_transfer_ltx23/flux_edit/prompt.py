#!/usr/bin/env python3
"""Builds the index-aware two-reference prompt for the FLUX.2 edit (Section 7)."""
from __future__ import annotations

MULTIREF_PREAMBLE = (
    "Image 1 is the target style/appearance reference. Image 2 is the scene to restyle. "
    "Re-render the scene in image 2 with the materials, colors, and lighting of image 1, "
    "preserving the layout and geometry of image 2."
)


def build_multiref_prompt(user_prompt: str, negative_prompt: str = "") -> str:
    """Combine the fixed role-disambiguation preamble with the user's restyle prompt."""
    user_prompt = user_prompt.strip()
    text = f"{MULTIREF_PREAMBLE} {user_prompt}".strip() if user_prompt else MULTIREF_PREAMBLE

    negative_prompt = negative_prompt.strip()
    if negative_prompt:
        text = f"{text}\n\nNegative prompt: {negative_prompt}"
    return text
