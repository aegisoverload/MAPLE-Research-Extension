"""Stable handoff contract: only public/donated examples and DP-derived metadata go to API."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class PublicExample:
    image_path: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class GenerationRequest:
    mode: Literal["random", "variation"]
    metadata: dict[str, str]
    examples: tuple[PublicExample, ...]
    output_path: Path
    seed: int
    source_image: Path | None = None  # previous synthetic (DP-derived) output ONLY
    variation_strength: float = 0.5


class ImageBackend(Protocol):
    def generate(self, request: GenerationRequest) -> Path:
        """Produce one image at request.output_path; return the saved image path."""
        ...
