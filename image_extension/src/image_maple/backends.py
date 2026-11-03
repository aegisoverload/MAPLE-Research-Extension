"""Mock image generator for end-to-end CI tests. NOT a scientific image model."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import hashlib
import random

from .contracts import GenerationRequest


class MockImageBackend:
    def generate(self, request: GenerationRequest) -> Path:
        rng = random.Random(request.seed)
        text = '|'.join(f'{k}={v}' for k, v in sorted(request.metadata.items()))
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        color = tuple(70 + (digest[i] % 140) for i in range(3))
        canvas = Image.new('RGB', (256, 256), color)
        draw = ImageDraw.Draw(canvas)
        for _ in range(12):
            x, y = rng.randrange(0, 250), rng.randrange(0, 250)
            r = rng.randrange(10, 70)
            fill = tuple(rng.randrange(20, 236) for _ in range(3))
            draw.ellipse((x, y, x + r, y + r), outline=fill, width=4)
        if request.mode == 'variation':
            if request.source_image is None:
                raise ValueError('variation requires a previous SYNTHETIC image')
            with Image.open(request.source_image) as prior:
                prior = prior.convert('RGB').resize((256, 256))
                canvas = Image.blend(prior, canvas, max(0.05, min(0.9, request.variation_strength)))
        canvas = canvas.filter(ImageFilter.GaussianBlur(0.4))
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(request.output_path)
        return request.output_path
