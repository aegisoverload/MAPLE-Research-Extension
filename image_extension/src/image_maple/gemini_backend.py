"""Optional real image API example. Only public/donated images and generated images are passed.

Requires: pip install -e '.[gemini]' ; GOOGLE_API_KEY in the OS environment.
Google can change model ids, reference limits, and output shapes; verify before paid runs.
"""
import io
import os
from pathlib import Path
from PIL import Image
from .contracts import GenerationRequest


class GeminiBackend:
    def __init__(self, model='gemini-3.1-flash-lite-image'):
        from google import genai
        key = os.environ.get('GOOGLE_API_KEY', '')
        if not key or key == 'REPLACE_ME_LOCALLY':
            raise EnvironmentError('Set GOOGLE_API_KEY in your local environment; NEVER commit it')
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, request: GenerationRequest) -> Path:
        from google.genai import types
        # Input policy: examples are PUBLIC/DONATED; variation parent is PREVIOUSLY GENERATED.
        text = 'Generate ONE new, photorealistic synthetic image for a DP image-generation research experiment.\n'
        text += 'Requested categorical metadata: ' + ', '.join(
            f'{k}={v}' for k, v in sorted(request.metadata.items())
        ) + '\n'
        text += 'Follow the attributes and visual concept, but do not copy faces or source photos.'
        if request.mode == 'variation':
            text += f'\nMake a DISTINCT new image related to the synthetic source, variation_strength={request.variation_strength:.2f}.'
        content = [text]
        imgs = []
        try:
            if request.mode == 'variation':
                if request.source_image is None:
                    raise ValueError('Missing previous synthetic image')
                source = Image.open(request.source_image)
                imgs.append(source.convert('RGB'))
                source.close()
                content.append(imgs[-1])
            for public in request.examples:
                img = Image.open(public.image_path)
                imgs.append(img.convert('RGB'))
                img.close()
                content.append(imgs[-1])
            response = self.client.models.generate_content(
                model=self.model,
                contents=content,
                config=types.GenerateContentConfig(response_modalities=['IMAGE']),
            )
            parts = []
            for c in (getattr(response, 'candidates', None) or []):
                if getattr(c, 'content', None):
                    parts.extend(c.content.parts or [])
            for part in parts:
                data = getattr(part, 'inline_data', None)
                if data is not None and getattr(data, 'data', None):
                    # SDK returns decoded bytes in inline_data.data.
                    image = Image.open(io.BytesIO(data.data))
                    request.output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.convert('RGB').save(request.output_path, format='PNG')
                    image.close()
                    return request.output_path
            raise RuntimeError('Image API returned no image (possibly blocked/refused); record and retry per policy')
        finally:
            for img in imgs:
                img.close()
