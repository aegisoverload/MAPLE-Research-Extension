"""Team member: edit ONLY this file to integrate any image API or local model.

Inputs to generate(request):
  request.mode             'random' or 'variation'
  request.metadata         DP-derived (provenance must be independently verified)
  request.examples         PUBLIC/DONATED examples; never private training images
  request.source_image     previous SYNTHETIC output for variation (or None)
  request.output_path      exact local PNG output location
  request.seed             non-DP generation randomness; NOT DP Gaussian seed
Output:
  write PNG at request.output_path and return that Path.

DO NOT log/upload private training images or access the private manifest from here.
"""
from pathlib import Path
from image_maple.contracts import GenerationRequest


class TeammateBackend:
    def __init__(self):
        # Read YOUR_API_KEY from environment variables here, not hardcoded strings.
        pass

    def generate(self, request: GenerationRequest) -> Path:
        # TODO: Build the provider-specific request with metadata and PUBLIC examples.
        # TODO: If request.mode == 'variation', include request.source_image (synthetic).
        # TODO: Make provider call, validate response, save image to output_path.
        raise NotImplementedError(
            'Implement TeammateBackend.generate(request) in providers/teammate_backend.py'
        )
