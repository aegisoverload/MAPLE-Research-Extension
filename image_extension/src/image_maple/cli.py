"""CLI entrypoint. Use --backend mock first, then API teammate supplies a backend."""
import argparse
import importlib
import json
from pathlib import Path

from .backends import MockImageBackend
from .embeddings import LocalCLIPEmbedder, PixelSmokeEmbedder
from .engine import run_pipeline
from .io import load_image_manifest, load_metadata_csv


def make_backend(args):
    if args.backend == 'mock':
        return MockImageBackend()
    if args.backend == 'gemini':
        from .gemini_backend import GeminiBackend
        return GeminiBackend(model=args.image_model)
    if args.backend == 'custom':
        if ':' not in args.custom_backend:
            raise ValueError('Use --custom-backend module:ClassName')
        module, classname = args.custom_backend.rsplit(':', 1)
        return getattr(importlib.import_module(module), classname)()
    raise ValueError('Unknown backend')


def main(argv=None):
    p = argparse.ArgumentParser(description='Image MAPLE MVP: NOT CERTIFIED DP')
    p.add_argument('--metadata', required=True, help='DP-derived categorical metadata CSV')
    p.add_argument('--private-manifest', required=True, help='PRIVATE images, local embedding ONLY')
    p.add_argument('--donated-manifest', required=True, help='PUBLIC/DONATED images allowed for API')
    p.add_argument('--out', default='runs/demo')
    p.add_argument('--backend', choices=['mock', 'gemini', 'custom'], default='mock')
    p.add_argument('--custom-backend', default='image_maple.providers.teammate_backend:TeammateBackend')
    p.add_argument('--image-model', default='gemini-3.1-flash-lite-image')
    p.add_argument('--embedding', choices=['pixel-smoke', 'clip'], default='pixel-smoke')
    p.add_argument('--clip-model', default='openai/clip-vit-base-patch32')
    p.add_argument('--num-syn', type=int, default=4)
    p.add_argument('--lookahead', type=int, default=1)
    p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--n-examples', type=int, default=2)
    p.add_argument('--sigma', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow-paid-api', action='store_true',
                   help='Explicit permission to make external API calls; no cost estimate is implied')
    p.add_argument('--max-paid-calls', type=int, default=200,
                   help='Hard preflight cap for expected API calls (default 200)')
    p.add_argument('--metadata-provenance', default='', help='JSON with AIM provenance (not a certificate)')
    a = p.parse_args(argv)
    expected_calls = a.num_syn * (a.lookahead + 1 + a.lookahead * a.iterations)
    if a.backend != 'mock':
        if not a.allow_paid_api:
            p.error('External API backend needs --allow-paid-api; do not run without permission and budget')
        if expected_calls > a.max_paid_calls:
            p.error(f'Expected {expected_calls} API calls exceed --max-paid-calls {a.max_paid_calls}')
        print(f'API REQUEST BUDGET: planned {expected_calls} calls (not an exact cost estimate)')
    metadata_rows = load_metadata_csv(a.metadata)
    private = load_image_manifest(a.private_manifest)
    donated = load_image_manifest(a.donated_manifest, donated=True)
    provenance = None
    if a.metadata_provenance:
        provenance = json.loads(Path(a.metadata_provenance).read_text(encoding='utf-8'))
    backend = make_backend(a)
    embedder = PixelSmokeEmbedder() if a.embedding == 'pixel-smoke' else LocalCLIPEmbedder(model=a.clip_model)
    report = run_pipeline(backend=backend, embedder=embedder, metadata_rows=metadata_rows,
                          private_images=private, donated_examples=donated,
                          output_dir=a.out, num_syn=a.num_syn, lookahead=a.lookahead,
                          iterations=a.iterations, sigma=a.sigma, seed=a.seed,
                          n_examples=a.n_examples, metadata_provenance=provenance)
    print('COMPLETE (experimental, NOT CERTIFIED DP)')
    print(f'Output: {Path(a.out).resolve()}')
    print(f'Rounds: {report["voting_rounds"]}; sigma: {report["sigma_used"]}')


if __name__ == '__main__':
    main()
