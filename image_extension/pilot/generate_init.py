"""Initialization pilot: N synthetic images per group, no voting.

Groups (survey step 3):
  random    text-only generic prompt
  meta      text-only prompt conditioned on one DP metadata row
  ref       k donated reference images (random), no metadata
  meta_ref  DP metadata + k Hamming-nearest donated (metadata, image) examples (MAPLE)

All groups share the Qwen-Image-Edit-2511 weights. Text-only groups run them through
QwenImagePipeline because QwenImageEditPlusPipeline requires at least one input image.
Shard across GPUs with --rank/--world; existing outputs are skipped (resumable).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pilot_common import QWEN_EDIT, SCHEMA, WORK, describe, hamming

BASE = ('A realistic, high-quality close-up portrait photograph of a single person, '
        'face centered and looking roughly at the camera, head and shoulders visible.')
REF_INSTRUCTION = ('The pictures above are portrait photos of {k} different real people. '
                   'Create one new portrait photograph of a completely different person who does not '
                   'appear in any of the pictures, matching the general photographic style of the '
                   'examples (framing, lighting, image quality). Do not copy any face from the pictures.')
COLLAGE_INSTRUCTION = ('The picture above is a reference collage of {k} portrait photos of {k} different real '
                       'people, numbered left to right, top to bottom. Do not reproduce the collage layout. '
                       'Create one single new portrait photograph of one completely different person who does '
                       'not appear in the collage, matching the general photographic style of the examples '
                       '(framing, lighting, image quality). Do not copy any face from the collage.')


def make_collage(paths, tile=256, cols=5):
    rows = -(-len(paths) // cols)
    cols = min(cols, len(paths))
    canvas = Image.new('RGB', (cols * tile, rows * tile), 'white')
    for j, p in enumerate(paths):
        with Image.open(p) as im:
            canvas.paste(im.convert('RGB').resize((tile, tile), Image.LANCZOS), ((j % cols) * tile, (j // cols) * tile))
    return canvas


def build_request(group, i, meta_row, donors, k, seed, ref_mode='multi'):
    rng = np.random.default_rng(seed)
    refs = []
    instruction = COLLAGE_INSTRUCTION if ref_mode == 'collage' else REF_INSTRUCTION
    label = 'Photo' if ref_mode == 'collage' else 'Picture'
    if group == 'random':
        prompt = BASE
    elif group == 'meta':
        prompt = f'{BASE} The person is {describe(meta_row)}.'
    elif group == 'ref':
        refs = rng.choice(len(donors), size=k, replace=False).tolist()
        prompt = instruction.format(k=k) + ' ' + BASE
    elif group == 'meta_ref':
        # Hamming-nearest donated examples; random tie-break so ties do not always pick the same donors.
        d = np.array([hamming(meta_row, r) for r in donors]) + rng.random(len(donors)) * 0.5
        refs = np.argsort(d)[:k].tolist()
        ex = ' '.join(f'{label} {j + 1} shows {describe(donors[r])}.' for j, r in enumerate(refs))
        prompt = (instruction.format(k=k) + ' ' + ex + ' ' + BASE +
                  f' The new person is {describe(meta_row)}.')
    else:
        raise ValueError(group)
    return prompt, refs


def load_pipelines(need_edit, need_t2i, ref_area):
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as edit_mod
    from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline
    edit_mod.VAE_IMAGE_SIZE = ref_area
    edit = QwenImageEditPlusPipeline.from_pretrained(QWEN_EDIT, torch_dtype=torch.bfloat16).to('cuda')
    edit.set_progress_bar_config(disable=True)
    t2i = None
    if need_t2i:
        t2i = QwenImagePipeline(scheduler=edit.scheduler, vae=edit.vae, text_encoder=edit.text_encoder,
                                tokenizer=edit.tokenizer, transformer=edit.transformer)
        t2i.set_progress_bar_config(disable=True)
    return (edit if need_edit else None), t2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='random,meta,ref,meta_ref')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--n-refs', type=int, default=10)
    ap.add_argument('--ref-mode', choices=['multi', 'collage'], default='multi',
                    help='multi: k separate input images; collage: k donors tiled into one input image')
    ap.add_argument('--ref-area', type=int, default=512 * 512, help='VAE pixel area per reference image')
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--metadata', default=str(WORK / 'dp_metadata_eps1.csv'))
    ap.add_argument('--run', default='init_pilot')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    groups = args.groups.split(',')
    meta = pd.read_csv(args.metadata)[list(SCHEMA)].astype(str).to_dict('records')
    if len(meta) < args.n:
        raise ValueError(f'Need {args.n} metadata rows, got {len(meta)}')
    donated = pd.read_csv(WORK / 'donated.csv')
    donors = donated[list(SCHEMA)].astype(str).to_dict('records')
    donor_paths = donated['image_full'].tolist()

    edit, t2i = load_pipelines(any(g in ('ref', 'meta_ref') for g in groups),
                               any(g in ('random', 'meta') for g in groups), args.ref_area)
    run_dir = WORK / 'runs' / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2))

    for g in groups:
        out_dir = run_dir / g
        out_dir.mkdir(exist_ok=True)
        log = (out_dir / f'log_rank{args.rank}.jsonl').open('a')
        for i in range(args.rank, args.n, args.world):
            path = out_dir / f'{i:05d}.png'
            if path.exists():
                continue
            seed = args.seed + i  # same seed per index across groups
            prompt, refs = build_request(g, i, meta[i], donors, args.n_refs, seed, args.ref_mode)
            common = dict(prompt=prompt, negative_prompt=' ', true_cfg_scale=args.cfg,
                          num_inference_steps=args.steps, height=args.size, width=args.size,
                          generator=torch.Generator('cuda').manual_seed(seed))
            t0 = time.perf_counter()
            with torch.inference_mode():
                if refs:
                    if args.ref_mode == 'collage':
                        images = [make_collage([donor_paths[r] for r in refs])]
                    else:
                        images = [Image.open(donor_paths[r]).convert('RGB') for r in refs]
                    img = edit(image=images, guidance_scale=1.0, **common).images[0]
                else:
                    img = t2i(**common).images[0]
            img.save(path)
            log.write(json.dumps({'i': i, 'seed': seed, 'prompt': prompt,
                                  'refs': [donated['file'].iloc[r] for r in refs],
                                  'metadata': meta[i] if g in ('meta', 'meta_ref') else None,
                                  'seconds': round(time.perf_counter() - t0, 2)}) + '\n')
            log.flush()
        log.close()


if __name__ == '__main__':
    main()
