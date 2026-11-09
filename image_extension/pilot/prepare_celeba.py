"""Build the pilot dataset from CelebAMask-HQ.

- Person-level unit: keep one image per CelebA identity (identity_CelebA.txt).
- Donated pool: 50 images from identities disjoint from the private set (MAPLE uses 50 donated).
- Private set: all remaining one-per-identity images; resized to 256x256 for FID reference.
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

from pilot_common import CELEBA_DIR, WORK, attrs_to_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-donated', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = CELEBA_DIR / 'CelebAMask-HQ'
    attr = pd.read_csv(root / 'CelebAMask-HQ-attribute-anno.txt', sep=r'\s+', skiprows=1)
    attr.index.name = 'file'
    attr = attr.reset_index()
    mapping = pd.read_csv(root / 'CelebA-HQ-to-CelebA-mapping.txt', sep=r'\s+')
    ident = pd.read_csv(CELEBA_DIR / 'identity_CelebA.txt', sep=' ', header=None,
                        names=['orig_file', 'identity'])
    mapping['file'] = mapping['idx'].astype(str) + '.jpg'
    df = attr.merge(mapping[['file', 'orig_file']], on='file').merge(ident, on='orig_file')
    assert len(df) == 30000, len(df)

    rng = np.random.default_rng(args.seed)
    df = df.sample(frac=1.0, random_state=args.seed).drop_duplicates('identity').reset_index(drop=True)
    identities = df['identity'].to_numpy()
    donated_ids = set(rng.choice(identities, size=args.n_donated, replace=False).tolist())
    df['split'] = np.where(df['identity'].isin(donated_ids), 'donated', 'private')

    meta = pd.DataFrame([attrs_to_schema(r) for r in df.to_dict('records')])
    out = pd.concat([df[['file', 'identity', 'split']], meta], axis=1)

    img_dir = WORK / 'images256'
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in out['file']:
        dst = img_dir / f.replace('.jpg', '.png')
        if not dst.exists():
            with Image.open(root / 'CelebA-HQ-img' / f) as im:
                im.convert('RGB').resize((256, 256), Image.LANCZOS).save(dst)
    out['image_256'] = [str(img_dir / f.replace('.jpg', '.png')) for f in out['file']]
    out['image_full'] = [str(root / 'CelebA-HQ-img' / f) for f in out['file']]

    out[out.split == 'private'].to_csv(WORK / 'private.csv', index=False)
    out[out.split == 'donated'].to_csv(WORK / 'donated.csv', index=False)
    summary = {'n_unique_identities': int(len(out)),
               'n_private': int((out.split == 'private').sum()),
               'n_donated': int((out.split == 'donated').sum()),
               'privacy_unit': 'person (one image per CelebA identity)',
               'private_marginals': {c: out[out.split == 'private'][c].value_counts(normalize=True).round(4).to_dict()
                                     for c in meta.columns}}
    (WORK / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
