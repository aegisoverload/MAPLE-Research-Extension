"""Copy-filtered re-evaluation of the initialization pilot.

Question: does a group's FID advantage survive once near-copies of donated images are removed?

Per group:
  1. embed every synthetic image (Inception / CLIP / FaceNet) and cache it with the per-image
     nearest-donated similarities (runs/<run>/emb_<group>.npz, per_image_<group>.csv), so later
     threshold changes need no GPU;
  2. drop images above the real-image p99 threshold (face only, and face OR clip);
  3. FID / KID of the kept set vs private, and -- because FID is biased upward at smaller N --
     the FID of random subsets of the SAME group at the same N (mean over draws) and of every other
     group at that N, so the comparison is N-matched;
  4. count outputs with more than one detected face (e.g. the reference collage copied back).
"""
import argparse
import json

import numpy as np
import pandas as pd

from evaluate_init import FaceNet, Clip, Inception, cached_embed, embed, fid, kid, load256
from pilot_common import WORK

GROUPS = [('init_pilot', 'random'), ('init_pilot', 'meta'), ('init_pilot', 'ref'),
          ('init_pilot', 'meta_ref'), ('init_pilot_k1', 'ref'), ('init_pilot_k1', 'meta_ref')]


def multi_face(paths, det, p_min=0.95):
    n = []
    for p in paths:
        boxes, probs = det.detect(load256(p))
        n.append(0 if boxes is None else int((np.asarray(probs) >= p_min).sum()))
    return np.array(n)


def fid_at_n(x, ref, n, draws, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([fid(x[rng.choice(len(x), n, replace=False)], ref) for _ in range(draws)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=10)
    args = ap.parse_args()

    models = {'inception': Inception(), 'clip': Clip(), 'face': FaceNet()}
    private = pd.read_csv(WORK / 'private.csv')
    donated = pd.read_csv(WORK / 'donated.csv')
    E_priv = cached_embed('private', private['image_256'].tolist(), models)
    E_don = cached_embed('donated', donated['image_256'].tolist(), models)

    # Same real-image baseline / thresholds as evaluate_init.py.
    base_idx = np.random.default_rng(0).choice(len(private), 500, replace=False)
    thr = {k: float(np.quantile((E_priv[k][base_idx] @ E_don[k].T).max(1), 0.99)) for k in ('face', 'clip')}
    ref = E_priv['inception']
    n_ref = min(5000, len(private))

    E = {}
    for run, g in GROUPS:
        d = WORK / 'runs' / run
        cache = d / f'emb_{g}.npz'
        paths = sorted((d / g).glob('*.png'))
        if cache.exists():
            e = dict(np.load(cache))
        else:
            e = embed([str(p) for p in paths], models)
            e['n_faces'] = multi_face(paths, models['face'].det)
            sf, sc = e['face'] @ E_don['face'].T, e['clip'] @ E_don['clip'].T
            e['face_nn'], e['face_nn_idx'] = sf.max(1), sf.argmax(1)
            e['clip_nn'], e['clip_nn_idx'] = sc.max(1), sc.argmax(1)
            np.savez(cache, **e)
            pd.DataFrame({'file': [p.name for p in paths], 'face_nn': e['face_nn'],
                          'face_nn_donated': donated['file'].to_numpy()[e['face_nn_idx']],
                          'clip_nn': e['clip_nn'],
                          'clip_nn_donated': donated['file'].to_numpy()[e['clip_nn_idx']],
                          'n_faces': e['n_faces'], 'face_found': e['face_found']}
                         ).to_csv(d / f'per_image_{g}.csv', index=False)
        E[f'{run}/{g}'] = e
        print('embedded', run, g, len(e['face_nn']), flush=True)

    res = {'thresholds': thr, 'draws': args.draws, 'groups': {}}
    for key, e in E.items():
        x = e['inception']
        keep_face = e['face_nn'] <= thr['face']
        keep_both = keep_face & (e['clip_nn'] <= thr['clip'])
        r = {'n': int(len(x)), 'fid_full': fid(x, ref), 'multi_face_outputs': int((e['n_faces'] > 1).sum()),
             'no_face_outputs': int((e['n_faces'] == 0).sum())}
        for name, keep in (('face', keep_face), ('face_or_clip', keep_both)):
            k = int(keep.sum())
            r[f'filter_{name}'] = {
                'kept': k, 'kept_frac': k / len(x),
                'fid_kept': fid(x[keep], ref),
                'kid_kept_x1e3': kid(x[keep], ref[:n_ref]),
                'fid_random_subset_same_n': fid_at_n(x, ref, k, args.draws),
                'fid_other_groups_at_n': {o: fid_at_n(E[o]['inception'], ref, k, args.draws)
                                          for o in E if o != key}}
        res['groups'][key] = r
        print(key, json.dumps(r, indent=1), flush=True)

    out = WORK / 'runs' / 'filter_eval.json'
    out.write_text(json.dumps(res, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
