"""Evaluate the initialization pilot (all local; private images never leave the node).

Per group:
  fid / kid           Inception pool3, synthetic vs private, both at 256x256 (LANCZOS)
  copy test           max cosine to the 50 donated images in FaceNet (identity) and CLIP space;
                      baseline = the same statistic for 500 real private images (unrelated people).
                      copy_rate = fraction of synthetic above the baseline's 99th percentile.
  diversity           mean pairwise CLIP cosine within the group (lower = more diverse)
  metadata            CLIP linear probes (trained on private labels) relabel synthetic images;
                      JSD of each attribute marginal vs private; for conditioned groups also the
                      agreement with the requested DP metadata.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pilot_common import CLIP_DIR, SCHEMA, WORK

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load256(p):
    with Image.open(p) as im:
        im = im.convert('RGB')
        return im if im.size == (256, 256) else im.resize((256, 256), Image.LANCZOS)


class Inception:
    def __init__(self):
        from pytorch_fid.inception import InceptionV3
        self.m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, imgs):
        x = torch.stack([torch.from_numpy(np.asarray(i, dtype=np.float32) / 255.).permute(2, 0, 1) for i in imgs])
        return self.m(x.to(DEV))[0].squeeze(-1).squeeze(-1).cpu().numpy()


class Clip:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.p = CLIPProcessor.from_pretrained(CLIP_DIR)
        self.m = CLIPModel.from_pretrained(CLIP_DIR).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, imgs):
        f = self.m.get_image_features(**self.p(images=imgs, return_tensors='pt').to(DEV))
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()


class FaceNet:
    def __init__(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.det = MTCNN(image_size=160, margin=16, device=DEV, post_process=True)
        self.m = InceptionResnetV1(pretrained='vggface2').eval().to(DEV)

    @torch.no_grad()
    def __call__(self, imgs):
        crops, found = [], []
        for im in imgs:
            c = self.det(im)
            found.append(c is not None)
            if c is None:  # fall back to a centre crop so every image gets an embedding
                c = torch.from_numpy(np.asarray(im.resize((160, 160)), dtype=np.float32)).permute(2, 0, 1)
                c = (c - 127.5) / 128.0
            crops.append(c)
        f = self.m(torch.stack(crops).to(DEV))
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy(), np.array(found)


def embed(paths, models, bs=64):
    out = {k: [] for k in ('inception', 'clip', 'face', 'face_found')}
    for s in range(0, len(paths), bs):
        imgs = [load256(p) for p in paths[s:s + bs]]
        out['inception'].append(models['inception'](imgs))
        out['clip'].append(models['clip'](imgs))
        f, found = models['face'](imgs)
        out['face'].append(f)
        out['face_found'].append(found)
    return {k: np.concatenate(v) for k, v in out.items()}


def cached_embed(name, paths, models):
    f = WORK / f'emb_{name}.npz'
    if f.exists():
        return dict(np.load(f))
    e = embed(paths, models)
    np.savez(f, **e)
    return e


def fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid(a, b, n_sub=100, sub=None, seed=0):
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    sub = sub or min(len(a), len(b), 1000)
    vals = []
    for _ in range(n_sub):
        x = a[rng.choice(len(a), sub, replace=False)]
        y = b[rng.choice(len(b), sub, replace=False)]
        k = lambda u, v: (u @ v.T / d + 1) ** 3
        kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
        m = sub
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals) * 1000), float(np.std(vals) * 1000)  # reported x1e3


def jsd(p, q):
    p, q = np.asarray(p) + 1e-12, np.asarray(q) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='init_pilot')
    ap.add_argument('--groups', default='random,meta,ref,meta_ref')
    ap.add_argument('--metadata', default=str(WORK / 'dp_metadata_eps1.csv'))
    args = ap.parse_args()
    run_dir = WORK / 'runs' / args.run

    models = {'inception': Inception(), 'clip': Clip(), 'face': FaceNet()}
    private = pd.read_csv(WORK / 'private.csv')
    donated = pd.read_csv(WORK / 'donated.csv')
    E_priv = cached_embed('private', private['image_256'].tolist(), models)
    E_don = cached_embed('donated', donated['image_256'].tolist(), models)
    dp_meta = pd.read_csv(args.metadata)[list(SCHEMA)].astype(str)

    # Real-image baseline for the copy test: 500 private people vs the donated pool.
    rng = np.random.default_rng(0)
    base_idx = rng.choice(len(private), 500, replace=False)
    base = {k: (E_priv[k][base_idx] @ E_don[k].T).max(1) for k in ('face', 'clip')}
    thr = {k: float(np.quantile(v, 0.99)) for k, v in base.items()}

    # Attribute probes on CLIP features, trained on private labels.
    probes, probe_acc = {}, {}
    for a in SCHEMA:
        y = private[a].astype(str).to_numpy()
        xtr, xte, ytr, yte = train_test_split(E_priv['clip'], y, test_size=0.2, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, ytr)
        probes[a], probe_acc[a] = clf, float((clf.predict(xte) == yte).mean())
    priv_marg = {a: private[a].astype(str).value_counts(normalize=True) for a in SCHEMA}

    def pair_sim(x):
        s = x @ x.T
        return float((s.sum() - np.trace(s)) / (len(x) * (len(x) - 1)))

    results = {'baseline_real': {
        'face_nn_donated_median': float(np.median(base['face'])), 'face_thr_p99': thr['face'],
        'clip_nn_donated_median': float(np.median(base['clip'])), 'clip_thr_p99': thr['clip'],
        'clip_pairwise_sim': pair_sim(E_priv['clip'][base_idx]),
        'face_detect_rate': float(E_priv['face_found'].mean())},
        'probe_heldout_acc': probe_acc}
    n_ref = min(5000, len(private))
    results['baseline_real']['fid_500_private_vs_rest'] = fid(E_priv['inception'][base_idx],
                                                              np.delete(E_priv['inception'], base_idx, 0))

    for g in args.groups.split(','):
        paths = sorted((run_dir / g).glob('*.png'))
        if not paths:
            continue
        E = embed([str(p) for p in paths], models)
        idx = [int(p.stem) for p in paths]
        nn_face = E['face'] @ E_don['face'].T
        nn_clip = E['clip'] @ E_don['clip'].T
        r = {'n': len(paths),
             'fid': fid(E['inception'], E_priv['inception']),
             'kid_x1e3': kid(E['inception'], E_priv['inception'][:n_ref]),
             'face_detect_rate': float(E['face_found'].mean()),
             'face_nn_donated_median': float(np.median(nn_face.max(1))),
             'face_copy_rate': float((nn_face.max(1) > thr['face']).mean()),
             'clip_nn_donated_median': float(np.median(nn_clip.max(1))),
             'clip_copy_rate': float((nn_clip.max(1) > thr['clip']).mean()),
             'clip_pairwise_sim': pair_sim(E['clip']),
             'donor_hits_top': pd.Series(nn_face.argmax(1)).value_counts().head(5).to_dict()}
        r['donor_hits_top'] = {donated['file'].iloc[k]: int(v) for k, v in r['donor_hits_top'].items()}
        marg_jsd, agree = {}, {}
        for a in SCHEMA:
            pred = pd.Series(probes[a].predict(E['clip']))
            q = pred.value_counts(normalize=True).reindex(priv_marg[a].index, fill_value=0)
            marg_jsd[a] = jsd(priv_marg[a].to_numpy(), q.to_numpy())
            if g in ('meta', 'meta_ref'):
                agree[a] = float((pred.to_numpy() == dp_meta[a].iloc[idx].to_numpy()).mean())
        r['metadata_jsd_mean'] = float(np.mean(list(marg_jsd.values())))
        r['metadata_jsd'] = marg_jsd
        if agree:
            r['metadata_agreement_mean'] = float(np.mean(list(agree.values())))
            r['metadata_agreement'] = agree
        # Visual copy check: 8 most donor-similar (face) synthetic images next to their donor.
        top = np.argsort(-nn_face.max(1))[:8]
        grid = Image.new('RGB', (512, 256 * len(top)))
        for row, t in enumerate(top):
            grid.paste(load256(paths[t]), (0, 256 * row))
            grid.paste(load256(donated['image_256'].iloc[int(nn_face[t].argmax())]), (256, 256 * row))
        grid.save(run_dir / f'copycheck_{g}.jpg', quality=90)
        sample = Image.new('RGB', (256 * 8, 256 * 2))
        for j, p in enumerate(paths[:16]):
            sample.paste(load256(p), (256 * (j % 8), 256 * (j // 8)))
        sample.save(run_dir / f'samples_{g}.jpg', quality=90)
        results[g] = r
        print(g, json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}))

    (run_dir / 'eval.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
