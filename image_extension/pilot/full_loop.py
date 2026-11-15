"""Full image-MAPLE loop, Route A (advisor-approved 2026-11-13).

  DP metadata (AIM) + 1 Hamming-nearest donated reference  -> RANDOM (Qwen-Image-Edit)
  -> copy filter (reject + retry if too close to a donated face)
  -> DP 1-NN voting of private images in Inception space, keep top N
  -> VARIATION: L img2img variants of every kept image (+ copy filter) -> vote -> ... for T rounds

Stages (all resumable; generation is sharded with --rank/--world, one process per GPU):
  gen-init  --eps E                  N*(L+1) initial candidates, shared by every T at this eps
  vote      --eps E --T T --iter t   noisy voting over the pool of iteration t, keep top N
  gen-var   --eps E --T T --iter t   L variations per image selected at iteration t-1
  evaluate  --eps E --T T            FID / KID / copy / metadata of the kept set at every iteration

Arms (--arm):
  maple   the method above.
  random  control: same init pool and VARIATION as maple, but every vote is replaced by a uniform random draw
          of N. Reuses maple's init images (never generates them; run maple gen-init first). Isolates what
          the DP votes add beyond repeated img2img.
  pe      Private Evolution baseline: init is text-to-image from the generic prompt (no DP metadata, no
          reference image; the pilot's 'random' group), VARIATION also uses the generic prompt, and the whole
          budget goes to the votes (no AIM). The copy filter stays on so all arms are filtered alike.

Layout: WORK/<root>/config.json, dp_metadata_eps<E>.{csv,json}
        WORK/<root>/eps<E>/init/{00000.png, log_rank*.jsonl}                  (maple; shared by random)
        WORK/<root>/eps<E>/T<T>/iter<t>/{var/*.png, var/log_rank*.jsonl, selected.csv, vote.json}
        WORK/<root>/eps<E>/T<T>/eval.json
        WORK/<root>/eps<E>/random/T<T>/...                                    (random arm)
        WORK/<root>/eps<E>/pe/{init, T<T>/...}                                (pe arm)

Privacy (conditional accounting; same assumptions as privacy_accounting_calibrated.py, not an audit):
  person-level replace-one adjacency (prepare_celeba keeps one image per identity), delta = --delta
  ('1/n' = 1/n_private, the default, or a number such as 1e-6; it is part of config.json, so use another --root),
  rho_total = zcdp_rho(eps, delta) (aim/accounting.py, as dp_metadata.py). AIM spends aim_fraction*rho_total;
  the T+1 voting rounds share the rest. The 1-NN histogram has L2 sensitivity sqrt(2), so
  sigma = sqrt(2) * sqrt((T+1) / (2 * rho_votes)). Only noisy votes are written to disk.
  A vote is never re-drawn: if selected.csv exists the vote stage skips.
  The copy filter compares synthetic images with the PUBLIC donated pool using FIXED thresholds, so it is
  post-processing of DP outputs plus public data. The thresholds are the pilot's real-image p99 values
  (measured on private images once, during the pilot) and are frozen here, never re-estimated.
"""
import argparse
import json
import math
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_common import SCHEMA, WORK, describe, hamming, load_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'aim'))

FACE_THR = 0.620   # pilot p99 of real-private vs donated FaceNet cosine (frozen)
CLIP_THR = 0.853   # pilot p99 of real-private vs donated CLIP cosine (frozen; reported, not filtered on)
CONFIG_KEYS = ('N', 'L', 'strengths', 'steps', 'size', 'cfg', 'face_thr', 'max_tries', 'aim_fraction', 'seed',
               'delta')
CONFIG_DEFAULTS = {'delta': '1/n'}  # keys added after a root was created; an old root implicitly has these
ARMS = ('maple', 'random', 'pe')


# ---------------------------------------------------------------- paths / config
def root(a):
    return WORK / a.root


def eps_dir(a):
    return root(a) / f'eps{a.eps:g}'


def arm_dir(a):
    return eps_dir(a) if a.arm == 'maple' else eps_dir(a) / a.arm


def init_dir(a):
    return (eps_dir(a) if a.arm == 'random' else arm_dir(a)) / 'init'  # random reuses maple's init


def iter_dir(a, t):
    return arm_dir(a) / f'T{a.T}' / f'iter{t}'


def delta_of(a, n_private):
    return 1.0 / n_private if a.delta == '1/n' else float(a.delta)


def meta_csv(a):
    return root(a) / f'dp_metadata_eps{a.eps:g}.csv'


def check_config(a):
    cfg = {k: getattr(a, k) for k in CONFIG_KEYS}
    f = root(a) / 'config.json'
    if f.exists():
        old = json.loads(f.read_text())
        diff = {k: (old.get(k, CONFIG_DEFAULTS.get(k)), v) for k, v in cfg.items()
                if old.get(k, CONFIG_DEFAULTS.get(k)) != v}
        if diff:
            raise SystemExit(f'{f} was created with different settings {diff}; use another --root')
        if any(k not in old for k in cfg):
            f.write_text(json.dumps(cfg, indent=2))
    else:
        root(a).mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cfg, indent=2))


def sigma_for(a, n_private):
    import accounting
    delta = delta_of(a, n_private)
    rho_total = accounting.zcdp_rho(a.eps, delta)
    aim_fraction = 0.0 if a.arm == 'pe' else a.aim_fraction  # PE has no metadata step
    rho_votes = (1 - aim_fraction) * rho_total
    K = a.T + 1
    sigma = math.sqrt(2) * math.sqrt(K / (2 * rho_votes))
    return sigma, {'arm': a.arm, 'epsilon_total': a.eps, 'delta': delta, 'rho_total': rho_total,
                   'rho_aim': aim_fraction * rho_total, 'rho_votes': rho_votes, 'voting_rounds': K,
                   'l2_sensitivity': math.sqrt(2), 'sigma': sigma,
                   'status': 'CONDITIONAL_ACCOUNTING_NOT_AUDITED'}


def read_log(d):
    """Last log entry per output name; an image counts only if its PNG exists."""
    rows = {}
    for f in sorted(d.glob('log_rank*.jsonl')):
        for line in f.open():
            r = json.loads(line)
            rows[r['name']] = r
    return {k: r for k, r in rows.items() if (d / f'{k}.png').exists()}


def seed_of(*parts):
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] % (2 ** 31))


# ---------------------------------------------------------------- generation
class Generator:
    def __init__(self, a, need_var, need_t2i=False):
        import torch
        from diffusers import QwenImageImg2ImgPipeline
        from evaluate_init import FaceNet
        from generate_init import load_pipelines
        self.torch, self.a = torch, a
        self.edit, self.t2i = load_pipelines(True, need_t2i, a.size * a.size)
        self.var = None
        if need_var:
            e = self.edit
            self.var = QwenImageImg2ImgPipeline(scheduler=e.scheduler, vae=e.vae, text_encoder=e.text_encoder,
                                                tokenizer=e.tokenizer, transformer=e.transformer)
            self.var.set_progress_bar_config(disable=True)
        self.face = FaceNet()
        self.don_face = np.load(WORK / 'emb_donated.npz')['face']

    def common(self, prompt, seed):
        return dict(prompt=prompt, negative_prompt=' ', true_cfg_scale=self.a.cfg,
                    num_inference_steps=self.a.steps, height=self.a.size, width=self.a.size,
                    generator=self.torch.Generator('cuda').manual_seed(seed))

    def face_nn(self, img):
        from PIL import Image
        # same 256px LANCZOS downscale as evaluate_init.load256, so filter and evaluation agree
        f, found = self.face([img.convert('RGB').resize((256, 256), Image.LANCZOS)])
        s = f @ self.don_face.T
        return float(s.max()), int(s.argmax()), bool(found[0])

    def run(self, img_fn):
        """img_fn(attempt) -> (PIL image, info). Retry until the face is not a donated copy."""
        tries = []
        for attempt in range(self.a.max_tries):
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                img, info = img_fn(attempt)
            sim, idx, found = self.face_nn(img)
            tries.append({**info, 'face_nn': round(sim, 4), 'face_nn_idx': idx, 'face_found': found,
                          'seconds': round(time.perf_counter() - t0, 2)})
            if sim <= self.a.face_thr:
                return img, tries, True
        return img, tries, False


def write(out_dir, log, name, img, payload):
    log.write(json.dumps({'name': name, **payload}) + '\n')
    log.flush()
    img.save(out_dir / f'{name}.png')


def gen_init(a):
    if a.arm == 'random':
        raise SystemExit('the random arm reuses the maple init; run gen-init with --arm maple')
    if a.arm == 'pe':
        return gen_init_pe(a)
    from PIL import Image
    from generate_init import BASE, REF_INSTRUCTION
    meta = pd.read_csv(meta_csv(a))[list(SCHEMA)].astype(str).to_dict('records')
    n = a.N * (a.L + 1)
    if len(meta) < n:
        raise SystemExit(f'{meta_csv(a)} has {len(meta)} rows, need {n}')
    donated = load_split('donated')
    donors = donated[list(SCHEMA)].astype(str).to_dict('records')
    out = init_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    done = read_log(out)
    todo = [i for i in range(a.rank, n, a.world) if f'{i:05d}' not in done]
    print(f'gen-init eps={a.eps} rank={a.rank}: {len(todo)} to do', flush=True)
    if not todo:
        return
    g = Generator(a, need_var=False)
    log = (out / f'log_rank{a.rank}.jsonl').open('a')
    for i in todo:
        m = meta[i]
        rng = np.random.default_rng(a.seed + i)  # same tie-break rule as the pilot's build_request
        order = np.argsort(np.array([hamming(m, r) for r in donors]) + rng.random(len(donors)) * 0.5)

        def make(attempt, m=m, order=order, i=i):
            r = int(order[attempt])  # retry with the next-nearest donor and a fresh seed
            prompt = (REF_INSTRUCTION.format(k=1) + f' Picture 1 shows {describe(donors[r])}. ' + BASE +
                      f' The new person is {describe(m)}.')
            seed = seed_of(a.seed, 0, i, attempt)
            ref = Image.open(donated['image_full'].iloc[r]).convert('RGB')
            img = g.edit(image=[ref], guidance_scale=1.0, **g.common(prompt, seed)).images[0]
            return img, {'seed': seed, 'ref': donated['file'].iloc[r], 'prompt': prompt}

        img, tries, ok = g.run(make)
        write(out, log, f'{i:05d}', img, {'i': i, 'metadata': m, 'accepted': ok, 'tries': tries})
    log.close()


def gen_init_pe(a):
    """PE init: generic text-to-image prompt; no metadata, no reference image."""
    from generate_init import BASE
    n = a.N * (a.L + 1)
    out = init_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    done = read_log(out)
    todo = [i for i in range(a.rank, n, a.world) if f'{i:05d}' not in done]
    print(f'gen-init arm=pe eps={a.eps} rank={a.rank}: {len(todo)} to do', flush=True)
    if not todo:
        return
    g = Generator(a, need_var=False, need_t2i=True)
    log = (out / f'log_rank{a.rank}.jsonl').open('a')
    for i in todo:
        def make(attempt, i=i):
            seed = seed_of(a.seed, 0, i, attempt)
            return g.t2i(**g.common(BASE, seed)).images[0], {'seed': seed, 'prompt': BASE}

        img, tries, ok = g.run(make)
        write(out, log, f'{i:05d}', img, {'i': i, 'metadata': None, 'accepted': ok, 'tries': tries})
    log.close()


def gen_var(a):
    from PIL import Image
    from generate_init import BASE
    t = a.iter
    sel = iter_dir(a, t - 1) / 'selected.csv'
    if not sel.exists():
        raise SystemExit(f'missing {sel}: run vote --iter {t - 1} first')
    parents = pd.read_csv(sel)
    strength = a.strengths[min(t, len(a.strengths)) - 1]
    out = iter_dir(a, t) / 'var'
    out.mkdir(parents=True, exist_ok=True)
    done = read_log(out)
    todo = [(j, k) for j in range(a.rank, len(parents), a.world) for k in range(a.L)
            if f'{j:04d}_{k}' not in done]
    print(f'gen-var eps={a.eps} T={a.T} iter={t} rank={a.rank}: {len(todo)} to do', flush=True)
    if not todo:
        return
    g = Generator(a, need_var=True)
    log = (out / f'log_rank{a.rank}.jsonl').open('a')
    for j, k in todo:
        p = parents.iloc[j]
        m = None if a.arm == 'pe' else {c: str(p[c]) for c in SCHEMA}
        prompt = BASE if m is None else f'{BASE} The person is {describe(m)}.'
        parent_img = Image.open(p['path']).convert('RGB')

        def make(attempt, j=j, k=k):
            seed = seed_of(a.seed, a.T, t, j, k, attempt)
            img = g.var(image=parent_img, strength=strength, **g.common(prompt, seed)).images[0]
            return img, {'seed': seed, 'prompt': prompt, 'strength': strength}

        img, tries, ok = g.run(make)
        write(out, log, f'{j:04d}_{k}', img, {'parent': j, 'parent_path': p['path'], 'metadata': m,
                                              'accepted': ok, 'tries': tries})
    log.close()


# ---------------------------------------------------------------- voting
def meta_cols(m):
    return {c: (m[c] if m else 'none') for c in SCHEMA}  # pe has no metadata


def pool_of(a, t):
    """Candidate pool of iteration t: init (t=0) or kept parents of t-1 plus their variations."""
    rows = []
    if t == 0:
        d = init_dir(a)
        n = a.N * (a.L + 1)
        log = read_log(d)
        missing = [i for i in range(n) if f'{i:05d}' not in log]
        if missing:
            raise SystemExit(f'init incomplete: {len(missing)} missing, e.g. {missing[:5]}')
        for name, r in sorted(log.items()):
            if r['accepted']:
                rows.append({'path': str(d / f'{name}.png'), 'origin': 'init', 'parent': -1,
                             **meta_cols(r['metadata'])})
    else:
        prev = pd.read_csv(iter_dir(a, t - 1) / 'selected.csv')
        for _, p in prev.iterrows():
            rows.append({'path': p['path'], 'origin': 'kept', 'parent': -1, **{c: p[c] for c in SCHEMA}})
        d = iter_dir(a, t) / 'var'
        log = read_log(d)
        missing = [f'{j:04d}_{k}' for j in range(len(prev)) for k in range(a.L) if f'{j:04d}_{k}' not in log]
        if missing:
            raise SystemExit(f'variations incomplete: {len(missing)} missing, e.g. {missing[:5]}')
        for name, r in sorted(log.items()):
            if r['accepted']:
                rows.append({'path': str(d / f'{name}.png'), 'origin': 'variation', 'parent': r['parent'],
                             **meta_cols(r['metadata'])})
    return pd.DataFrame(rows), len(log)


def vote(a):
    from evaluate_init import Inception, load256
    t = a.iter
    d = iter_dir(a, t)
    if (d / 'selected.csv').exists():
        print(f'{d}/selected.csv exists; votes are never re-drawn. Skipping.')
        return
    pool, n_generated = pool_of(a, t)
    if len(pool) < a.N:
        raise SystemExit(f'pool has {len(pool)} accepted images < N={a.N}')
    if a.arm == 'random':
        return select_random(a, t, pool, n_generated)
    priv = np.load(WORK / 'emb_private.npz')['inception'].astype(np.float64)
    sigma, acct = sigma_for(a, len(priv))

    inc = Inception()
    syn = np.concatenate([inc([load256(p) for p in pool['path'].iloc[s:s + 64]])
                          for s in range(0, len(pool), 64)]).astype(np.float64)
    hist = np.zeros(len(syn))
    sq = (syn ** 2).sum(1)
    for s in range(0, len(priv), 512):
        b = priv[s:s + 512]
        hist += np.bincount(np.argmin(sq[None, :] - 2 * b @ syn.T, axis=1), minlength=len(syn))
    rnd = secrets.SystemRandom()
    noisy = hist + np.array([rnd.normalvariate(0.0, sigma) for _ in range(len(hist))])
    del hist  # only the noisy histogram is used or stored
    keep = np.argsort(-noisy, kind='stable')[:a.N]
    sel = pool.iloc[keep].copy()
    sel['vote_noisy'] = noisy[keep]
    d.mkdir(parents=True, exist_ok=True)
    sel.to_csv(d / 'selected.csv', index=False)
    info = {'iter': t, 'pool': len(pool), 'generated': n_generated,
            'rejected_as_copy': n_generated - int((pool.origin != 'kept').sum()),
            'kept_origin': sel.origin.value_counts().to_dict(), 'privacy': acct}
    (d / 'vote.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def select_random(a, t, pool, n_generated):
    """Control arm: uniform draw of N from the pool; reads no private data, so it spends no budget."""
    rng = np.random.default_rng(seed_of(a.seed, a.T, t, 999))
    keep = np.sort(rng.choice(len(pool), a.N, replace=False))
    sel = pool.iloc[keep].copy()
    d = iter_dir(a, t)
    d.mkdir(parents=True, exist_ok=True)
    sel.to_csv(d / 'selected.csv', index=False)
    n_priv = len(load_split('private'))
    import accounting
    delta = delta_of(a, n_priv)
    rho_total = accounting.zcdp_rho(a.eps, delta)
    info = {'iter': t, 'selection': 'uniform_random', 'pool': len(pool), 'generated': n_generated,
            'rejected_as_copy': n_generated - int((pool.origin != 'kept').sum()),
            'kept_origin': sel.origin.value_counts().to_dict(),
            'privacy': {'arm': 'random', 'epsilon_total': a.eps, 'delta': delta, 'rho_total': rho_total,
                        'rho_aim': a.aim_fraction * rho_total, 'rho_votes': 0.0,
                        'note': 'only the AIM metadata touches private data; the unspent vote budget is unused',
                        'status': 'CONDITIONAL_ACCOUNTING_NOT_AUDITED'}}
    (d / 'vote.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------- evaluation
def evaluate(a):
    from sklearn.linear_model import LogisticRegression
    from evaluate_init import Clip, FaceNet, Inception, embed, fid, jsd, kid
    models = {'inception': Inception(), 'clip': Clip(), 'face': FaceNet()}
    private = load_split('private')
    E_priv = dict(np.load(WORK / 'emb_private.npz'))
    E_don = dict(np.load(WORK / 'emb_donated.npz'))
    probes = {c: LogisticRegression(max_iter=2000).fit(E_priv['clip'], private[c].astype(str)) for c in SCHEMA}
    priv_marg = {c: private[c].astype(str).value_counts(normalize=True) for c in SCHEMA}
    res = {'arm': a.arm, 'thresholds': {'face': a.face_thr, 'clip': CLIP_THR}, 'iters': {}}
    t = 0
    while (iter_dir(a, t) / 'selected.csv').exists():
        sel = pd.read_csv(iter_dir(a, t) / 'selected.csv')
        E = embed(sel['path'].tolist(), models)
        nf = (E['face'] @ E_don['face'].T).max(1)
        nc = (E['clip'] @ E_don['clip'].T).max(1)
        r = {'n': len(sel), 'fid': fid(E['inception'], E_priv['inception']),
             'kid_x1e3': kid(E['inception'], E_priv['inception'][:5000]),
             'face_copy_rate': float((nf > a.face_thr).mean()), 'clip_copy_rate': float((nc > CLIP_THR).mean()),
             'origin': sel.origin.value_counts().to_dict(), 'metadata_jsd': {}, 'metadata_agreement': {}}
        for c in SCHEMA:
            pred = pd.Series(probes[c].predict(E['clip']))
            q = pred.value_counts(normalize=True).reindex(priv_marg[c].index, fill_value=0)
            r['metadata_jsd'][c] = jsd(priv_marg[c].to_numpy(), q.to_numpy())
            if a.arm == 'pe':
                continue  # no target metadata to agree with
            want = sel[c].astype(str).reset_index(drop=True)
            mask = want != 'unspecified'  # hair: score only labeled classes
            r['metadata_agreement'][c] = float((pred[mask] == want[mask]).mean())
        r['metadata_jsd_mean'] = float(np.mean(list(r['metadata_jsd'].values())))
        r['metadata_agreement_mean'] = (float(np.mean(list(r['metadata_agreement'].values())))
                                        if r['metadata_agreement'] else None)
        res['iters'][t] = r
        print(t, json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}), flush=True)
        t += 1
    out = arm_dir(a) / f'T{a.T}' / 'eval.json'
    out.write_text(json.dumps(res, indent=2))
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['gen-init', 'vote', 'gen-var', 'evaluate'])
    ap.add_argument('--eps', type=float, required=True)
    ap.add_argument('--T', type=int, default=1)
    ap.add_argument('--iter', type=int, default=0)
    ap.add_argument('--root', default='full')
    ap.add_argument('--arm', choices=ARMS, default='maple')
    ap.add_argument('--delta', default='1/n', help="'1/n' (1/n_private) or a number, e.g. 1e-6")
    ap.add_argument('--N', type=int, default=500, help='synthetic set size kept per iteration')
    ap.add_argument('--L', type=int, default=2, help='variations per kept image; init makes N*(L+1)')
    ap.add_argument('--strengths', type=lambda s: [float(x) for x in s.split(',')], default=[0.6, 0.5, 0.4],
                    help='img2img strength for iteration 1, 2, 3, ... (last value repeats)')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--cfg', type=float, default=4.0)
    ap.add_argument('--face-thr', type=float, default=FACE_THR)
    ap.add_argument('--max-tries', type=int, default=4)
    ap.add_argument('--aim-fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    a = ap.parse_args()
    if a.delta != '1/n':
        a.delta = f'{float(a.delta):g}'
    check_config(a)
    {'gen-init': gen_init, 'vote': vote, 'gen-var': gen_var, 'evaluate': evaluate}[a.stage](a)


if __name__ == '__main__':
    main()
