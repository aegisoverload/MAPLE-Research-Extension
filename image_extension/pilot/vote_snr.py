"""Vote signal-to-noise diagnostic for the Route-A loop (CPU only, uses cached pilot embeddings).

Question: at the sigma that full_loop.py would use for each (eps, T), does noisy 1-NN voting still pick
better images than random selection, or is the top-N essentially random?

Simulation matched to the full run's votes-per-candidate:
  full run: 6167 private voters over a pool of N*(L+1) = 1500 (L=2) or 1000 (L=1) -> mean 4.1 / 6.2 votes.
  here: pool of P pilot images (copy-filtered), V = 6167 * P / pool_full random private images vote,
  keep k = P / (L+1); the remaining private images are a held-out set for FID, so the clean (sigma=0)
  selection is not scored on the same images that voted for it.
Per sigma, R noise draws: overlap of the noisy top-k with the clean top-k, FID of the selection vs held-out
private, and the fraction of the clean selection's FID gain over random-k that survives the noise.

Diagnostic only: it reads the clean private histogram, which the real loop never releases.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from evaluate_init import fid as fid_ref
from pilot_common import WORK

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'aim'))
import accounting  # noqa: E402

FACE_THR = 0.620
GROUPS = [('init_pilot', 'random'), ('init_pilot', 'meta'), ('init_pilot', 'ref'),
          ('init_pilot', 'meta_ref'), ('init_pilot_k1', 'ref'), ('init_pilot_k1', 'meta_ref')]


class FastFID:
    """FID vs a fixed reference set. tr sqrt(S1 S2) = sum sqrt(eig(Xc S2 Xc^T / (k-1))), a k x k problem."""

    def __init__(self, ref):
        self.mu = ref.mean(0)
        self.S = np.cov(ref, rowvar=False)
        self.trS = np.trace(self.S)

    def __call__(self, x):
        mu = x.mean(0)
        xc = x - mu
        k = len(x)
        tr1 = (xc ** 2).sum() / (k - 1)
        ev = np.linalg.eigvalsh(xc @ self.S @ xc.T / (k - 1))
        return float(((mu - self.mu) ** 2).sum() + tr1 + self.trS - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def sigmas(n_private, aim_fraction=0.1):
    out = {}
    for dname, delta in (('1/n', 1.0 / n_private), ('1e-6', 1e-6)):
        for eps in (1, 2, 4):
            rho_votes = (1 - aim_fraction) * accounting.zcdp_rho(eps, delta)
            for T in (1, 2, 3):
                out[f'eps{eps}_T{T}_d{dname}'] = math.sqrt(2) * math.sqrt((T + 1) / (2 * rho_votes))
    return out


def hist_1nn(voters, cand):
    sq = (cand ** 2).sum(1)
    h = np.zeros(len(cand))
    for s in range(0, len(voters), 512):
        b = voters[s:s + 512]
        h += np.bincount(np.argmin(sq[None, :] - 2 * b @ cand.T, axis=1), minlength=len(cand))
    return h


def experiment(name, cand, origin, priv, pool_full, L, sig, R, splits, seed):
    P = len(cand)
    V = int(round(len(priv) * P / pool_full))
    k = int(round(P / (L + 1)))
    if V >= len(priv) - 500:
        raise SystemExit(f'{name}: V={V} leaves too few held-out private images')
    res = {'P': P, 'V': V, 'k': k, 'L': L, 'pool_full': pool_full, 'mean_votes': V / P, 'splits': []}
    for sp in range(splits):
        rng = np.random.default_rng(seed + sp)
        perm = rng.permutation(len(priv))
        voters, held = priv[perm[:V]], priv[perm[V:]]
        F = FastFID(held)
        if sp == 0:  # check the fast FID against evaluate_init.fid once
            x = cand[:k]
            a, b = F(x), fid_ref(x, held)
            assert abs(a - b) < 1e-3 * max(1, abs(b)), (a, b)
        h = hist_1nn(voters, cand)
        clean = np.argsort(-h, kind='stable')[:k]
        clean_set = set(clean.tolist())
        fid_clean = F(cand[clean])
        fid_rand = [F(cand[rng.choice(P, k, replace=False)]) for _ in range(R)]
        srt = np.sort(h)[::-1]
        split = {'hist': {'max': float(srt[0]), 'zero_frac': float((h == 0).mean()),
                          'topk_vote_share': float(srt[:k].sum() / h.sum()),
                          'kth_votes': float(srt[k - 1]), 'median': float(np.median(h)), 'std': float(h.std())},
                 'fid_clean': fid_clean, 'fid_random': float(np.mean(fid_rand)),
                 'fid_random_sd': float(np.std(fid_rand)), 'noisy': {}}
        if origin is not None:
            split['origin_clean'] = {o: float((origin[clean] == o).mean()) for o in np.unique(origin)}
            split['origin_pool'] = {o: float((origin == o).mean()) for o in np.unique(origin)}
        for key, s in sig.items():
            ov, fs, org = [], [], []
            for _ in range(R):
                sel = np.argsort(-(h + rng.normal(0, s, P)), kind='stable')[:k]
                ov.append(len(clean_set.intersection(sel.tolist())) / k)
                fs.append(F(cand[sel]))
                if origin is not None:
                    org.append({o: float((origin[sel] == o).mean()) for o in np.unique(origin)})
            gain = split['fid_random'] - fid_clean
            d = {'sigma': s, 'overlap': float(np.mean(ov)), 'fid': float(np.mean(fs)), 'fid_sd': float(np.std(fs)),
                 'gain_kept': float((split['fid_random'] - np.mean(fs)) / gain) if gain > 0 else None}
            if org:
                d['origin'] = {o: float(np.mean([x[o] for x in org])) for o in org[0]}
            split['noisy'][key] = d
        res['splits'].append(split)
    # average over splits
    keys = list(sig)
    res['summary'] = {
        'fid_clean': float(np.mean([s['fid_clean'] for s in res['splits']])),
        'fid_random': float(np.mean([s['fid_random'] for s in res['splits']])),
        'overlap_random_expected': k / P,
        **{kk: {m: float(np.mean([s['noisy'][kk][m] for s in res['splits']
                                   if s['noisy'][kk][m] is not None] or [np.nan]))
                for m in ('sigma', 'overlap', 'fid', 'gain_kept')} for kk in keys}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, default=50, help='noise draws per sigma')
    ap.add_argument('--splits', type=int, default=3, help='voter/held-out splits')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(WORK / 'analysis' / 'vote_snr.json'))
    a = ap.parse_args()

    priv = np.load(WORK / 'emb_private.npz')['inception'].astype(np.float64)
    sig = {'clean': 0.0, **sigmas(len(priv))}
    emb = {}
    for run, g in GROUPS:
        z = np.load(WORK / 'runs' / run / f'emb_{g}.npz')
        keep = z['face_nn'] <= FACE_THR  # same copy filter as the loop
        emb[f'{run}/{g}'] = z['inception'][keep].astype(np.float64)
        print(f'{run}/{g}: {keep.sum()}/{len(keep)} pass copy filter', flush=True)

    out = {'sigmas': sig, 'n_private': len(priv), 'experiments': {}}
    ra = emb['init_pilot_k1/meta_ref']
    for L, pool_full in ((2, 1500), (1, 1000)):
        name = f'routeA_L{L}'
        out['experiments'][name] = experiment(name, ra, None, priv, pool_full, L, sig, a.R, a.splits, a.seed)
        print(name, json.dumps({k: v for k, v in out['experiments'][name].items() if k not in ('splits', 'summary')}),
              flush=True)

    # mixed pool: does voting prefer the better pilot arms? equal share per arm, P matched to V=half of private
    rng = np.random.default_rng(a.seed)
    per = int(len(priv) / 2 * 1500 / len(priv) / len(GROUPS))  # V ~ n/2 at mean 4.1 votes per candidate
    cand, origin = [], []
    for g, e in emb.items():
        idx = rng.choice(len(e), min(per, len(e)), replace=False)
        cand.append(e[idx]); origin += [g] * len(idx)
    out['experiments']['mixed_L2'] = experiment('mixed_L2', np.concatenate(cand), np.array(origin), priv, 1500, 2,
                                                sig, a.R, a.splits, a.seed)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    for name, ex in out['experiments'].items():
        s = ex['summary']
        print(f"\n== {name}: P={ex['P']} V={ex['V']} k={ex['k']} mean_votes={ex['mean_votes']:.2f}  "
              f"FID clean={s['fid_clean']:.2f} random={s['fid_random']:.2f}  random overlap={s['overlap_random_expected']:.2f}")
        h = ex['splits'][0]['hist']
        print('   hist (split 0):', {k: round(v, 3) for k, v in h.items()})
        print(f"   {'setting':<20}{'sigma':>7}{'overlap':>9}{'FID':>9}{'gain_kept':>10}")
        for kk in sig:
            r = s[kk]
            print(f"   {kk:<20}{r['sigma']:>7.2f}{r['overlap']:>9.3f}{r['fid']:>9.2f}{r['gain_kept']:>10.2f}")
        if 'origin_clean' in ex['splits'][0]:
            sp = ex['splits'][0]
            print('   origin share  pool:', {k: round(v, 3) for k, v in sp['origin_pool'].items()})
            print('                clean:', {k: round(v, 3) for k, v in sp['origin_clean'].items()})
            for kk in ('eps1_T1_d1/n', 'eps4_T1_d1/n', 'eps1_T3_d1/n'):
                print(f'   {kk:>18}:', {k: round(v, 3) for k, v in sp['noisy'][kk]['origin'].items()})
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
