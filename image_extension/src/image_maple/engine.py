"""Functional image-MAPLE pipeline: local NN voting, public donor examples, external image backend.

Privacy: This MVP is NOT end-to-end DP certified. Do not use with sensitive private images
until metadata provenance, accounting, person-level adjacency and floating-point DP are audited.
"""
import math
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .contracts import GenerationRequest
from .io import write_csv, write_json


@dataclass
class Candidate:
    image: Path
    metadata: dict[str, str]
    score: float = 0.0
    origin: str = ''


def choose_public_examples(metadata, donors, n):
    if n <= 0:
        return tuple()
    # Hamming distance on *public* donor metadata, consistent with MAPLE's premise.
    ranked = sorted(enumerate(donors), key=lambda pair: (
        sum(pair[1].metadata.get(k) != str(v) for k, v in metadata.items()), pair[0]
    ))
    return tuple(donor for _, donor in ranked[:n])


def secure_noisy_nearest_neighbor_votes(private_emb, synthetic_emb, sigma):
    """Each private image votes for one closest synthetic image (Euclidean distance).

    Neighboring dataset bound only CONDITIONAL on fixed, DP-derived candidates and
    one private record contributing one vote. Use audited DP mechanism for real privacy.
    """
    if sigma < 0:
        raise ValueError('sigma must be >= 0')
    # Normalize embeddings in the embedder; compute Euclidean squared distance in blocks.
    hist = np.zeros(len(synthetic_emb), dtype=np.float64)
    for start in range(0, len(private_emb), 128):
        block = private_emb[start:start + 128]
        dists = (np.sum(block ** 2, axis=1, keepdims=True) +
                 np.sum(synthetic_emb ** 2, axis=1)[None, :] -
                 2 * block @ synthetic_emb.T)
        matches = np.argmin(dists, axis=1)
        hist += np.bincount(matches, minlength=len(synthetic_emb))
    if sigma > 0:
        csprng = secrets.SystemRandom()
        hist += np.array([csprng.normalvariate(0, sigma) for _ in hist], dtype=np.float64)
    return hist


def run_pipeline(*, backend, embedder, metadata_rows, private_images, donated_examples,
                 output_dir, num_syn, lookahead, iterations, sigma, seed=42, n_examples=2,
                 metadata_provenance=None):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing nonempty run: {output_dir}')
    if num_syn < 1 or lookahead < 1 or iterations < 0 or n_examples < 0:
        raise ValueError('Invalid experiment sizes')
    if sigma < 0:
        raise ValueError('sigma cannot be negative')
    if not metadata_rows or not private_images:
        raise ValueError('Missing metadata or private manifest')
    private_set = {p.resolve() for p in private_images}
    donors_set = {d.image_path.resolve() for d in donated_examples}
    if private_set.intersection(donors_set):
        raise ValueError('PRIVATE and DONATED examples overlap; aborting before API calls')
    # Never send private images to the backend. They are accessed exclusively here.
    private_emb = embedder.encode(private_images)
    if len(private_emb) != len(private_images):
        raise RuntimeError('Invalid private embedding count')
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)  # not used for DP noise
    current = []
    report = {'status': 'EXPERIMENTAL_NOT_CERTIFIED_DP', 'samples_per_round': num_syn,
              'lookahead': lookahead, 'iterations': iterations, 'voting_rounds': iterations + 1,
              'sigma_used': sigma, 'seed_non_dp_only': seed, 'private_unit': 'image_record',
              'metadata_provenance': metadata_provenance or 'UNVERIFIED',
              'donated_examples': len(donated_examples),
              'notice': 'Pixel smoke encoder and mock backend are NOT research-quality. '
                        'No automatic claim of epsilon-DP; donor license and identity-level DP unresolved.'}
    # Initial: num_syn*(lookahead + 1) like the text MAPLE prototype
    for i in range(num_syn * (lookahead + 1)):
        metadata = dict(metadata_rows[int(rng.integers(0, len(metadata_rows)))])
        public_examples = choose_public_examples(metadata, donated_examples, n_examples)
        path = output_dir / 'images' / f'initial_{i:05d}.png'
        req = GenerationRequest('random', metadata, public_examples, path,
                                seed=seed * 100000 + i)
        saved = backend.generate(req)
        validate_output(saved, path)
        current.append(Candidate(path, metadata, origin='initial'))

    round_stats = []
    for iteration in range(iterations + 1):
        started = time.perf_counter()
        if iteration > 0:
            candidates = list(current)
            for j, parent in enumerate(current):
                for k in range(lookahead):
                    path = output_dir / 'images' / f'iter_{iteration}_{j:05d}_{k:02d}.png'
                    req = GenerationRequest('variation', dict(parent.metadata),
                                            choose_public_examples(parent.metadata, donated_examples, n_examples),
                                            path, seed=seed * 100000 + iteration * 10000 + j * lookahead + k,
                                            source_image=parent.image)
                    saved = backend.generate(req)
                    validate_output(saved, path)
                    candidates.append(Candidate(path, dict(parent.metadata), origin='variation'))
        else:
            candidates = current
        synthetic_emb = embedder.encode([c.image for c in candidates])
        scores = secure_noisy_nearest_neighbor_votes(private_emb, synthetic_emb, sigma)
        selected_indices = np.argsort(scores, kind='stable')[::-1][:num_syn]
        current = []
        for index in selected_indices:
            cand = candidates[int(index)]
            current.append(Candidate(cand.image, cand.metadata, float(scores[index]), cand.origin))
        rows = [{'image_path': str(c.image.relative_to(output_dir)),
                 **{f'meta_{k}': v for k, v in c.metadata.items()},
                 'noisy_vote': c.score,
                 'origin': c.origin} for c in current]
        write_csv(output_dir / f'PE_iter_{iteration}.csv', rows)
        round_stats.append({'iteration': iteration, 'retained': len(current),
                            'selected_variations': sum(c.origin == 'variation' for c in current),
                            'duration_s': round(time.perf_counter() - started, 3)})
    report['round_stats'] = round_stats
    write_json(output_dir / 'run_manifest.json', report)
    return report


def validate_output(actual, expected):
    from PIL import Image
    actual, expected = Path(actual).resolve(), Path(expected).resolve()
    if actual != expected:
        raise ValueError('Backend must write directly to request.output_path')
    with Image.open(actual) as im:
        im.verify()
