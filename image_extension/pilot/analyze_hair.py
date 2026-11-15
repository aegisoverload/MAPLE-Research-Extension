"""What colour is CelebA's unlabeled hair ('other' = none of Black/Blond/Brown/Gray/Bald set)?

Uses the cached CLIP image embeddings of the private set (emb_private.npz, same order as private.csv)
and zero-shot text prompts that include colours CelebA has no label for (red, auburn).
  1. validate zero-shot on images that DO carry a CelebA hair label (accuracy + confusion);
  2. apply it to 'other' images, overall and split by Wearing_Hat;
  3. kNN in CLIP space using only the labeled classes, as a second opinion for 'other';
  4. save a grid of random 'other' images for visual inspection.
Reads the raw pilot CSV (pre-merge labels) on purpose.
"""
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pilot_common import CLIP_DIR, WORK

COLOURS = ['black', 'blond', 'brown', 'gray', 'red', 'auburn']
TEMPLATES = ['a photo of a person with {} hair.', 'a close-up portrait of someone with {} hair.',
             'a face photo, the hair color is {}.']
EXTRA = {'bald': ['a photo of a bald person.', 'a portrait of a man with a shaved bald head.'],
         'hat': ['a photo of a person wearing a hat that covers the hair.']}


def main():
    priv = pd.read_csv(WORK / 'private.csv')
    img = np.load(WORK / 'emb_private.npz')['clip']
    assert len(img) == len(priv)

    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    model = CLIPModel.from_pretrained(CLIP_DIR).eval()
    classes, texts = [], []
    for c in COLOURS:
        classes.append(c); texts.append([t.format(c) for t in TEMPLATES])
    for c, ts in EXTRA.items():
        classes.append(c); texts.append(ts)
    with torch.no_grad():
        T = []
        for ts in texts:
            f = model.get_text_features(**proc(text=ts, return_tensors='pt', padding=True))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            f = torch.nn.functional.normalize(f, dim=-1).mean(0)
            T.append(torch.nn.functional.normalize(f, dim=0).numpy())
    T = np.stack(T)
    pred = np.array(classes)[(img @ T.T).argmax(1)]
    priv['zs'] = pred

    out = {}
    lab = priv[priv.hair_color != 'other']
    out['zeroshot_on_labeled_confusion'] = pd.crosstab(lab.hair_color, lab.zs, normalize='index').round(3).to_dict('index')
    out['zeroshot_on_labeled_acc'] = float((lab.hair_color == lab.zs).mean())

    oth = priv[priv.hair_color == 'other']
    out['n_other'] = int(len(oth))
    out['other_zeroshot'] = oth.zs.value_counts(normalize=True).round(3).to_dict()
    out['other_zeroshot_by_hat'] = {('hat' if h == 1 else 'no_hat'): g.zs.value_counts(normalize=True).round(3).to_dict()
                                    for h, g in oth.groupby(oth.hat.map({'yes': 1, 'no': -1}))}

    # kNN (k=15, cosine) over labeled images only -> which labeled class do 'other' images sit closest to.
    li = lab.index.to_numpy()
    sims = img[oth.index.to_numpy()] @ img[li].T
    nn = np.argsort(-sims, axis=1)[:, :15]
    votes = lab.hair_color.to_numpy()[nn]
    knn = pd.Series([pd.Series(v).value_counts().idxmax() for v in votes])
    out['other_knn_labeled'] = knn.value_counts(normalize=True).round(3).to_dict()
    # Same kNN on labeled images (leave-one-out) as its own accuracy reference.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(li), 1000, replace=False)
    s2 = img[li[sub]] @ img[li].T
    s2[np.arange(len(sub)), sub] = -1
    v2 = lab.hair_color.to_numpy()[np.argsort(-s2, axis=1)[:, :15]]
    out['knn_on_labeled_acc'] = float(np.mean([pd.Series(v).value_counts().idxmax() == t
                                              for v, t in zip(v2, lab.hair_color.to_numpy()[sub])]))

    adir = WORK / 'analysis'
    adir.mkdir(exist_ok=True)
    pick = oth.sample(48, random_state=0)
    grid = Image.new('RGB', (128 * 8, 128 * 6))
    for j, (p, z) in enumerate(zip(pick.image_256, pick.zs)):
        with Image.open(p) as im:
            grid.paste(im.convert('RGB').resize((128, 128)), (128 * (j % 8), 128 * (j // 8)))
    grid.save(adir / 'hair_other_grid.jpg', quality=90)
    pick[['file', 'zs', 'hat']].to_csv(adir / 'hair_other_grid.csv', index=False)
    (adir / 'hair_other.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
