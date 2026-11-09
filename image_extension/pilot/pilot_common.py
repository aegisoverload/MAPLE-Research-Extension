"""Shared schema and paths for the CelebA-HQ initialization pilot (survey §風險與實驗規劃 step 3).

Metadata schema: 9 categorical attributes derived from CelebA's native 40 binary labels,
matching MAPLE's 9-attribute bioRxiv setting. No data-dependent binning.
"""
import os
from pathlib import Path

WORK = Path(os.environ.get('PILOT_WORK', f"/work/{os.environ['USER']}/maple_pilot"))
CELEBA_DIR = Path(f"/work/{os.environ['USER']}/data/celeba-hq")
QWEN_EDIT = Path(f"/work/{os.environ['USER']}/models/Qwen-Image-Edit-2511")
CLIP_DIR = Path(f"/work/{os.environ['USER']}/models/clip-vit-base-patch16")

SCHEMA = {
    'gender': ['female', 'male'],
    'age': ['young', 'older'],
    'smiling': ['no', 'yes'],
    'hair_color': ['black', 'blond', 'brown', 'gray', 'bald', 'other'],
    'eyeglasses': ['no', 'yes'],
    'facial_hair': ['none', 'beard'],
    'makeup': ['no', 'yes'],
    'hat': ['no', 'yes'],
    'bangs': ['no', 'yes'],
}


def attrs_to_schema(a):
    """a: mapping from CelebA attribute name to +1/-1."""
    pos = lambda k: int(a[k]) == 1
    if pos('Bald'):
        hair = 'bald'
    elif pos('Gray_Hair'):
        hair = 'gray'
    elif pos('Blond_Hair'):
        hair = 'blond'
    elif pos('Black_Hair'):
        hair = 'black'
    elif pos('Brown_Hair'):
        hair = 'brown'
    else:
        hair = 'other'
    return {
        'gender': 'male' if pos('Male') else 'female',
        'age': 'young' if pos('Young') else 'older',
        'smiling': 'yes' if pos('Smiling') else 'no',
        'hair_color': hair,
        'eyeglasses': 'yes' if pos('Eyeglasses') else 'no',
        'facial_hair': 'beard' if (not pos('No_Beard') or pos('Goatee') or pos('Mustache')) else 'none',
        'makeup': 'yes' if pos('Heavy_Makeup') else 'no',
        'hat': 'yes' if pos('Wearing_Hat') else 'no',
        'bangs': 'yes' if pos('Bangs') else 'no',
    }


def describe(m):
    """Natural-language rendering of one metadata row for the generation prompt."""
    person = {'female': 'woman', 'male': 'man'}[m['gender']]
    parts = [f"a {'young' if m['age'] == 'young' else 'middle-aged or older'} {person}"]
    if m['hair_color'] == 'bald':
        parts.append('bald')
    elif m['hair_color'] == 'other':
        parts.append('with hair of another color (e.g. red or dyed)')
    else:
        parts.append(f"with {m['hair_color']} hair")
    parts.append('with bangs' if m['bangs'] == 'yes' else 'without bangs')
    parts.append('smiling' if m['smiling'] == 'yes' else 'not smiling')
    parts.append('wearing eyeglasses' if m['eyeglasses'] == 'yes' else 'not wearing eyeglasses')
    parts.append('with a beard or mustache' if m['facial_hair'] == 'beard' else 'clean-shaven / no facial hair')
    parts.append('wearing heavy makeup' if m['makeup'] == 'yes' else 'without heavy makeup')
    parts.append('wearing a hat' if m['hat'] == 'yes' else 'not wearing a hat')
    return ', '.join(parts)


def hamming(a, b):
    return sum(a[k] != b[k] for k in SCHEMA)
