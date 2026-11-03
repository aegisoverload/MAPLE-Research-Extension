"""Read simple relative-path CSV manifests. Do not silently load training/private metadata."""
from pathlib import Path
import csv
import json
from .contracts import PublicExample


def load_metadata_csv(file):
    file = Path(file)
    with file.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'Empty metadata CSV: {file}')
        if any(x in reader.fieldnames for x in ('text', 'image_path', 'private_image_path')):
            raise ValueError('DP metadata CSV should contain CATEGORICAL ATTRIBUTES ONLY')
        rows = [{k: (v or '').strip() for k, v in row.items()} for row in reader]
    if not rows:
        raise ValueError('Metadata CSV has no rows')
    return rows


def load_image_manifest(file, donated=False):
    file = Path(file).resolve()
    with file.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'image_path' not in reader.fieldnames:
            raise ValueError(f'{file} requires image_path column')
        entries = []
        for row in reader:
            path = Path(row['image_path'])
            path = (file.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Image missing: {path}')
            metadata = {k: (v or '').strip() for k, v in row.items() if k != 'image_path'}
            entries.append(PublicExample(path, metadata) if donated else path)
    if not entries:
        raise ValueError(f'{file} contains no images')
    return entries


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
