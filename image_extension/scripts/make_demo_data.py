"""Creates 4 toy PRIVATE images, 3 toy donated images, and 8 toy metadata rows.
The toy private data is synthetic and safe for a local smoke test, but is not DP metadata.
"""
from pathlib import Path
import csv
import random
from PIL import Image, ImageDraw


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make():
    # Repo .gitignore excludes /data/ entirely.
    root = Path(__file__).resolve().parents[1] / 'data' / 'demo'
    private, donated = root / 'private', root / 'donated'
    private.mkdir(parents=True, exist_ok=True)
    donated.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261103)
    for folder, n in [(private, 4), (donated, 3)]:
        for i in range(n):
            img = Image.new('RGB', (256, 256), (rng.randrange(25, 215), rng.randrange(25, 215), rng.randrange(25, 215)))
            draw = ImageDraw.Draw(img)
            draw.rectangle((25 + i * 5, 50, 160, 170), outline='white', width=6)
            draw.ellipse((70, 90, 140, 160), fill=(rng.randrange(20, 235), 40, 90))
            img.save(folder / f'{i:02d}.png')
    write_csv(root / 'private.csv', ['image_path'],
              [[f'private/{i:02d}.png'] for i in range(4)])
    write_csv(root / 'donated.csv', ['image_path', 'expression', 'lighting'],
              [[f'donated/{i:02d}.png', ['smile', 'neutral'][i % 2], ['daylight', 'studio'][i % 2]]
               for i in range(3)])
    write_csv(root / 'metadata.csv', ['expression', 'lighting'],
              [[['smile', 'neutral'][i % 2], ['daylight', 'studio'][(i // 2) % 2]] for i in range(8)])
    print(f'Demo data written under {root}')
    print('IMPORTANT: this is PUBLIC TOY METADATA, not AIM-privatized metadata')


if __name__ == '__main__':
    make()
