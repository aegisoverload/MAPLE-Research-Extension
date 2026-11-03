import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from image_maple.backends import MockImageBackend
from image_maple.contracts import PublicExample
from image_maple.embeddings import PixelSmokeEmbedder
from image_maple.engine import run_pipeline
from image_maple.io import load_image_manifest, load_metadata_csv


class SpyBackend(MockImageBackend):
    def __init__(self):
        self.seen = []

    def generate(self, request):
        self.seen.append(request)
        return super().generate(request)


class EndToEndSmokeTest(unittest.TestCase):
    def test_no_private_files_ever_reach_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / 'private.png'
            donor = root / 'donated.png'
            Image.new('RGB', (64, 64), '#ff5533').save(private)
            Image.new('RGB', (64, 64), '#3366ff').save(donor)
            spy = SpyBackend()
            report = run_pipeline(
                backend=spy, embedder=PixelSmokeEmbedder(),
                metadata_rows=[{'style': 'photo', 'lighting': 'soft'}],
                private_images=[private],
                donated_examples=[PublicExample(donor, {'style': 'photo', 'lighting': 'soft'})],
                output_dir=root / 'results', num_syn=2, lookahead=1, iterations=1,
                sigma=1.0, seed=42, n_examples=1)
            self.assertEqual(report['voting_rounds'], 2)
            self.assertEqual(len(spy.seen), 6)
            self.assertTrue(all(private not in [x.image_path for x in r.examples] for r in spy.seen))
            self.assertTrue(all(r.source_image != private for r in spy.seen))
            self.assertTrue((root / 'results' / 'PE_iter_1.csv').exists())
            self.assertFalse(report['status'] == 'CERTIFIED_DP')

    def test_rejects_donated_private_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            same = root / 'same.png'
            Image.new('RGB', (64, 64)).save(same)
            spy = SpyBackend()
            with self.assertRaisesRegex(ValueError, 'overlap'):
                run_pipeline(backend=spy, embedder=PixelSmokeEmbedder(),
                             metadata_rows=[{'x':'y'}], private_images=[same],
                             donated_examples=[PublicExample(same, {'x': 'y'})],
                             output_dir=root / 'results', num_syn=1, lookahead=1,
                             iterations=0, sigma=0.0)
            self.assertEqual(len(spy.seen), 0)


if __name__ == '__main__':
    unittest.main()
