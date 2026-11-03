"""Encoders must run LOCALLY on private images. Pixel encoder is for smoke tests only."""
import numpy as np
from PIL import Image


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class PixelSmokeEmbedder:
    """Tiny local diagnostic embedding. Do NOT report FID/CLIP quality from this."""
    def encode(self, image_paths):
        rows = []
        for p in image_paths:
            with Image.open(p) as im:
                arr = np.asarray(im.convert('RGB').resize((16, 16)), dtype=np.float32)
            rows.append(arr.reshape(-1) / 255.0)
        return unit_rows(np.stack(rows))


class LocalCLIPEmbedder:
    def __init__(self, model='openai/clip-vit-base-patch32', device=None, batch_size=8):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).eval().to(self.device)
        self.batch_size = batch_size

    def encode(self, image_paths):
        out = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = []
            try:
                for p in image_paths[i:i + self.batch_size]:
                    img = Image.open(p)
                    imgs.append(img.convert('RGB'))
                inputs = self.processor(images=imgs, return_tensors='pt', padding=True)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                with self.torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                if not isinstance(feats, self.torch.Tensor):
                    feats = feats.pooler_output
                out.append(feats.cpu().numpy())
            finally:
                for img in imgs:
                    img.close()
        return unit_rows(np.concatenate(out, axis=0))
