"""Compute DINOv2 embeddings for the cleaned photos (plus flipped / cropped copies
for training augmentation). Usage: python embed.py [small|base]
"""
import json, os, sys
import numpy as np, torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel

DATA = os.path.join(os.path.dirname(__file__), "data")
SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained(f"facebook/dinov2-{SIZE}").to(dev).eval()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(img):
    a = (np.asarray(img.resize((224, 224), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1))


def center(img):  # same preprocessing the phone app uses: centre square crop -> 224
    w, h = img.size; s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def views(img):
    c = center(img)
    w, h = c.size; z = int(w * 0.75)
    return [c, ImageOps.mirror(c), c.crop((0, 0, z, z)), c.crop((w - z, h - z, w, h)),
            c.rotate(90), c.crop(((w - z) // 2, (h - z) // 2, (w + z) // 2, (h + z) // 2))]


def feats(x):
    with torch.no_grad():
        h = model(pixel_values=x.to(dev)).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)   # CLS + mean patch token
    return f.cpu().numpy().astype(np.float32)


rows = [json.loads(l) for l in open(os.path.join(DATA, "clean.jsonl"))]
rows = [r for r in rows if r["keep"]]
X, meta = [], []
NV = 6
for i in tqdm(range(0, len(rows), 16)):
    ts, rs = [], []
    for r in rows[i:i + 16]:
        img = Image.open(os.path.join(DATA, "img", r["cls"], f"{r['photo_id']}.jpg")).convert("RGB")
        ts += [to_tensor(v) for v in views(img)]; rs.append(r)
    f = feats(torch.stack(ts)).reshape(len(rs), NV, -1)
    X.append(f); meta += rs
X = np.concatenate(X)
np.save(os.path.join(DATA, f"emb_{SIZE}.npy"), X)
with open(os.path.join(DATA, f"emb_{SIZE}.jsonl"), "w") as f:
    for r in meta:
        f.write(json.dumps({k: r[k] for k in ("cls", "photo_id", "obs_id", "user", "quality")}) + "\n")
print(X.shape)
