"""Drop downloaded photos that aren't actually scat (animal shots, tracks, habitat).

Uses CLIP zero-shot. Writes data/clean.jsonl with a `scat_p` score per photo and
keeps only photos that look like scat (for animal classes) or clearly don't
(for the notscat class).
"""
import json, os
import torch, open_clip
from PIL import Image
from tqdm import tqdm

DATA = os.path.join(os.path.dirname(__file__), "data")
SCAT = ["a photo of animal scat", "a photo of animal droppings on the ground",
        "a close-up photo of feces", "a pile of animal poop", "a photo of deer pellets",
        "animal scat full of hair and berries"]
OTHER = ["a photo of a wild animal", "a photo of a bird", "a photo of animal footprints in mud or snow",
         "a photo of a landscape", "a photo of a forest trail", "a photo of a person",
         "a photo of a plant or flower", "a photo of a mushroom", "a photo of animal bones",
         "a photo of a dead animal", "a screenshot or map",
         "a photo of a pine cone", "a photo of a rock", "a photo of a tree trunk"]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model = model.to(dev).eval()
with torch.no_grad():
    t = model.encode_text(tok(SCAT + OTHER).to(dev)); t = t / t.norm(dim=-1, keepdim=True)

rows = {}
for l in open(os.path.join(DATA, "meta.jsonl")):
    r = json.loads(l); rows[r["photo_id"]] = r   # dedupe
rows = list(rows.values())

out = open(os.path.join(DATA, "clean.jsonl"), "w")
kept = {}
B = 64
for i in tqdm(range(0, len(rows), B)):
    batch, imgs = [], []
    for r in rows[i:i + B]:
        try:
            imgs.append(prep(Image.open(os.path.join(DATA, "img", r["cls"], f"{r['photo_id']}.jpg")).convert("RGB")))
            batch.append(r)
        except Exception:
            pass
    if not imgs:
        continue
    with torch.no_grad():
        f = model.encode_image(torch.stack(imgs).to(dev)); f = f / f.norm(dim=-1, keepdim=True)
        p = (100 * f @ t.T).softmax(-1)
        sp = p[:, :len(SCAT)].sum(-1).cpu().tolist()
    for r, s in zip(batch, sp):
        r["scat_p"] = round(s, 4)
        keep = s < 0.2 if r["cls"] == "notscat" else s > 0.3
        r["keep"] = keep
        out.write(json.dumps(r) + "\n")
        kept.setdefault(r["cls"], [0, 0]); kept[r["cls"]][0] += keep; kept[r["cls"]][1] += 1
out.close()
for c, (k, n) in sorted(kept.items()):
    print(f"{c:10s} kept {k:5d} / {n}")
