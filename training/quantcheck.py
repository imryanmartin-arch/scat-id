"""How much does compressing the model change its answers on real photos?"""
import json, random, sys, numpy as np, onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import quantize_dynamic, QuantType
SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
fp32 = f"data/scat_{SIZE}_fp32.onnx"
variants = {"u8": dict(weight_type=QuantType.QUInt8),
            "u8_perchan": dict(weight_type=QuantType.QUInt8, per_channel=True),
            "s8_perchan": dict(weight_type=QuantType.QInt8, per_channel=True)}
for k, kw in variants.items():
    quantize_dynamic(fp32, f"data/q_{SIZE}_{k}.onnx", **kw)
rows = [json.loads(l) for l in open("data/clean.jsonl")]; rows = [r for r in rows if r["keep"]]
random.seed(0); rows = random.sample(rows, 400)
def load(r):
    img = Image.open(f"data/img/{r['cls']}/{r['photo_id']}.jpg").convert("RGB"); w, h = img.size; s = min(w, h)
    c = img.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((224, 224), Image.BICUBIC)
    return (np.asarray(c, np.float32)/255).transpose(2, 0, 1)
X = np.stack([load(r) for r in rows])
def run(f):
    s = ort.InferenceSession(f); return np.concatenate([s.run(None, {"image": X[i:i+32]})[0] for i in range(0, len(X), 32)])
ref = run(fp32)
for k in variants:
    q = run(f"data/q_{SIZE}_{k}.onnx")
    print(k, "argmax agree", round((q.argmax(1) == ref.argmax(1)).mean(), 3), "mean |dp|", round(np.abs(q-ref).mean(), 4))
