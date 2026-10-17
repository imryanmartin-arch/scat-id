"""Bundle DINOv2 + trained head into one ONNX file for the phone.

Input:  "image"  float32 [N,3,224,224], RGB in 0..1 (centre-square crop)
Output: "probs"  float32 [N,num_classes]
Usage: python export.py [small|base]
"""
import json, os, sys
import numpy as np, torch
from transformers import AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType

DATA = os.path.join(os.path.dirname(__file__), "data")
APP = os.path.join(os.path.dirname(__file__), "..", "app", "public", "model")
SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
os.makedirs(APP, exist_ok=True)
head = np.load(os.path.join(DATA, f"head_{SIZE}.npz"))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(f"facebook/dinov2-{SIZE}", attn_implementation="eager").eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.lin = torch.nn.Linear(head["W"].shape[1], head["W"].shape[0])
        self.lin.weight.data = torch.from_numpy(head["W"]); self.lin.bias.data = torch.from_numpy(head["b"])
        self.T = float(head["T"])

    def forward(self, image):
        h = self.bb(pixel_values=(image - self.mean) / self.std).last_hidden_state
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)
        return torch.softmax(self.lin(f) / self.T, -1)


net = Net().eval()
fp32 = os.path.join(DATA, f"scat_{SIZE}_fp32.onnx")
torch.onnx.export(net, (torch.rand(1, 3, 224, 224),), fp32, input_names=["image"], output_names=["probs"],
                  dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
out = os.path.join(APP, "scat.onnx")
quantize_dynamic(fp32, out, weight_type=QuantType.QInt8, per_channel=True)

# Sanity check: quantized model agrees with PyTorch.
import onnxruntime as ort
x = torch.rand(4, 3, 224, 224)
ref = net(x).detach().numpy()
q = ort.InferenceSession(out).run(None, {"image": x.numpy()})[0]
print("max prob diff after int8:", float(np.abs(ref - q).max()), " argmax agree:", (ref.argmax(1) == q.argmax(1)).mean())
print("model size MB:", round(os.path.getsize(out) / 1e6, 1))
json.dump({"classes": [str(c) for c in head["classes"]], "input": 224, "backbone": SIZE},
          open(os.path.join(APP, "model.json"), "w"))
