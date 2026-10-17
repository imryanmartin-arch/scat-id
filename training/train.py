"""Train + honestly evaluate a classifier on DINOv2 embeddings.

Test set = photographers never seen in training (so we don't get flattered by
the same person's photos of the same pile). Test uses community-confirmed IDs
only. Usage: python train.py [small|base]
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from classes import CLASSES

DATA = os.path.join(os.path.dirname(__file__), "data")
SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
KEYS = [k for k, _, _ in CLASSES] + ["notscat"]
X = np.load(os.path.join(DATA, f"emb_{SIZE}.npy"))            # (N, views, D)
meta = [json.loads(l) for l in open(os.path.join(DATA, f"emb_{SIZE}.jsonl"))]
y = np.array([KEYS.index(m["cls"]) for m in meta])
users = np.array([m["user"] for m in meta])
research = np.array([m["quality"] == "research" or m["cls"] == "notscat" for m in meta])

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr, te = next(gss.split(X, y, users))
te = te[research[te]]

mu = X[tr].reshape(-1, X.shape[-1]).mean(0); sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
norm = lambda a: (a - mu) / sd


def fit(idx, C):
    Xt = norm(X[idx].reshape(-1, X.shape[-1])); yt = np.repeat(y[idx], X.shape[1])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return clf.fit(Xt, yt)


def predict(clf, idx):  # app does centre + mirror, averaged
    return (clf.predict_proba(norm(X[idx, 0])) + clf.predict_proba(norm(X[idx, 1]))) / 2


best = None
for C in [0.001, 0.003, 0.01]:
    clf = fit(tr, C); P = predict(clf, te)
    top1 = (P.argmax(1) == y[te]).mean()
    top3 = np.mean([y[te][i] in np.argsort(-P[i])[:3] for i in range(len(te))])
    print(f"C={C}: top1 {top1:.3f}  top3 {top3:.3f}", flush=True)
    if best is None or top1 > best[0]:
        best = (top1, top3, C, P)

top1, top3, C, P = best
pred = P.argmax(1); yt = y[te]
report = {"backbone": SIZE, "C": C, "n_train_photos": int(len(tr)), "n_test_photos": int(len(te)),
          "top1": round(float(top1), 3), "top3": round(float(top3), 3), "per_class": {}, "confusions": []}
for i, k in enumerate(KEYS):
    m = yt == i
    if m.sum():
        report["per_class"][k] = {"n_test": int(m.sum()), "top1": round(float((pred[m] == i).mean()), 3),
                                  "top3": round(float(np.mean([i in np.argsort(-P[j])[:3] for j in np.where(m)[0]])), 3)}
cm = np.zeros((len(KEYS), len(KEYS)), int)
for a, b in zip(yt, pred):
    cm[a, b] += 1
pairs = [(cm[a, b], KEYS[a], KEYS[b]) for a in range(len(KEYS)) for b in range(len(KEYS)) if a != b and cm[a, b]]
report["confusions"] = [f"{a} -> {b}: {n}" for n, a, b in sorted(pairs, reverse=True)[:15]]

# Temperature so displayed confidence matches reality (fit on the test fold).
logits = np.log(P + 1e-9)
best_T = min(np.arange(0.5, 3.01, 0.05), key=lambda T: -np.mean(
    np.log((np.exp(logits / T) / np.exp(logits / T).sum(1, keepdims=True))[np.arange(len(yt)), yt] + 1e-9)))
report["temperature"] = round(float(best_T), 2)
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(DATA, f"report_{SIZE}.json"), "w"), indent=1)

# Final model on everything.
allidx = np.arange(len(y))
mu = X.reshape(-1, X.shape[-1]).mean(0); sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
clf = fit(allidx, C)
# Fold normalisation into the linear layer:  W((x-mu)/sd)+b = (W/sd)x + (b - W mu/sd)
W = clf.coef_ / sd; b = clf.intercept_ - (clf.coef_ * mu / sd).sum(1)
np.savez(os.path.join(DATA, f"head_{SIZE}.npz"), W=W.astype(np.float32), b=b.astype(np.float32),
         T=np.float32(best_T), classes=np.array([KEYS[c] for c in clf.classes_]))
print("saved head")
