# Scat ID – Jackson County, Oregon

An offline phone app that suggests which animal left a pile of scat, from a photo.
Built for hunters around Medford / the Rogue Valley. No app store, no account, no signal needed after install.

**Not for commercial use.** The model is trained on Creative Commons photos from iNaturalist, many of which are licensed non-commercial (CC BY-NC).

## Install on an iPhone

1. Open the app link in **Safari**.
2. Tap **Share** → **Add to Home Screen**.
3. Open **Scat ID** from the home screen **while you have signal** and wait until it says **Ready · works offline**.

Android: open the link in Chrome → ⋮ → **Install app**.

## What it identifies

Black-tailed deer, Roosevelt elk, black bear, cougar, coyote, bobcat, gray fox, raccoon, ringtail,
Virginia opossum, skunk, porcupine, rabbit/hare, river otter, gray wolf, wild turkey, grouse/quail,
domestic dog, cattle, horse/mule — plus a "not scat" check.

Results are a ranked list with confidence, not a certainty. Photo-only scat ID is genuinely hard:
even trained trackers checked against DNA misidentify coyote/bobcat/fox scat often.
Optional size and shape taps improve the answer a lot.

## How it works

- `training/` — downloads scat-annotated photos from iNaturalist (`download.py`), removes photos that
  aren't scat with CLIP (`clean.py`), computes DINOv2 image features (`embed.py`), trains and evaluates
  a classifier on photographers held out from training (`train.py`), and exports one int8 ONNX model (`export.py`).
- `app/public/` — the installable web app. Runs the model on the phone with onnxruntime-web; a service
  worker stores everything for offline use. Finds are logged on the device only (IndexedDB).

## Photo credits

Training photos are © their photographers on iNaturalist, used under the licenses listed in
[`training/credits.csv`](training/credits.csv).
