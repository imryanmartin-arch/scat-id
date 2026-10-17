"""Download scat photos from iNaturalist for each class.

Writes data/meta.jsonl (one line per photo) and data/img/<class>/<photo_id>.jpg
Safe to re-run: skips photos already on disk.
"""
import json, os, time, sys
from concurrent.futures import ThreadPoolExecutor
import requests
from classes import CLASSES, NEGATIVES, MAX_OBS_PER_CLASS, LICENSES

API = "https://api.inaturalist.org/v1/observations"
DATA = os.path.join(os.path.dirname(__file__), "data")
S = requests.Session()
S.headers["User-Agent"] = "scat-id-personal-research (non-commercial)"
PHOTOS_PER_OBS = 3


def fetch_obs(taxon_id, term_value, quality, limit):
    params = {"taxon_id": taxon_id, "photos": "true", "photo_license": LICENSES,
              "quality_grade": quality, "per_page": 200, "order_by": "id", "order": "asc"}
    if term_value:
        params.update(term_id=22, term_value_id=term_value)
    out, last = [], 0
    while len(out) < limit:
        params["id_above"] = last
        for attempt in range(5):
            try:
                r = S.get(API, params=params, timeout=60); r.raise_for_status(); break
            except Exception as e:
                print("  retry", e, flush=True); time.sleep(5 * (attempt + 1))
        res = r.json()["results"]
        time.sleep(1.1)
        if not res:
            break
        out += res
        last = res[-1]["id"]
    return out[:limit]


def photo_rows(cls, obs, quality):
    rows = []
    for o in obs:
        for p in o.get("photos", [])[:PHOTOS_PER_OBS]:
            if not p.get("license_code"):
                continue
            rows.append({
                "cls": cls, "photo_id": p["id"], "obs_id": o["id"],
                "user": o["user"]["login"], "quality": quality,
                "license": p["license_code"], "attribution": p.get("attribution"),
                "url": p["url"].replace("/square.", "/medium."),
                "taxon": o.get("taxon", {}).get("name"),
                "loc": o.get("location"),
            })
    return rows


def download(row):
    d = os.path.join(DATA, "img", row["cls"]); path = os.path.join(d, f"{row['photo_id']}.jpg")
    if os.path.exists(path):
        return True
    for attempt in range(3):
        try:
            r = S.get(row["url"], timeout=60); r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            time.sleep(2)
    return False


def main():
    os.makedirs(DATA, exist_ok=True)
    meta_path = os.path.join(DATA, "meta.jsonl")
    done = set()
    if os.path.exists(meta_path):
        done = {json.loads(l)["cls"] for l in open(meta_path)}
    jobs = [(k, tids, 25) for k, _, tids in CLASSES] + \
           [("notscat", [t], tv, n) for t, tv, n in NEGATIVES]
    with open(meta_path, "a") as meta:
        for job in jobs:
            cls, tids, tv = job[0], job[1], job[2]
            limit = job[3] if len(job) > 3 else MAX_OBS_PER_CLASS
            if cls in done and cls != "notscat":
                print("skip", cls); continue
            os.makedirs(os.path.join(DATA, "img", cls), exist_ok=True)
            rows = []
            per_taxon = limit // len(tids) if cls != "notscat" else limit
            for tid in tids:
                # Community-confirmed IDs first, then fill with unconfirmed ones.
                obs = fetch_obs(tid, tv, "research", per_taxon)
                rows += photo_rows(cls, obs, "research")
                n = len(obs)
                if n < per_taxon:
                    more = fetch_obs(tid, tv, "needs_id,casual", per_taxon - n)
                    rows += photo_rows(cls, more, "other")
                    n += len(more)
                print(f"{cls} taxon {tid}: {n} observations", flush=True)
            with ThreadPoolExecutor(8) as ex:
                ok = list(ex.map(download, rows))
            for row, good in zip(rows, ok):
                if good:
                    meta.write(json.dumps(row) + "\n")
            meta.flush()
            print(f"== {cls}: {sum(ok)} photos", flush=True)


if __name__ == "__main__":
    main()
