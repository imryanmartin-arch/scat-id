"""Write credits.csv: every iNaturalist photo used for training, with photographer + license."""
import csv, json, os
rows = [json.loads(l) for l in open("data/clean.jsonl")]
rows = [r for r in rows if r["keep"]]
with open("credits.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["class", "taxon", "attribution", "license", "observation_url", "photo_url"])
    for r in sorted(rows, key=lambda r: (r["cls"], r["photo_id"])):
        w.writerow([r["cls"], r["taxon"], r["attribution"], r["license"],
                    f"https://www.inaturalist.org/observations/{r['obs_id']}", r["url"]])
print(len(rows), "photos credited")
