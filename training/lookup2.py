import requests, time
L="cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
for n in ["Tetraonini","Odontophoridae","Bonasa umbellus","Dendragapus","Callipepla californica","Oreortyx pictus","Vulpes vulpes","Bassariscus astutus","Neogale vison","Martes caurina","Pekania pennanti","Castor canadensis","Sciuridae","Marmota flaviventris","Homo sapiens","Sus scrofa","Spilogale gracilis","Mephitis mephitis"]:
    t=requests.get("https://api.inaturalist.org/v1/taxa",params={"q":n,"per_page":10}).json()["results"]; time.sleep(1)
    m=[x for x in t if x["name"].lower()==n.lower()]
    if not m: print(n,"NOT FOUND",[x["name"] for x in t][:5]); continue
    tid=m[0]["id"]
    r=requests.get("https://api.inaturalist.org/v1/observations",params={"term_id":22,"term_value_id":25,"taxon_id":tid,"photos":"true","photo_license":L,"per_page":0,"quality_grade":"research,needs_id,casual"}).json(); time.sleep(1)
    print(n,tid,m[0].get("preferred_common_name"),r["total_results"],flush=True)
