import requests, time, json
names = ["Odocoileus hemionus","Cervus canadensis","Ursus americanus","Puma concolor","Canis latrans","Lynx rufus",
"Urocyon cinereoargenteus","Procyon lotor","Meleagris gallopavo","Leporidae","Erethizon dorsatum","Mephitidae",
"Lontra canadensis","Canis lupus","Canis familiaris","Bos taurus","Equus caballus","Didelphis virginiana",
"Tetraoninae","Odontophoridae","Bonasa umbellus","Dendragapus","Vulpes vulpes","Bassariscus astutus","Neovison vison","Martes caurina","Pekania pennanti","Castor canadensis","Sciuridae","Marmota flaviventris","Homo sapiens"]
L="cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
out={}
for n in names:
    t=requests.get("https://api.inaturalist.org/v1/taxa",params={"q":n,"per_page":5}).json()["results"]
    t=[x for x in t if x["name"].lower()==n.lower()] or t
    tid=t[0]["id"]; time.sleep(1)
    c={}
    for q in ["research","research,needs_id,casual"]:
        r=requests.get("https://api.inaturalist.org/v1/observations",params={"term_id":22,"term_value_id":25,"taxon_id":tid,"photos":"true","photo_license":L,"quality_grade":q,"per_page":0}).json()
        c[q]=r["total_results"]; time.sleep(1)
    out[n]=(tid,t[0]["name"],t[0].get("preferred_common_name"),c)
    print(n,out[n],flush=True)
json.dump(out,open("taxa_lookup.json","w"),indent=1)
