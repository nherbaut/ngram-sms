import requests
import itertools
import sys
import urllib.request
from urllib.error import HTTPError

query = '''
( TITLE-ABS-KEY(Blockchain) OR TITLE-ABS-KEY("distributed ledger")) AND (TITLE-ABS-KEY(architecture) OR TITLE-ABS-KEY(component) OR TITLE-ABS-KEY(solution)) AND (TITLE-ABS-KEY(pattern) OR TITLE-ABS-KEY("case study") OR TITLE-ABS-KEY("software connector") OR TITLE-ABS-KEY("data storage") OR TITLE-ABS-KEY("computational element") OR TITLE-ABS-KEY("asset management") OR TITLE-ABS-KEY("access control") OR TITLE-ABS-KEY("trust management") OR TITLE-ABS-KEY("blockchain as a"))
'''
p = 'https://api.elsevier.com/content/search/scopus?start=%d&count=%d&query=%s&apiKey=e0bb646bc7985f4a1eced53309285663'
refs = list(itertools.chain(*[aa["search-results"]["entry"] for aa in
                              [requests.get(p % (i, 25, query.replace(" ", "+").replace("\\", "%%22"))).json() for i in
                               range(0, 100, 25)]]))

BASE_URL = 'http://dx.doi.org/'

for i, r in enumerate(refs):
    print(f"{i}/{len(refs)} done", end="\r")
    doi = r.get("prism:doi", None)
    if doi is None:
        continue
    url = BASE_URL + doi
    req = urllib.request.Request(url)
    req.add_header('Accept', 'application/x-bibtex')
    try:
        with urllib.request.urlopen(req) as f:
            r["bibtex"] = f.read().decode()
    except:
        print(f"error with {doi}")
        continue

print("\n".join([r["bibtex"] for r in refs if r.get("bibtex", None) is not None]))
