import requests
import itertools
import sys
import urllib.request
from urllib.error import HTTPError

query = '''
blockchain
'''
p = 'https://api.elsevier.com/content/search/scopus?start=%d&count=%d&query=%s&apiKey=e0bb646bc7985f4a1eced53309285663'
refs = list(itertools.chain(*[aa["search-results"]["entry"] for aa in
                              [requests.get(p % (i, 25, query.replace(" ", "+").replace("\\", "%%22"))).json() for i in
                               range(0, 100, 25)]]))

BASE_URL = 'http://dx.doi.org/'

for i, r in enumerate(refs):

    doi = r.get("prism:doi", None)
    if doi is None:
        continue
    url = BASE_URL + doi
    resp = requests.get(url,headers={'Accept': 'application/x-bibtex'})
    r["bibtex"]=resp



print("\n".join([r["bibtex"] for r in refs if r.get("bibtex", None) is not None]))
