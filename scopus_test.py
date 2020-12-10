import requests
import itertools
import sys
import urllib.request
from urllib.error import HTTPError
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

def add_bib(params):
    try:
        i,r=params
        print(f'processing {i}')
        doi = r.get("prism:doi", None)
        if doi is None:
            return None
        url = BASE_URL + doi
        resp = requests.get(url,headers={'Accept': 'application/x-bibtex'})
        r["bibtex"]=resp.text
        print(f'processing {i} done')
        return resp.text
    except Exception as e:
        print(f"Exception at result {i}: {str(e)}")
        return None





#query = '''("blockchain" OR "distributed ledger") AND ("data sharing" OR "data transfer" OR "data transmit") AND ("architecture" OR"design pattern" OR "solution" OR "proposal" OR "system" OR "application") AND ("personal data" OR "personal identifiable information" OR "personal information" OR "privacy") AND ( "GDPR" OR "HIPPA" OR "CalOPPA") '''
query='''( TITLE-ABS-KEY("ICT") OR TITLE-ABS-KEY("Information and Communication Technology"))
AND ("impact" OR "effect" OR "benefit" OR "consequence" OR "repercussion")
AND ("environnmental assessment" OR "ecological assessment" OR "environnmental evaluation" OR "ecological evaluation" OR "environnmental analysis" OR "ecological analysis")
'''
p = 'https://api.elsevier.com/content/search/scopus?start=%d&count=%d&query=%s&apiKey=e0bb646bc7985f4a1eced53309285663'
count=int(requests.get(p % (0, 1, query.replace(" ", "+").replace("\\", "%%22"))).json()["search-results"]["opensearch:totalResults"])

print(f"fetching {count} results")

refs = list(itertools.chain(*[aa["search-results"]["entry"] for aa in
                              [requests.get(p % (i, 25, query.replace(" ", "+").replace("\\", "%%22"))).json() for i in
                               range(0, count, 25)]]))

BASE_URL = 'http://dx.doi.org/'

with ThreadPoolExecutor(max_workers=20) as executor:
    res=list(executor.map(add_bib,[(i,r) for i, r in enumerate(refs)]))
        


print("\n".join([r for r in res if r is not None]))
