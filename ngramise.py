import nltk
import math
from nltk.stem import WordNetLemmatizer
from glob import glob
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from tika import parser
from os import path
import collections
from itertools import chain
import itertools
import functools
import bibtexparser

lemmatizer = WordNetLemmatizer()

paper_count = 8
grams_limit = 5
min_paper_occurence = 1

custom_stop_words = set(
    ["introduction", "abstract", "data", "author", "doi", "www", "content", "http", "fig", "figure", "section",
     "presents", ("future", "work"),
     ("paper", "organized", "follows")])
stop_words = set(stopwords.words('english'))
# stop_words = set()
stop_words.update(custom_stop_words)

TITLE_AND_ABSTRACT = lambda x: x[x.find("abstract"):].find("introduction\n")
TITLE_ABSTRACT_TEXT = lambda x: x.find("references\n")

# paper_reader_stopper = TITLE_ABSTRACT_TEXT


paper_reader_stopper = TITLE_AND_ABSTRACT


def get_grams(pattern, ):
    grams = collections.defaultdict(set)

    f = nltk.FreqDist()

    i = 0
    for v in glob(pattern):

        if v.endswith(".pdf"):
            raw = parser.from_file(v)
            f = __extract_ngrams_from_raw_content(f, grams, i, raw["content"])
            i += 1
        elif v.endswith(".bib"):
            with open(v) as bibtex_file:
                bib_database = bibtexparser.load(bibtex_file)
                for bib_entry in bib_database.entries:
                    raw = bib_entry["title"] + " abstract " + bib_entry.get("abstract", "") + " " + " ".join(
                        bib_entry.get("keyword", [])) + " "
                    f = __extract_ngrams_from_raw_content(f, grams, i, raw)
                    i += 1

    return grams, f


def __extract_ngrams_from_raw_content(f, grams, i, raw):
    doc = raw.lower()
    doc = doc[0:paper_reader_stopper(doc)]
    doc = doc.replace("\n", " ").replace("- ", "")
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z_]{3,}")
    new_words = tokenizer.tokenize(doc)
    new_words = [lemmatizer.lemmatize(n) for n in new_words if n not in stop_words]
    for j in range(1, grams_limit):

        sixgrams = [x for x in nltk.ngrams(new_words, j)]
        f += nltk.FreqDist(sixgrams)

        for gram in sixgrams:
            grams[i].add(gram)
    return f


# good_grams, good_dist = get_grams('qgs/*.pdf')
good_grams, good_dist = get_grams('scopus/*.bib')
bad_grams, bad_dist = get_grams('dummy/*.pdf')

good_grams_dict = collections.defaultdict(list)
for k, v in good_grams.items():
    for gram in v:
        good_grams_dict[gram].append(k)

bad_grams_dict = collections.defaultdict(list)
for k, v in bad_grams.items():
    for gram in v:
        bad_grams_dict[gram].append(k)

all_grams = set()
for k, v in good_grams.items():
    for g in v:
        all_grams.add(g)

keywords_list = []
most_frequents_ngrams = {}
for w in all_grams:
    most_frequents_ngrams[w]=sum([1 for k,v in good_grams.items() if w in v])





def get_score(params, good_grams, bad_grams):
    selected_documents = [set(), set()]

    for p in params:
        for k, v in good_grams.items():
            if p in v:
                selected_documents[0].add(k)
        for p in params:
            for k, v in bad_grams.items():
                if p in v:
                    selected_documents[1].add(k)

    good_count = len(good_grams)
    bad_count = len(bad_grams)

    precision = 100 * (len(selected_documents[0]) + 1) / (len(selected_documents[0]) + len(selected_documents[1]) + 1)
    recall = 100 * len(selected_documents[0]) / good_count
    return precision, recall, selected_documents[0], selected_documents[1]


get_score([("blockchain",)], good_grams, bad_grams)
from itertools import chain

queries = list(({cc[3] for cc in c} for c in
                chain(itertools.combinations(keywords_list, 1), itertools.combinations(keywords_list, 2)) if
                len(functools.reduce(lambda x, y: x | set(y[2]), c, set())) == len(good_grams)))

results = []
for i, query in enumerate(queries):
    print(f'{i:5}/{len(queries):05}', end="\r")
    p, r, tp, fp = get_score(query, good_grams, bad_grams)
    # print("%s\n\tprecision=%3.2f%%\trecall=%3.2f%%" % (" AND ".join([" ".join(q) for q in query]), p, r))
    results.append((query, p, r))

print()

results = sorted(results, key=lambda x: -x[1])
results_dict = {"' AND '".join(["  ".join(c) for c in r[0]]): r for r in results}

for i, (k, v) in enumerate(results_dict.items()):
    if i > 200:
        break

    print(f"'{k:60}': {v[1]}/{v[2]}")
