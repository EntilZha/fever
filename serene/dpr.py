"""
Export data to DPR format
"""
import json
from tqdm import tqdm
from pedroai.io import read_jsonlines
from serene.wiki_db import WikiDatabase
from serene import constants as c


def convert_examples(fever_path: str, out_path: str):
    examples = read_jsonlines(fever_path)
    db = WikiDatabase()
    dpr_examples = []
    for ex in tqdm(examples):
        claim_label = ex['label']
        if claim_label == c.NOT_ENOUGH_INFO:
            continue
        evidence_sets = ex['evidence']
        flattened_evidence = []
        for ev_set in evidence_sets:
            for _, _, page, sent_id in ev_set:
                if page is not None:
                    page = page.replace("_", " ")
                    maybe_evidence = db.get_page_sentence(page, sent_id)
                    if maybe_evidence is not None:
                        flattened_evidence.append({'title': page, 'text': maybe_evidence})
        dpr_examples.append({
            'question': ex['claim'],
            "positive_ctxs": flattened_evidence,
            "negative_ctxs": [],
            "hard_negative_ctxs": [],
            'claim_id': ex['id'],
            'claim_label': claim_label
        })
    with open(out_path, 'w') as f:
        json.dump(dpr_examples, f)


def convert_wiki():
    pass