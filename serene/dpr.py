"""
Export data to DPR format
"""
import json
from tqdm import tqdm
from pedroai.io import read_jsonlines

from serene.wiki_db import WikiDatabase
from serene.protos.fever_pb2 import WikipediaDump
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


def convert_wiki(tsv_path: str, map_path: str):
    db = WikiDatabase()
    id_to_page_sent = {}
    with open(tsv_path, 'w') as f:
        f.write('id\ttext\ttitle\n')
        idx = 1
        for page_proto in tqdm(db.get_all_pages()):
            page_proto = WikipediaDump.FromString(page_proto)
            title = page_proto.title.replace('\t', ' ')
            for sent_id, sent_proto in page_proto.sentences.items():
                text = sent_proto.text.replace('\t', ' ')
                f.write(f'{idx}\t{text}\t{title}\n')
                id_to_page_sent[str(idx)] = [title, sent_id]
                idx += 1
    
    with open(map_path, 'w') as f:
        json.dump(id_to_page_sent, f)