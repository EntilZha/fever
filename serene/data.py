"""
Utilities for data such as
* Exporting to DPR/Lucene formats
* Scoring outputs of DPR/Lucene
"""
import json
from tqdm import tqdm
from pedroai.io import read_jsonlines
import numpy as np

from serene.wiki_db import WikiDatabase
from serene.protos.fever_pb2 import WikipediaDump
from serene.util import get_logger
from serene import constants as c


log = get_logger(__name__)


def convert_examples_for_training(fever_path: str, out_path: str):
    """
    DPR trains based on a json formatted file where each entry contains
    and example along with all of its positive/negative contexts.
    """
    examples = read_jsonlines(fever_path)
    db = WikiDatabase()
    dpr_examples = []
    for ex in tqdm(examples):
        claim_label = ex["label"]
        if claim_label == c.NOT_ENOUGH_INFO:
            continue
        evidence_sets = ex["evidence"]
        flattened_evidence = []
        for ev_set in evidence_sets:
            for _, _, page, sent_id in ev_set:
                if page is not None:
                    page = page.replace("_", " ")
                    maybe_evidence = db.get_page_sentence(page, sent_id)
                    if maybe_evidence is not None:
                        flattened_evidence.append(
                            {"title": page, "text": maybe_evidence}
                        )
        dpr_examples.append(
            {
                "question": ex["claim"],
                "positive_ctxs": flattened_evidence,
                "negative_ctxs": [],
                "hard_negative_ctxs": [],
                "claim_id": ex["id"],
                "claim_label": claim_label,
            }
        )
    with open(out_path, "w") as f:
        json.dump(dpr_examples, f)


def convert_examples_to_kotlin_json(fever_path: str, out_path: str):
    examples = read_jsonlines(fever_path)
    with open(out_path, "w") as f:
        for ex in tqdm(examples):
            out = {"label": ex["label"], "id": ex["id"], "claim": ex["claim"]}
            out_all_sets = []
            for evidence_set in ex["evidence"]:
                out_evidence_set = []
                for _, _, title, sentence_id in evidence_set:
                    evidence = {"wikipedia_url": None, "sentence_id": None}
                    if title is not None and sentence_id is not None:
                        evidence["wikipedia_url"] = title
                        evidence["sentence_id"] = sentence_id
                    out_evidence_set.append(evidence)
                out_all_sets.append(out_evidence_set)
            out["evidence"] = out_all_sets
            f.write(json.dumps(out))
            f.write("\n")


def convert_examples_for_inference(fever_path: str, out_path: str):
    """
    DPR inferences takes a TSV file with the quesetion and answers.
    In this case, we only care about inputting a question.
    """
    examples = read_jsonlines(fever_path)
    with open(out_path, "w") as f:
        for ex in tqdm(examples):
            claim = ex["claim"].replace("\t", " ")
            f.write(f"{claim}\t[]\n")


def convert_wiki(tsv_path: str, map_path: str):
    db = WikiDatabase()
    id_to_page_sent = {}
    with open(tsv_path, "w") as f:
        f.write("id\ttext\ttitle\n")
        idx = 1
        for page_proto in tqdm(db.get_all_pages()):
            page_proto = WikipediaDump.FromString(page_proto)
            title = page_proto.title.replace("\t", " ")
            for sent_id, sent_proto in page_proto.sentences.items():
                text = sent_proto.text.replace("\t", " ")
                f.write(f"{idx}\t{text}\t{title}\n")
                id_to_page_sent[str(idx)] = [title, sent_id]
                idx += 1

    with open(map_path, "w") as f:
        json.dump(id_to_page_sent, f)


def convert_wiki_to_kotlin_json(out_path: str):
    db = WikiDatabase()
    with open(out_path, "w") as f:
        for page_proto in tqdm(db.get_all_pages()):
            page_proto = WikipediaDump.FromString(page_proto)
            out = {
                "id": page_proto.id,
                "title": page_proto.title,
                "text": page_proto.text,
            }
            sentences = {}
            for idx, sent in page_proto.sentences.items():
                sentences[idx] = sent.text
            out["sentences"] = sentences
            f.write(json.dumps(out))
            f.write("\n")


def score_evidence(fever_path: str, id_map_path: str, pred_path: str):
    examples = read_jsonlines(fever_path)
    with open(pred_path) as f:
        evidence_preds = json.load(f)

    with open(id_map_path) as f:
        id_to_page_sent = json.load(f)

    scores = []
    n_total = 0
    n_correct = 0
    n_title_correct = 0
    for ex, pred in zip(tqdm(examples), evidence_preds):
        if ex["label"] == c.NOT_ENOUGH_INFO:
            continue
        n_total += 1
        gold_evidence = set()
        gold_pages = set()
        for ev_set in ex["evidence"]:
            for _, _, page, sent_id in ev_set:
                gold_evidence.add((page, sent_id))
                gold_pages.add(page)

        rank = 1
        correct = False
        page_correct = False
        for evidence in pred["ctxs"]:
            ctx_id = evidence["id"]
            predicted_evidence = tuple(id_to_page_sent[ctx_id])
            if predicted_evidence[0] in gold_pages:
                page_correct = True
            if predicted_evidence in gold_evidence:
                correct = True
                n_correct += 1
                break
            rank += 1

        if correct:
            mrr = 1 / rank
            scores.append(mrr)

        if page_correct:
            n_title_correct += 1

    scores = np.array(scores)
    recall_100 = n_correct / n_total
    log.info(f"MRR: {scores.mean()}, % in MRR: {recall_100}")
    log.info(f"N Correct: {n_correct} Total: {n_total}")
    log.info(f"N Title Correct: {n_title_correct}")


def score_lucene_evidence(fever_path: str, pred_path: str):
    pass
