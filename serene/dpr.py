"""
Export data to DPR format
"""
import json
from tqdm import tqdm
from pedroai.io import read_jsonlines
import numpy as np

from serene.wiki_db import WikiDatabase
from serene.protos.fever_pb2 import WikipediaDump, FeverExample
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


def convert_examples_to_protos(fever_path: str, out_path: str):
    examples = read_jsonlines(fever_path)
    with open(out_path, "wb") as f:
        for ex in tqdm(examples):
            proto_example = FeverExample()
            proto_example.label = ex["label"]
            proto_example.id = ex["id"]
            proto_example.claim = ex["claim"]
            for ev_set in ex["evidence"]:
                proto_ev_set = proto_example.evidences.add()
                for _, _, title, sentence_id in ev_set:
                    if title is not None and sentence_id is not None:
                        proto_ev = proto_ev_set.evidence.add()
                        proto_ev.wikipedia_url = title
                        proto_ev.sentence_id = sentence_id
            proto_str = proto_example.SerializeToString()
            f.write(proto_str)
            f.write(b"@@@")


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


def score_evidence(fever_path: str, id_map_path: str, pred_path: str):
    examples = read_jsonlines(fever_path)
    with open(pred_path) as f:
        evidence_preds = json.load(f)

    with open(id_map_path) as f:
        id_to_page_sent = json.load(f)

    scores = []
    n_correct = 0
    n_title_correct = 0
    for ex, pred in zip(tqdm(examples), evidence_preds):
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
    recall_100 = n_correct / len(examples)
    log.info(f"MRR: {scores.mean()}, % in MRR: {recall_100}")
    log.info(f"N Correct: {n_correct} Total: {len(examples)}")
    log.info(f"N Title Correct: {n_title_correct}")

