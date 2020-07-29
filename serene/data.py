"""
Utilities for data such as
* Exporting to DPR/Lucene formats
* Scoring outputs of DPR/Lucene
"""
from typing import Set, Dict, List, Tuple, Optional, Iterable
import json
from dataclasses import dataclass

from comet_ml import ExistingExperiment
from plotnine import ggplot, aes, geom_tile, geom_text
from tqdm import tqdm
from pedroai.io import read_jsonlines, read_json, write_jsonlines
import numpy as np
from pydantic import BaseModel
import pandas as pd

from serene.wiki_db import WikiDatabase
from serene.protos.fever_pb2 import WikipediaDump
from serene.util import get_logger
from serene.constants import config
from serene import constants as c


log = get_logger(__name__)


class LuceneDocument(BaseModel):
    wikipedia_url: str
    sentence_id: int
    text: str
    score: float


class LucenePredictions(BaseModel):
    claim_id: int
    documents: List[LuceneDocument]


def parse_lucene_predictions(path: str) -> Dict[int, List[LuceneDocument]]:
    preds = {}
    with open(path) as f:
        for line in f:
            claim_pred: LucenePredictions = LucenePredictions.parse_raw(line)
            preds[claim_pred.claim_id] = claim_pred.documents

    return preds


def convert_examples_for_dpr_training(
    *, fever_path: str, out_path: str, hard_neg_path: str = None, nth_best_neg: int = 1
):
    """
    DPR trains based on a json formatted file where each entry contains
    and example along with all of its positive/negative contexts.
    
    If nth_best_neg is defined, then skip that many negatives before taking one.
    If none are found, then skip the hard negative
    
    Hard negatives will be in the lucene output format
    """
    log.info(
        f"fever_path: {fever_path} out_path: {out_path} hard_neg_path: {hard_neg_path}"
    )
    examples = read_jsonlines(fever_path)
    db = WikiDatabase()
    if hard_neg_path is None:
        hard_negs = {}
    else:
        hard_negs = parse_lucene_predictions(hard_neg_path)
    dpr_examples = []
    n_negatives = 0
    for ex in tqdm(examples):
        claim_label = ex["label"]
        if claim_label == c.NOT_ENOUGH_INFO:
            continue
        evidence_sets = ex["evidence"]
        flattened_evidence = []
        # Note the gold, so they don't become negatives
        gold_pairs = set()
        for ev_set in evidence_sets:
            for _, _, page, sent_id in ev_set:
                if page is not None and sent_id is not None:
                    # Do not change underscores to spaces
                    maybe_evidence = db.get_page_sentence(page, sent_id)
                    gold_pairs.add((page, sent_id))
                    if maybe_evidence is not None:
                        flattened_evidence.append(
                            {"title": page, "text": maybe_evidence}
                        )
        claim_id = ex["id"]

        if hard_neg_path is None:
            hard_negative_ctxs = []
        else:
            # Don't check errors here, I'd rather this crash
            # and point out an unexpected absence of a negative
            claim_negatives = hard_negs[claim_id]
            nth_neg = 1
            for neg in claim_negatives:
                # TODO: After rerunning DB generation, remove this
                neg_wikipedia_url = neg.wikipedia_url.replace(" ", "_")
                if (neg_wikipedia_url, neg.sentence_id) not in gold_pairs:
                    if nth_best_neg == nth_neg:
                        hard_negative_ctxs = [
                            {
                                "title": neg_wikipedia_url,
                                "text": neg.text,
                                "score": neg.score,
                                "sentence_id": neg.sentence_id,
                            }
                        ]
                        n_negatives += 1
                        break
                    else:
                        nth_neg += 1

        dpr_examples.append(
            {
                "question": ex["claim"],
                "positive_ctxs": flattened_evidence,
                "negative_ctxs": [],
                "hard_negative_ctxs": hard_negative_ctxs,
                "claim_id": claim_id,
                "claim_label": claim_label,
            }
        )
    log.info(f"N Total: {len(dpr_examples)} N Negatives: {n_negatives}")
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


def convert_examples_for_dpr_inference(*, fever_path: str, out_path: str):
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


@dataclass
class GoldEvidence:
    pages: Set[str]
    sentences: Set[Tuple[str, int]]


def create_gold_evidence(examples: Iterable) -> Dict[int, GoldEvidence]:
    id_to_gold = {}
    for ex in examples:
        if ex["label"] == c.NOT_ENOUGH_INFO:
            continue
        gold_evidence = set()
        gold_pages = set()
        for ev_set in ex["evidence"]:
            for _, _, page, sent_id in ev_set:
                gold_evidence.add((page, sent_id))
                gold_pages.add(page)
        # This should already be an int, but make sure it is
        id_to_gold[int(ex["id"])] = GoldEvidence(gold_pages, gold_evidence)
    return id_to_gold


def score_dpr_evidence(fever_path: str, id_map_path: str, pred_path: str):
    examples = read_jsonlines(fever_path)
    with open(pred_path) as f:
        evidence_preds = json.load(f)

    with open(id_map_path) as f:
        id_to_page_sent = json.load(f)

    scores = []
    n_total = 0
    n_correct = 0
    n_title_correct = 0
    n_recall_5 = 0
    id_to_gold = create_gold_evidence(examples)
    for ex, pred in zip(tqdm(examples), evidence_preds):
        if ex["label"] == c.NOT_ENOUGH_INFO:
            continue
        n_total += 1
        gold = id_to_gold[ex["id"]]
        gold_evidence = gold.sentences
        gold_pages = gold.pages

        rank = 1
        correct = False
        page_correct = False
        for evidence in pred["ctxs"]:
            ctx_id = evidence["id"]
            page, sentence_id = id_to_page_sent[ctx_id]
            # TODO: Undo spaces, can remove after full rerun
            page = page.replace(" ", "_")
            predicted_evidence = (page, sentence_id)
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
            if rank <= 5:
                n_recall_5 += 1

        if page_correct:
            n_title_correct += 1

    np_scores = np.array(scores)
    recall_100 = n_correct / n_total
    log.info(
        f"MRR Mean: {np_scores.mean():.3f}, MRR Median: {np.median(np_scores)} % in MRR: {recall_100:.3f}"
    )
    log.info(f"Recall@5: {n_recall_5 / n_total:.3f}")
    log.info(f"N Correct: {n_correct} Total: {n_total}")
    log.info(f"N Title Correct: {n_title_correct}")


def score_lucene_evidence(fever_path: str, pred_path: str):
    examples = read_jsonlines(fever_path)
    id_to_gold = create_gold_evidence(examples)
    predictions = {int(k): v for k, v in read_json(pred_path)["documents"].items()}
    scores = []
    n_correct_pages = 0
    n_correct = 0
    n_recall_5 = 0
    for ex_id, gold in id_to_gold.items():
        # Implicitly, NOT ENOUGH INFO are already filtered out
        rank = 1
        correct_sent = False
        correct_page = False
        for doc in predictions[ex_id]:
            page = doc["wikipedia_url"]
            # TODO: Can remove space replacement after full rerun
            page = page.replace(" ", "_")
            sent_id = doc["sentence_id"]
            if page in gold.pages:
                correct_page = True
            if (page, sent_id) in gold.sentences:
                correct_sent = True
                break
            rank += 1

        if correct_sent:
            n_correct += 1
            scores.append(1 / rank)
            if rank <= 5:
                n_recall_5 += 1

        if correct_page:
            n_correct_pages += 1

    np_scores = np.array(scores)
    n_total = len(id_to_gold)
    recall_100 = n_correct / n_total
    log.info(
        f"MRR Mean: {np_scores.mean():.3f}, MRR Median: {np.median(np_scores)} % in MRR: {recall_100:.3f}"
    )
    log.info(f"Recall@5: {n_recall_5 / n_total:.3f}")
    log.info(f"N Correct: {n_correct} Total: {n_total}")
    log.info(f"N Title Correct: {n_correct_pages}")


def convert_evidence_for_claim_eval(
    fever_path: str, id_map_path: str, pred_path: str, out_path: str
):
    """
    Convert evidence predictions from the DPR model into the same format used by
    the training/dev data of the Fever task. This makes it easy to test a model
    that consumes claim plus evidence to predict the inference label (eg support/refute/nei)
    """
    examples = read_jsonlines(fever_path)
    evidence_preds = read_json(pred_path)
    id_to_page_sent = read_json(id_map_path)
    out_examples = []
    for ex, preds in zip(tqdm(examples), evidence_preds):
        claim_id = ex["id"]
        claim = ex["claim"]
        # We only care about the top ranked evidence for this
        ctx_id = preds["ctxs"][0]["id"]
        page, sentence_id = id_to_page_sent[ctx_id]
        # TODO: Undo spaces, can remove after full rerun
        page = page.replace(" ", "_")
        predicted_evidence = (page, sentence_id)
        evidence_set = [[None, None, page, sentence_id]]
        out_examples.append(
            {
                "id": int(claim_id),
                "claim": claim,
                "verifiable": ex["verifiable"],
                "label": ex["label"],
                "evidence": [evidence_set],
            }
        )
    if len(examples) != len(out_examples):
        raise ValueError(
            f"Length of examples does not match: {len(examples)} vs {len(out_examples)}"
        )
    write_jsonlines(out_path, out_examples)


def _label_to_vector(label: int) -> List[int]:
    zeros = [0, 0, 0]
    zeros[label] = 1
    return zeros


class ConfusionData:
    def __init__(self, fever_path: str, pred_path: str):
        examples = read_jsonlines(fever_path)
        preds = read_jsonlines(pred_path)
        if len(examples) != len(preds):
            raise ValueError("Mismatch length of examples and predictions")

        idx_to_label = {}
        label_to_idx = {}
        true_labels = []
        pred_labels = []
        pred_probs = []
        for ex, p in zip(examples, preds):
            true_labels.append(ex["label"])
            pred_labels.append(p["pred_readable"])
            probs = p["probs"]
            pred_probs.append(probs)
            idx_to_label[p["preds"]] = p["pred_readable"]
            label_to_idx[p["pred_readable"]] = p["preds"]

        label_names = [idx_to_label[idx] for idx in range(3)]
        labels = [_label_to_vector(label_to_idx[l]) for l in true_labels]

        self.labels = labels
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.pred_probs = pred_probs
        self.label_names = label_names
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx


def log_confusion_matrix(experiment_id: str, fever_path: str, pred_path: str):
    conf_data = ConfusionData(fever_path, pred_path)
    experiment = ExistingExperiment(previous_experiment=experiment_id)
    experiment.log_confusion_matrix(
        conf_data.labels, conf_data.pred_probs, labels=conf_data.label_names
    )


def plot_confusion_matrix(fever_path: str, pred_path: str, out_path: str):
    conf_data = ConfusionData(fever_path, pred_path)
    df = pd.DataFrame(
        {
            "true_labels": pd.Categorical(
                conf_data.true_labels, categories=conf_data.label_names, ordered=True
            ),
            "pred_labels": pd.Categorical(
                conf_data.pred_labels, categories=conf_data.label_names, ordered=True
            ),
        }
    )
    df["n"] = 1
    df = df.groupby(["true_labels", "pred_labels"]).sum().reset_index()
    p = (
        ggplot(df, aes(x="pred_labels", y="true_labels", fill="n"))
        + geom_tile(aes(width=0.95, height=0.95))
        + geom_text(aes(label="n"), size=10)
    )
    p.save(out_path)
