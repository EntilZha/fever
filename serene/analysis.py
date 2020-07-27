import random
import typer
import pandas as pd
import streamlit as st
from pedroai.io import read_jsonlines, read_json
from tqdm import tqdm

from serene import constants as c
from serene.data import create_gold_evidence, GoldEvidence
from serene.wiki_db import WikiDatabase


app = typer.Typer()


def _max_width_():
    max_width_str = f"max-width: 1000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    th {{
        text-align: left;
        font-size: 110%;
       
     }}
    tr:hover {{
        background-color: #ffff99;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()


class FeverPredictions:
    def __init__(self, fever_path: str, pred_path: str):
        self.db = WikiDatabase()
        self.examples = {ex["id"]: ex for ex in read_jsonlines(fever_path)}
        self.id_to_gold = create_gold_evidence(list(self.examples.values()))

        random.seed(42)
        predictions = list(read_json(pred_path)["documents"].items())
        random.shuffle(predictions)
        self.preds = {int(k): v for k, v in predictions}
        rows = []
        empty_evidence = GoldEvidence(set(), set())
        n_total = len(self.preds)
        for ex_id, docs in tqdm(self.preds.items()):
            example = self.examples[ex_id]
            claim_text = example["claim"]
            label = example["label"]
            if label == c.NOT_ENOUGH_INFO:
                gold = empty_evidence
            else:
                gold = self.id_to_gold[ex_id]

            gold_sentences = list(gold.sentences)
            if len(gold_sentences) == 0:
                gold_text = ""
                gold_page = ""
                gold_sent_id = -1
            else:
                ev = gold_sentences[0]
                gold_page = ev[0]
                gold_sent_id = int(ev[1])
                gold_text = self.db.get_page_sentence(gold_page, gold_sent_id)
            top_docs = docs[:5]
            for d in top_docs:
                wikipedia_url = d["wikipedia_url"]
                # TODO: Remove after full rerun without spaces
                wikipedia_url = wikipedia_url.replace(" ", "_")
                sent_id = d["sentence_id"]
                rows.append(
                    {
                        "fid": ex_id,
                        "wikipedia_url": wikipedia_url,
                        "sentence_id": sent_id,
                        "label": label,
                        "claim": claim_text,
                        "gold": (wikipedia_url, sent_id) in gold.sentences,
                        "text": self.db.get_page_sentence(wikipedia_url, sent_id),
                        "score": d["score"],
                        "gold_text": gold_text,
                        "gold_page": gold_page,
                        "gold_sent_id": gold_sent_id,
                    }
                )
        self.df = pd.DataFrame(
            rows,
            columns=[
                "fid",
                "label",
                "gold",
                "score",
                "wikipedia_url",
                "sentence_id",
                "claim",
                "text",
                "gold_text",
                "gold_page",
                "gold_sent_id",
            ],
        )


# @st.cache
def load_prediction_df(fever_path, pred_path):
    return FeverPredictions(fever_path, pred_path).df


@app.command()
def lucene(fever_path: str, pred_path: str):
    ALL = "ALL"
    df = load_prediction_df(fever_path, pred_path)
    st.write("Fever Lucene Predictions")
    label = st.sidebar.selectbox(
        "Fever label", [ALL, c.SUPPORTS, c.REFUTES, c.NOT_ENOUGH_INFO]
    )
    is_gold = st.sidebar.selectbox("Evidence", [ALL, "Gold", "Not Gold"])

    if label != ALL:
        df = df[df.label == label]

    if is_gold != ALL:
        if is_gold == "Gold":
            is_gold = 1
        else:
            is_gold = 0
        df = df[df.gold == is_gold]

    step = 50
    offset = st.sidebar.number_input(
        "Offset (Size: %d)" % len(df),
        min_value=0,
        max_value=int(len(df)) - step,
        value=0,
        step=step,
    )
    n_max = st.sidebar.number_input("N Max", min_value=0, value=100)

    st.table(df.iloc[offset : offset + n_max])


@app.command()
def dpr(fever_path: str, pred_path: str):
    # fd = FeverData(fever_path)
    pass


if __name__ == "__main__":
    try:
        app()
    except SystemExit as se:
        if se.code != 0:
            raise
