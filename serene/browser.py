# pylint: disable-all
from typing import Dict
import random
import typer
import pandas as pd
import streamlit as st
from pedroai.io import read_jsonlines, read_json
from tqdm import tqdm
import plotly.graph_objects as go

from serene import constants as c
from serene.constants import config
from serene.data import create_gold_evidence, GoldEvidence
from serene.wiki_db import WikiDatabase
from serene.analysis import ConfusionData, colorize


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


model_names = [
    "claim_only+claim_only",
    "gold+gold",
    "dpr_neg_0+batch",
    "dpr_neg_1+batch",
    "dpr_neg_2+batch",
    "dpr_neg_3+batch",
]


@st.cache
def load_confusion(fold):
    fever_path = config["fever"][fold]["examples"]
    frames = []
    by_name = {}
    for name in model_names:
        conf_data = ConfusionData(fever_path, config["pipeline"][name][fold]["preds"])
        conf_data.df["model"] = name
        by_name[name] = conf_data.df
        frames.append(conf_data.df)
    df = pd.concat(frames)
    df["model"] = pd.Categorical(df["model"], categories=model_names, ordered=True)
    return df, by_name


@st.cache
def load_fever(fold: str):
    fever_path = config["fever"][fold]["examples"]
    return {ex["id"]: ex for ex in read_jsonlines(fever_path)}


@st.cache()
def load_model_evidence(name: str):
    retriever_name = name.split("+")[0]
    fever_path = config["retriever"][retriever_name]["dev"]["verify_examples"]
    evidence = {}
    for ex in read_jsonlines(fever_path):
        _, _, page, sent_id = ex["evidence"][0][0]
        evidence[ex["id"]] = page, sent_id
    return evidence


def fetch_evidence_text(db: WikiDatabase, evidence_map: Dict, fever_id: int):
    page, sent = evidence_map[fever_id]
    if page is None or sent is None:
        return None
    else:
        return db.get_page_sentence(page, sent)


@app.command()
def confusion(fold: str = "dev"):
    df, by_name = load_confusion(fold)
    fever_examples = load_fever(fold)
    db = WikiDatabase()
    st.write("Fever Lucene Predictions")
    source_model = st.sidebar.selectbox("Source Model", model_names, 0)
    target_model = st.sidebar.selectbox("Target Model", model_names, 1)
    source_evidence = load_model_evidence(source_model)
    target_evidence = load_model_evidence(target_model)
    claim_df = by_name[source_model].set_index("fever_id")
    best_df = by_name[target_model].set_index("fever_id")
    join_df = claim_df.join(best_df, how="left", lsuffix="_source", rsuffix="_target")
    link_df = (
        join_df.groupby(["combo_label_source", "combo_label_target"])
        .sum("n_target")
        .reset_index()
        .dropna()
    )
    source_cats = list(set(join_df.combo_label_source.dtype.categories.values.tolist()))
    target_cats = list(set(join_df.combo_label_target.dtype.categories.values.tolist()))
    cross_cats = list(set(target_cats) | set(source_cats))
    label_to_idx = {c: idx for idx, c in enumerate(cross_cats)}
    nodes = 2 * cross_cats
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": nodes,
                    "color": [colorize(n) for n in nodes],
                    "line": {"color": "black", "width": 0.5},
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": link_df["combo_label_source"]
                    .map(lambda x: label_to_idx[x])
                    .values,
                    "target": link_df["combo_label_target"]
                    .map(lambda x: label_to_idx[x] + len(label_to_idx))
                    .values,
                    "value": link_df["n_target"].values,
                },
            )
        ]
    )
    fig.update_layout(
        height=1200, width=800, title=f"{source_model} -> {target_model} Evidence"
    )
    st.plotly_chart(fig)
    source_group = st.selectbox("Source Group", source_cats)
    target_group = st.selectbox("Target Group", target_cats)
    step = 50
    offset = st.sidebar.number_input(
        "Offset (Size: %d)" % len(join_df),
        min_value=0,
        max_value=int(len(join_df)) - step,
        value=0,
        step=step,
    )
    n_max = st.sidebar.number_input("N Max", min_value=0, value=100)
    columns = [
        "combo_label_source",
        "combo_label_target",
    ]
    display_df = join_df[
        (join_df.combo_label_source == source_group)
        & (join_df.combo_label_target == target_group)
    ]
    display_df = (
        display_df.iloc[offset : offset + n_max][columns].astype("object").reset_index()
    )
    display_df["claim"] = display_df["fever_id"].map(
        lambda x: fever_examples[x]["claim"]
    )
    display_df["source_page"] = display_df["fever_id"].map(
        lambda x: source_evidence[x][0]
    )
    display_df["source_evidence"] = display_df["fever_id"].map(
        lambda x: fetch_evidence_text(db, source_evidence, x)
    )
    display_df["target_page"] = display_df["fever_id"].map(
        lambda x: target_evidence[x][0]
    )
    display_df["target_evidence"] = display_df["fever_id"].map(
        lambda x: fetch_evidence_text(db, target_evidence, x)
    )
    st.table(display_df)


if __name__ == "__main__":
    try:
        app()
    except SystemExit as se:
        if se.code != 0:
            raise
