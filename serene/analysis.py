from typing import List
import os
import itertools

from comet_ml import ExistingExperiment
from plotnine import (
    ggplot,
    aes,
    geom_tile,
    geom_text,
    facet_wrap,
    theme,
    scale_fill_cmap,
)
import plotly.graph_objects as go
import pandas as pd
from pedroai.io import read_jsonlines

from serene.constants import config


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
        fever_ids = []
        label_combo = []
        for ex, p in zip(examples, preds):
            t_label = ex["label"]
            p_label = p["pred_readable"]
            true_labels.append(t_label)
            pred_labels.append(p_label)
            label_combo.append(f"T: {t_label} P: {p_label}")
            probs = p["probs"]
            pred_probs.append(probs)
            idx_to_label[p["preds"]] = p_label
            label_to_idx[p_label] = p["preds"]
            fever_ids.append(ex["id"])

        label_names = [idx_to_label[idx] for idx in range(3)]
        labels = [_label_to_vector(label_to_idx[l]) for l in true_labels]

        self.labels = labels
        self.true_labels = true_labels
        self.pred_labels = pred_labels
        self.label_combo = label_combo
        self.combo_set = set(label_combo)
        self.pred_probs = pred_probs
        self.label_names = label_names
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx
        self.fever_ids = fever_ids
        self.df = pd.DataFrame(
            {
                "true_label": pd.Categorical(
                    self.true_labels, categories=self.label_names, ordered=True
                ),
                "pred_label": pd.Categorical(
                    self.pred_labels, categories=self.label_names, ordered=True
                ),
                "fever_id": self.fever_ids,
                "combo_label": pd.Categorical(
                    self.label_combo, categories=list(self.combo_set), ordered=True
                ),
            }
        )
        self.df["n"] = 1


def log_confusion_matrix(experiment_id: str, fever_path: str, pred_path: str):
    conf_data = ConfusionData(fever_path, pred_path)
    experiment = ExistingExperiment(previous_experiment=experiment_id)
    experiment.log_confusion_matrix(
        conf_data.labels, conf_data.pred_probs, labels=conf_data.label_names
    )


def plot_confusion_matrix(fever_path: str, pred_path: str, out_path: str):
    conf_data = ConfusionData(fever_path, pred_path)
    df = conf_data.df
    df = df.groupby(["true_label", "pred_label"]).sum().reset_index()
    p = (
        ggplot(df, aes(x="pred_label", y="true_label", fill="n"))
        + geom_tile(aes(width=0.95, height=0.95))
        + geom_text(aes(label="n"), size=10)
    )
    p.save(out_path)


def plot_all_confusion_matrices(fold: str, out_path: str):
    frames = []
    by_name = {}
    model_names = [
        "claim_only+claim_only",
        "gold+gold",
        "dpr_neg_0+batch",
        "dpr_neg_1+batch",
        "dpr_neg_2+batch",
        "dpr_neg_3+batch",
    ]
    for name in model_names:
        conf_data = ConfusionData(
            config["fever"][fold]["examples"], config["pipeline"][name][fold]["preds"]
        )
        conf_data.df["model"] = name
        by_name[name] = conf_data.df
        frames.append(conf_data.df)
    df = pd.concat(frames)
    df["model"] = pd.Categorical(df["model"], categories=model_names, ordered=True)
    pdf = df.groupby(["true_label", "pred_label", "model"]).sum().reset_index()
    p = (
        ggplot(pdf)
        + aes(x="pred_label", y="true_label", fill="n")
        + facet_wrap("model")
        + geom_tile(aes(width=0.95, height=0.95))
        + geom_text(aes(label="n"), size=10)
        + scale_fill_cmap("cividis")
        + theme(figure_size=(15, 10))
    )
    p.save(out_path)


def _colorize(n):
    if "T: SUPPORTS" in n:
        return "green"
    elif "T: NOT ENOUGH INFO" in n:
        return "blue"
    else:
        return "red"


def plot_confusion_flow(fold: str, out_dir: str):
    fever_path = config["fever"][fold]["examples"]
    frames = []
    by_name = {}
    model_names = [
        "claim_only+claim_only",
        "gold+gold",
        "dpr_neg_0+batch",
        "dpr_neg_1+batch",
        "dpr_neg_2+batch",
        "dpr_neg_3+batch",
    ]
    for name in model_names:
        conf_data = ConfusionData(fever_path, config["pipeline"][name][fold]["preds"])
        conf_data.df["model"] = name
        by_name[name] = conf_data.df
        frames.append(conf_data.df)
    df = pd.concat(frames)
    df["model"] = pd.Categorical(df["model"], categories=model_names, ordered=True)
    confusion_flow_models = ["claim_only+claim_only", "dpr_neg_0+batch", "gold+gold"]
    for source_model, target_model in itertools.combinations(confusion_flow_models, 2):
        claim_df = by_name[source_model].set_index("fever_id")
        best_df = by_name[target_model].set_index("fever_id")
        join_df = claim_df.join(best_df, how="left", lsuffix="_source")
        link_df = (
            join_df.groupby(["combo_label_source", "combo_label"])
            .sum("n")
            .reset_index()
            .dropna()
        )
        cross_cats = list(
            set(join_df.combo_label.dtype.categories.values.tolist())
            | set(join_df.combo_label_source.dtype.categories.values.tolist())
        )
        label_to_idx = {c: idx for idx, c in enumerate(cross_cats)}
        nodes = 2 * cross_cats
        fig = go.Figure(
            data=[
                go.Sankey(
                    node={
                        "label": nodes,
                        "color": [_colorize(n) for n in nodes],
                        "line": {"color": "black", "width": 0.5},
                        "pad": 15,
                        "thickness": 20,
                    },
                    link={
                        "source": link_df["combo_label_source"]
                        .map(lambda x: label_to_idx[x])
                        .values,
                        "target": link_df["combo_label"]
                        .map(lambda x: label_to_idx[x] + len(label_to_idx))
                        .values,
                        "value": link_df["n"].values,
                    },
                )
            ]
        )
        fig.update_layout(
            height=1200, width=800, title=f"{source_model} -> {target_model} Evidence"
        )
        short_source = source_model.split("+")[0]
        short_target = target_model.split("+")[0]
        fig.write_image(
            os.path.join(out_dir, f"confusion_flow-{short_source}-{short_target}.png")
        )

