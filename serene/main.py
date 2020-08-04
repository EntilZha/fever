import json
import argparse
import typer

# comet_ml needs to be imported before anything else
import comet_ml
from allennlp import commands

from pedroai.io import safe_file

from serene import wiki
from serene import data
from serene.constants import config

# Init the logger
from serene.util import get_logger


log = get_logger(__name__)


app = typer.Typer()


@app.command()
def train(model_key: str):
    serialization_dir = config["verifier"][model_key]["serialization_dir"]
    overrides = json.dumps(config["verifier"][model_key].get("overrides", {}))
    model_config = config[model_key]["config"]
    log.info(f"Model Key: {model_key}")
    log.info(f"Overrides: {overrides}")
    log.info(f"Model Config: {model_config}")
    commands.train.train_model_from_file(
        parameter_filename=model_config,
        serialization_dir=serialization_dir,
        force=True,
        overrides=overrides,
    )


@app.command()
def evaluate(retriever_name: str, verifier_name: str, fold: str = "dev"):
    overrides = json.dumps(config["verifier"][verifier_name].get("overrides", {}))
    args = argparse.Namespace(
        archive_file=config["verifier"][verifier_name]["archive"],
        cuda_device=0,
        weights_file=None,
        overrides=overrides,
        batch_size=None,
        input_file=config["retriever"][retriever_name][fold]["verify_examples"],
        embedding_sources_mapping="",
        extend_vocab=False,
        batch_weight_key="",
        output_file=config["pipeline"][retriever_name + "+" + verifier_name][fold][
            "metrics"
        ],
    )
    commands.evaluate.evaluate_from_args(args)


@app.command()
def predict(
    retriever_name: str,
    verifier_name: str,
    fold: str = "dev",
    batch_size: int = 32,
    silent: bool = True,
):
    overrides = json.dumps(config["verifier"][verifier_name].get("overrides", {}))
    output_file = config["pipeline"][retriever_name + "+" + verifier_name][fold][
        "preds"
    ]
    args = argparse.Namespace(
        cuda_device=0,
        overrides=overrides,
        batch_size=batch_size,
        predictor="serene.predictor.FeverVerifierPredictor",
        dataset_reader_choice="validation",
        use_dataset_reader=False,
        archive_file=config["verifier"][verifier_name]["archive"],
        input_file=config["retriever"][retriever_name][fold]["verify_examples"],
        output_file=safe_file(output_file),
        weights_file=None,
        silent=silent,
    )
    commands.predict._predict(args)  # pylint: disable=protected-access


@app.command()
def wiki_to_proto():
    """
    Convert the fever wikipedia dump to a sqlite db
    where each row is stored as a protobuf
    """
    wiki.build_wiki_db()


@app.command()
def fever_to_dpr_train(
    fold: str, model_key: str, nth_best_neg: int,
):
    """
    Convert Fever examples for DPR training. If hard_neg_path is defined,
    then add these in as well.
    """
    data.convert_examples_for_dpr_training(
        fever_path=config["fever"][fold]["examples"],
        out_path=config["retriever"][model_key][fold]["hard_neg"],
        hard_neg_path=config["lucene"][fold]["preds"],
        nth_best_neg=nth_best_neg,
    )


@app.command()
def fever_to_dpr_inference(fever_path: str, out_path: str):
    """
    Convert Fever examples for DPR inference
    """
    data.convert_examples_for_dpr_inference(fever_path=fever_path, out_path=out_path)


@app.command()
def wiki_to_dpr(tsv_path: str, map_path: str):
    """
    Convert Wikipedia to the format for generating DPR dense embeddings
    """
    data.convert_wiki(tsv_path, map_path)


@app.command()
def score_dpr_preds(fold: str, model_key: str):
    """
    Score the DPR Predictions
    """
    data.score_dpr_evidence(
        config["fever"][fold]["examples"],
        config["dpr_id_map"],
        config["retriever"][model_key][fold]["evidence_preds"],
        config["retriever"][model_key][fold]["metrics"],
    )


@app.command()
def convert_examples_to_kotlin_json(fold: str):
    """
    Convert fever examples to the Json used in the Lucene code written in Kotlin
    at github.com/entilzha/fever-lucene
    """
    data.convert_examples_to_kotlin_json(
        config["fever"][fold]["examples"], config["fever"][fold]["kotlin_examples"]
    )


@app.command()
def convert_wiki_to_kotlin_json(out_path: str):
    """
    Convert wikipedia to the Json used in the Lucene code written in Kotlin
    at github.com/entilzha/fever-lucene
    """
    data.convert_wiki_to_kotlin_json(out_path)


@app.command()
def score_lucene_evidence(fold: str = "dev"):
    """
    Score lucene predictions
    """
    data.score_lucene_evidence(
        config["fever"][fold]["examples"],
        config["lucene"][fold]["preds"],
        config["lucene"][fold]["metrics"],
    )


@app.command()
def convert_dpr_evidence_to_fever(model_key: str, fold: str):
    data.convert_evidence_for_claim_eval(
        config["fever"][fold]["examples"],
        config["dpr_id_map"],
        config["retriever"][model_key][fold]["evidence_preds"],
        config["retriever"][model_key][fold]["verify_examples"],
    )


@app.command()
def log_confusion_matrix(experiment_id: str, fold: str, pred_file: str):
    data.log_confusion_matrix(
        experiment_id, config["fever"][fold]["examples"], pred_file
    )


@app.command()
def plot_confusion_matrix(retriever_name: str, verifier_name: str, fold="dev"):
    data.plot_confusion_matrix(
        config["fever"][fold]["examples"],
        config["pipeline"][retriever_name + "+" + verifier_name][fold]["preds"],
        config["pipeline"][retriever_name + "+" + verifier_name][fold]["confusion"],
    )


@app.command()
def plot_all_confusions(fold: str):
    data.plot_all_confusion_matrices(
        fold, safe_file(config["stats"]["dev"]["confusion_matrices"])
    )


if __name__ == "__main__":
    app()
