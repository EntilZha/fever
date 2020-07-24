import typer

# comet_ml needs to be imported before anything else
import comet_ml
from allennlp import commands

from serene import wiki
from serene import data

# Init the logger
from serene.util import get_logger


log = get_logger(__name__)


app = typer.Typer()


@app.command()
def train(serialization_dir: str, model_config: str):
    commands.train.train_model_from_file(
        parameter_filename=model_config,
        serialization_dir=serialization_dir,
        force=True,
    )


@app.command()
def hyper(serialization_dir: str, model_config: str):
    commands.train.train_model_from_file(
        parameter_filename=model_config,
        serialization_dir=serialization_dir,
        force=True,
    )


@app.command()
def wiki_to_proto():
    """
    Convert the fever wikipedia dump to a sqlite db
    where each row is stored as a protobuf
    """
    wiki.build_wiki_db()


@app.command()
def fever_to_dpr_train(
    fever_path: str, out_path: str, hard_neg_path: str = None, nth_best_neg: int = 1
):
    """
    Convert Fever examples for DPR training. If hard_neg_path is defined,
    then add these in as well.
    """
    data.convert_examples_for_dpr_training(
        fever_path=fever_path,
        out_path=out_path,
        hard_neg_path=hard_neg_path,
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
def score_dpr_preds(fever_path: str, id_map_path: str, dpr_path: str):
    """
    Score the DPR Predictions
    """
    data.score_evidence(fever_path, id_map_path, dpr_path)


@app.command()
def convert_examples_to_kotlin_json(fever_path: str, out_path: str):
    """
    Convert fever examples to the Json used in the Lucene code written in Kotlin
    at github.com/entilzha/fever-lucene
    """
    data.convert_examples_to_kotlin_json(fever_path, out_path)


@app.command()
def convert_wiki_to_kotlin_json(out_path: str):
    """
    Convert wikipedia to the Json used in the Lucene code written in Kotlin
    at github.com/entilzha/fever-lucene
    """
    data.convert_wiki_to_kotlin_json(out_path)


@app.command()
def score_lucene_evidence(fever_path: str, out_path: str):
    """
    Score lucene predictions
    """
    data.score_lucene_evidence(fever_path, out_path)


if __name__ == "__main__":
    app()
