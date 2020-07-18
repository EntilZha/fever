import typer
# comet_ml needs to be imported before anything else
import comet_ml
from allennlp import commands

from serene import wiki
from serene import dpr
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
    wiki.build_wiki_db()


@app.command()
def fever_to_dpr(fever_path: str, out_path: str):
    dpr.convert_examples(fever_path, out_path)


@app.command()
def wiki_to_dpr(tsv_path: str, map_path: str):
    dpr.convert_wiki(tsv_path, map_path)


if __name__ == "__main__":
    app()
