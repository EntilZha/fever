from typing import Dict, Any
import os
import comet_ml
from allennlp.training.trainer import EpochCallback, Trainer


@EpochCallback.register('comet_epoch_callback')
class CometEpochCallback(EpochCallback):
    def __init__(self, project_name: str):
        self._project_name = project_name
        self._experiment = None
    
    def __call__(self, trainer: Trainer, metrics: Dict[str, Any], epoch: int, is_master: bool):
        if epoch == -1:
            self._experiment = comet_ml.Experiment(project_name=self._project_name)
            config_file = os.environ.get('MODEL_CONFIG_FILE')
            if config_file is not None:
                self._experiment.log_asset(config_file)
        else:
            for key, val in metrics.items():
                self._experiment.log_metric(key, val, epoch=epoch)