from allennlp.predictors import Predictor
from serene.model import FeverVerifier
from serene.dataset import FeverReader


class FeverVerifierPredictor(Predictor):
    def _json_to_instance(self, json_dict):
        claim_text = json_dict["claim"]
        label = json_dict.get("label")
        claim_id = json_dict.get("id")
        evidence_text = json_dict.get("evidence_text")
        evidence = json_dict.get("evidence")
        return self._dataset_reader.text_to_instance(
            claim_text=claim_text,
            claim_id=claim_id,
            evidence_text=evidence_text,
            evidence=evidence,
        )
