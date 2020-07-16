from allennlp.training.metrics import Metric


class Recall(Metric):
    def __init__(self):
        self._n_recalled = 0.0
        self._n_total = 0.0

    def __call__(self, predictions, gold_labels, mask = None):
        if mask is not None:
            raise ValueError('Not implemented')
        # Assuming gold labels and predictions are 1/0
        self._n_recalled += (predictions * gold_labels).sum().item()
        self._n_total += gold_labels.sum().item()
    
    def get_metric(self, reset: bool = False):
        if self._n_total > 0:
            recall = self._n_recalled / self._n_total
        else:
            recall = 0.0

        if reset:
            self.reset()
        
        return recall

    def reset(self):
        self._n_recalled = 0.0
        self._n_total = 0.0