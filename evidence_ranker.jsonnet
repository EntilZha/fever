local transformer = "roberta-base";

function(lr=1e-5) {
  dataset_reader: {
    type: 'serene.dataset.FeverReader',
    transformer: transformer,
    include_evidence: true
  },
  train_data_path: 'data/train.jsonl',
  validation_data_path: 'data/shared_task_dev.jsonl',
  model: {
    type: 'serene.model.FeverEvidenceRanker',
    dropout: 0.1,
    transformer: transformer
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      sorting_keys: ['claim_tokens'],
      batch_size: 8
    },
  },
  trainer: {
    optimizer: {
      type: 'huggingface_adamw',
      lr: lr,
      correct_bias: true
    },
    validation_metric: '+accuracy',
    num_epochs: 50,
    patience: 1,
    cuda_device: 0,
  },
}
