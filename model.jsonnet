local transformer = "roberta-base";

function(lr=1e-4) {
  dataset_reader: {
    type: 'fever',
    transformer: transformer
  },
  train_data_path: 'data/train.jsonl',
  validation_data_path: 'data/shared_task_dev.jsonl',
  model: {
    type: 'claim_only',
    dropout: 0.5,
    pool: 'cls',
    transformer: transformer
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      sorting_keys: ['claim_tokens'],
      batch_size: 32
    },
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    validation_metric: '+accuracy',
    num_epochs: 50,
    patience: 1,
    cuda_device: 0,
  },
}
