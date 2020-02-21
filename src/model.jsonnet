function(lr=0.0001) {
  dataset_reader: {
    type: 'fever',
  },
  train_data_path: 'data/train.jsonl',
  validation_data_path: 'data/shared_task_dev.jsonl',
  model: {
    type: 'claim_only',
    dropout: 0.25,
    pool: 'mean',
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['text', 'num_tokens']],
    batch_size: 32,
  },
  trainer: {
    optimizer: {
      type: 'adam',
      lr: lr,
    },
    validation_metric: '+accuracy',
    num_serialized_models_to_keep: 1,
    num_epochs: 50,
    patience: 2,
    cuda_device: 0,
  },
}