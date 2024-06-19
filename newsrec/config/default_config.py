# default training arguments for all models
# check https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments
TRAINING_ARGS = {
    "log_level": "info",  # ‘debug’, ‘info’, ‘warning’, ‘error’ and ‘critical’, plus a ‘passive’
    "logging_steps": 50,  # log every logging_steps
    "save_strategy": "epoch",  # epoch/steps, default saved by epoch
    "save_steps": 1000,  # if save_strategy is steps, save every save_steps
    "save_total_limit": 1,  # limit the total amount of checkpoints, delete the older checkpoints
    "evaluation_strategy": "epoch",  # epoch/steps, default saved by epoch
    "eval_steps": 1000,  # if evaluation_strategy is steps, evaluate every eval_steps
    "load_best_model_at_end": True,  # load the best model found so far at the end of training
    "metric_for_best_model": "eval_monitor_metric",  # metric name for monitoring the best model
    "greater_is_better": True,  # whether the metric is better when greater
    "skip_memory_metrics": True,  # report memory metrics will slow train and evaluation speed
    "disable_tqdm": False,  # disable tqdm progress bar
    "learning_rate": 0.0002,  # initial learning rate for AdamW optimizer, PLM usually use 1e-5
    "gradient_accumulation_steps": 1,  # number of updates steps to accumulate before performing a backward pass
    "per_device_train_batch_size": 64,  # batch size per device during training
    "per_device_eval_batch_size": 128,  # batch size per device for evaluation
    "num_train_epochs": 10,  # total number of training epochs to perform
    "label_names": ["label"],  # label names for ground truth
    "report_to": "all",  # "all": all installed integrations, default use wandb
    "group_by_length": False,  # Whether group together samples of roughly the same length for the training dataset
    "remove_unused_columns": False,  # Whether to automatically remove the columns unused by the model forward method.
    "accelerator_config": {  # accelerator config for multi-device training
        "split_batches": False, "dispatch_batches": False, "even_batches": True, "use_seedable_sampler": True
    }
}

# setup some default arguments of loading dataset and model
DEFAULT_ARGS = {
    "running_mode": "train_only",  # running mode can be "train_only" or "hyper_search"
    "n_trials": 100,  # if running_mode is "hyper_search", number of trials for hyperparameter search
    "subset_name": "small",  # Subset name of the dataset, can be "small" or "large", default using MIND
    # processed dataset: https://huggingface.co/datasets/Rui98/mind
    "text_feature": ["title"],  # text feature used to train the model, can be ["title", "abstract", "body"]
    "title_len": 30, "abstract_len": 0, "body_len": 0,  # default using title with max length 30
    "cat_feature": [],  # category feature can be ["category", "subvert"], set category_dim if using category features
    "entity_feature": [],  # entity feature can be ["entity", "abstract"], "entity" is the entity of the news title
    # default using pretrained entity vector by Wiki Data
    "max_history_size": 50,  # max reading history news number for each user
    "use_flash_att": False,  # whether to use flash attention for attention models
    "use_cached_feature_mapper": True,  # whether to use cached feature mapper, change to False if feature changed
    "use_cached_news_graph": True, "directed": False,  # whether the graph is directed
    "early_stopping_patience": 3,  # early stopping patience for training
    "fast_evaluation": False, "news_batch_size": 1024, "user_batch_size": 256,  # fast evaluation arguments
    "loss": "categorical_loss",  # loss functions categorical_loss/nce_loss
    "metric_list": ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"],  # metric list for evaluation
}
