# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/8 17:02
# @Function      : Basic Trainer class inherited from transformers.trainer
from datetime import datetime
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from newsrec.dataset import UserInteractionDataset
from newsrec.utils import collate_fn, init_model_class, get_project_root, compute_metrics


class NRSTrainer(Trainer):
    def __init__(self, **kwargs):
        # add date and time to run_name
        callbacks = [EarlyStoppingCallback(early_stopping_patience=kwargs.get("early_stopping_patience", 3))]
        train_dataset = UserInteractionDataset(split="train", **kwargs)
        eval_dataset = UserInteractionDataset(split="dev", **kwargs)
        model_name = kwargs.get("model_name", "NRMSRSModel")  # NRMSRSModel/BaseNRS
        run_name = kwargs.get("run_name", f"{model_name}_{datetime.now().strftime(r'%y%m%d_%H%M')}")

        def model_init(trial):
            model_config = kwargs.copy()
            if trial is not None:
                from optuna import Trial
                if isinstance(trial, dict):
                    model_config.update(trial)
                elif isinstance(trial, Trial):
                    model_config.update(trial.params)
            return init_model_class(model_name, model_config)

        accelerator_config = {
            "split_batches": False, "dispatch_batches": True, "even_batches": True, "use_seedable_sampler": True
        }
        train_args_dict = {
            "output_dir": f"{get_project_root()}/output_dir/{model_name}/",
            "logging_dir": f"{get_project_root()}/output_dir/{model_name}/logs/",
            "run_name": run_name,
            "accelerator_config": accelerator_config,
            "log_level": "info",
            "learning_rate": 0.001,
            "per_device_train_batch_size": 64,
            "gradient_accumulation_steps": 4,
            "per_device_eval_batch_size": 256,
            "logging_steps": 50,
            "save_steps": 1000,
            "save_total_limit": 3,
            "evaluation_strategy": "epoch",  # epoch/steps
            "save_strategy": "epoch",
            "eval_steps": 100,
            "num_train_epochs": 1,
            "metric_for_best_model": "eval_monitor_metric",
            "label_names": ["label"],
            "report_to": "all",  # "all": all installed integrations, default use wandb
            # "auto_find_batch_size": True,
            "group_by_length": False,
            "skip_memory_metrics": True,  # report memory metrics will slow train and evaluation speed
            "greater_is_better": True,
            "load_best_model_at_end": True,
            "remove_unused_columns": False,
            "disable_tqdm": False,
        }
        train_args_dict.update({k: v for k, v in kwargs.items() if hasattr(TrainingArguments, k)})
        train_args = TrainingArguments(**kwargs.get("train_args", train_args_dict))
        metric_list = kwargs.get("metric_list", ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"])
        self.ignore_keys_for_eval = ["extra_output"]
        super(NRSTrainer, self).__init__(
            None, train_args, collate_fn, train_dataset, eval_dataset,
            model_init=model_init, callbacks=callbacks,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric_list=metric_list)
        )


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }


if __name__ == "__main__":
    override_kwargs = {
        "model_name": "NPARSModel",  # NRMSRSModel/BaseNRS/LSTURRSModel/NAMLRSModel/NPARSModel
        "user_embed_method": "concat",  # init/concat
        "text_feature": ["title", "body"],  # ["title", "abstract", "body"]
        "cat_feature": ["category", "subvert"],  # ["category", "subvert"]
        "subset_name": "small", "max_history_size": 50, "title_len": 30, "abstract_len": 0, "body_len": 70,
        "use_cached_feature_mapper": True, "fast_evaluation": False,
        "per_device_eval_batch_size": 32, "news_batch_size": 1024, "user_batch_size": 256
    }
    trainer = NRSTrainer(**override_kwargs)
    # import time
    # start = time.time()
    trainer.evaluate(ignore_keys=trainer.ignore_keys_for_eval)
    # print(f"evaluation time: {time.time() - start} seconds")
    trainer.train(ignore_keys_for_eval=trainer.ignore_keys_for_eval)
    # best_trials = trainer.hyperparameter_search(
    #     n_trials=4, backend="optuna", hp_space=optuna_hp_space, direction="maximize"
    # )
