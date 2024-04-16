# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/8 17:02
# @Function      : Basic Trainer class inherited from transformers.trainer
from datetime import datetime
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from accelerate import DataLoaderConfiguration
from newsrec.dataset import UserInteractionDataset
from newsrec.utils import collate_fn, init_model_class, get_project_root, compute_metrics


class NRSTrainer(Trainer):
    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "NRMSRSModel")  # NRMSRSModel/BaseNRS
        default_model_config = {
            "text_feature": ["title", "abstract", "body"]
        }
        model_config = kwargs.get("model_config", default_model_config)
        model = init_model_class(model_name, model_config)
        # add date and time to run_name
        run_name = kwargs.get("run_name", f"{model_name}_{datetime.now().strftime(r'%y%m%d_%H%M')}")
        callbacks = [EarlyStoppingCallback(early_stopping_patience=kwargs.get("early_stopping_patience", 3))]
        train_dataset = UserInteractionDataset(split="train", **kwargs)
        eval_dataset = UserInteractionDataset(split="dev", **kwargs)
        accelerator_config = {
            "split_batches": False, "dispatch_batches": True, "even_batches": True, "use_seedable_sampler": True
        }
        default_train_args = {
            "output_dir": f"{get_project_root()}/output_dir/{model_name}/",
            "logging_dir": f"{get_project_root()}/output_dir/{model_name}/logs/",
            "run_name": run_name,
            "accelerator_config": accelerator_config,
            "log_level": "info",
            "learning_rate": 0.001,
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 32,
            "logging_steps": 50,
            "save_steps": 1000,
            "save_total_limit": 3,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "num_train_epochs": 1,
            "metric_for_best_model": "eval_monitor_metric",
            "label_names": ["label"],
            "report_to": "all",  # "all": all installed integrations, default use wandb
            # "auto_find_batch_size": True,
            "skip_memory_metrics": True,  # report memory metrics will slow train and evaluation speed
            "greater_is_better": True,
            "load_best_model_at_end": True,
            "remove_unused_columns": False,
            "disable_tqdm": False,
        }
        train_args = TrainingArguments(**kwargs.get("train_args", default_train_args))
        metric_list = kwargs.get("metric_list", ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"])
        self.ignore_keys_for_eval = ["extra_output"]
        super(NRSTrainer, self).__init__(
            model, train_args, collate_fn, train_dataset, eval_dataset, callbacks=callbacks,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric_list=metric_list)
        )


if __name__ == "__main__":
    trainer = NRSTrainer()
    # trainer.evaluate(ignore_keys=trainer.ignore_keys_for_eval)
    trainer.train(ignore_keys_for_eval=trainer.ignore_keys_for_eval)
