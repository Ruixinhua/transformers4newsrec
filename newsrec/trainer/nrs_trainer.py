# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/8 17:02
# @Function      : Basic Trainer class inherited from transformers.trainer
from datetime import datetime
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from omegaconf import OmegaConf
from newsrec.data import *
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
            "split_batches": False, "dispatch_batches": False, "even_batches": True, "use_seedable_sampler": True
        }
        train_args_dict = {
            "output_dir": f"{get_project_root()}/output_dir/{run_name}/",
            "logging_dir": f"{get_project_root()}/output_dir/{run_name}/logs/",
            "run_name": run_name,
            "accelerator_config": accelerator_config,
            "log_level": "info",
            "learning_rate": 0.001,
            "gradient_accumulation_steps": 1,
            "per_device_eval_batch_size": 256,
            "logging_steps": 50,
            "save_steps": 1000,
            "save_total_limit": 3,
            "evaluation_strategy": "epoch",  # epoch/steps
            "save_strategy": "epoch",
            "eval_steps": 100,
            "num_train_epochs": 5,
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


if __name__ == "__main__":
    override_kwargs = {
        "model_name": "NRMSRSModel",  # NRMSRSModel/BaseNRS/LSTURRSModel/NAMLRSModel/NPARSModel/GLORYRSModel
        # "user_embed_method": "concat",  # init/concat
        "s": ["title"],  # ["title", "abstract", "body"]
        "early_stopping_patience": 3, "directed": False,
        # "cat_feature": ["category", "subvert"],  # ["category", "subvert"]
        "subset_name": "small", "max_history_size": 50, "title_len": 30, "abstract_len": 0, "body_len": 0,
        # "entity_feature": ["entity"],  # ["entity", "abstract"]
        "use_cached_feature_mapper": True, "fast_evaluation": False, "use_cached_news_graph": True,
        "per_device_eval_batch_size": 128, "news_batch_size": 1024, "user_batch_size": 256, "use_flash_att": False,
        "per_device_train_batch_size": 64, "num_train_epochs": 5, "learning_rate": 0.001,
        "loss": "categorical_loss"  # categorical_loss/nce_loss
    }
    config = OmegaConf.create(override_kwargs)
    config.merge_with(OmegaConf.from_cli())
    print(config)

    trainer = NRSTrainer(**config)
    running_mode = config.get("running_mode", "train")
    if running_mode == "train":
        trainer.train(ignore_keys_for_eval=trainer.ignore_keys_for_eval)
    elif running_mode == "hyper_search":
        def set_hp_name(trial):
            run_name = config.get("run_name", f"{config.get('model_name')}")
            for k, v in trial.params.items():
                if isinstance(v, float):
                    v = f"{v:.6f}"
                run_name += f"_{v}"
            return run_name

        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.001, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                         [16, 32, 64, 128]),
            }
        best_trials = trainer.hyperparameter_search(
            n_trials=config.get("n_trials", 10), backend="optuna", hp_space=optuna_hp_space, direction="maximize",
            hp_name=set_hp_name
        )
        print(best_trials)
    # trainer.evaluate(ignore_keys=trainer.ignore_keys_for_eval)
