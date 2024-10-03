# -*- coding: utf-8 -*-
# @Author        : Rui
# @Time          : 2024/4/8 17:02
# @Function      : Basic Trainer class inherited from transformers.trainer
import copy
import os

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from omegaconf import OmegaConf
from newsrec.config import TRAINING_ARGS, DEFAULT_ARGS
from newsrec.data import *
from newsrec.utils import collate_fn, init_model_class, get_project_root, compute_metrics, save_test_results


class NRSTrainer(Trainer):
    def __init__(self, **kwargs):
        # add date and time to run_name
        train_dataset = UserInteractionDataset(split="train", **kwargs)
        eval_dataset = UserInteractionDataset(split="dev", **kwargs)
        model_name = kwargs.get("model_name")  # NRMSRSModel/BaseNRS
        run_name = kwargs.get("run_name")
        if model_name is None:
            raise ValueError("model_name should be provided")

        def model_init(trial):
            model_config = kwargs.copy()
            if trial is not None:
                from optuna import Trial
                if isinstance(trial, dict):
                    model_config.update(trial)
                elif isinstance(trial, Trial):
                    model_config.update(trial.params)
            return init_model_class(model_name, model_config)

        train_args_dict = copy.deepcopy(TRAINING_ARGS)
        train_args_dict.update({
            "run_name": run_name,  # setup running name
            # setup logging directory
            "logging_dir": f"{get_project_root()}/output_dir/{run_name}/logs/",
        })
        train_args_dict.update(kwargs.get("train_args", {}))
        train_args_dict["output_dir"] = kwargs.get("output_dir", f"{get_project_root()}/output_dir/{run_name}/")
        train_args_dict.update({k: v for k, v in kwargs.items() if hasattr(TrainingArguments, k)})
        callbacks = []
        if train_args_dict["evaluation_strategy"] == "epoch":
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=kwargs.get("early_stopping_patience", 3)))
        train_args = TrainingArguments(**train_args_dict)
        self.ignore_keys_for_eval = ["extra_output"]
        super(NRSTrainer, self).__init__(
            None, train_args, collate_fn, train_dataset, eval_dataset,
            model_init=model_init, callbacks=callbacks,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric_list=kwargs.get("metric_list"))
        )


if __name__ == "__main__":
    config = OmegaConf.create(DEFAULT_ARGS)
    config.merge_with(OmegaConf.from_cli())
    config.setdefault("run_name", f"{config.get('model_name')}")
    trainer = NRSTrainer(**config)
    running_mode = config.get("running_mode")
    if running_mode == "train_only":
        trainer.train(ignore_keys_for_eval=trainer.ignore_keys_for_eval)
        test_results = save_test_results(trainer, config)
    elif running_mode == "hyper_search":
        def set_hp_name(trial):
            run_name = config.get("run_name", f"{config.get('model_name')}")
            for k, v in trial.params.items():
                if isinstance(v, float):
                    v = f"{v:.6f}"
                run_name += f"/{k}-{v}"
            return run_name

        def optuna_hp_space(trial):
            search_space_all = OmegaConf.load(f"{get_project_root()}/hyper_search/base_nrs.yaml")
            for path in os.listdir(f"{get_project_root()}/hyper_search"):
                if path.endswith(".yaml") and path != "base_nrs.yaml":
                    search_space_all.merge_with(OmegaConf.load(f"{get_project_root()}/hyper_search/{path}"))
            search_param_name = config.get("search_param_name", ["learning_rate"])
            params = {n: search_space_all[n] for n in search_param_name if n in search_space_all}
            search_space = {}
            for k, v in params.items():
                try:
                    search_space[v["name"]] = trial.suggest_categorical(**v)
                except TypeError:
                    search_space[v["name"]] = trial.suggest_float(**v)
            # for
            return search_space
        best_trials = trainer.hyperparameter_search(
            n_trials=config.get("n_trials"), backend="optuna", hp_space=optuna_hp_space, direction="maximize",
            hp_name=set_hp_name
        )
        os.makedirs(f"{trainer.args.output_dir}/hyper_search", exist_ok=True)
        torch.save(best_trials, f"{trainer.args.output_dir}/hyper_search/{config.get('run_name')}.pt")
    else:
        raise ValueError(f"Invalid running mode: {running_mode}, should be in ['train_only', 'hyper_search']")
