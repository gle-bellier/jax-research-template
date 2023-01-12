
# Standard libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from flax.core import apply
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict

# JAX/Flax
# If you run this code on Colab, remember to install flax and optax
# !pip install --quiet --upgrade flax optax
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

# Logging with Tensorboard or Weights and Biases
# If you run this code on Colab, remember to install pytorch_lightning
# !pip install --quiet --upgrade pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    rng: Any = None


class TrainerModule:
    def __init__(self,
                 model_class: nn.Module,
                 model_hparams: Dict[str, Any],
                 optimizer_hparams: Dict[str, Any],
                 example_input: Any,
                 logger_params: Dict[str, Any],
                 seed: int = 42,
                 enable_progress_bar: bool = True,
                 debug: bool = False,
                 check_val_every_n_epoch: int = 1,
                 **kwargs) -> None:

        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.example_input = example_input
        self.seed = seed
        self.logger_params = logger_params
        self.debug = debug
        self.enable_progress_bar = enable_progress_bar
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # set list of hyperparameters to save

        self.config = {
            "model_class": model_class.__name__,
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "logger_params": logger_params,
            "debug": debug,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "seed": seed
        }
        self.config.update(kwargs)

        # create empty model
        self.model = self.model_class(**self.model_hparams)
        # print model summary
        self.print_tabulate(example_input)

        # init trainer parts
        self.__init_logger(logger_params)
        self.__create_jitted_functions()
        self.__init_model(example_input)

    def __init_logger(self, logger_params: Dict[str, Any]) -> None:

        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get('log_dir', None)
        if not log_dir:
            base_log_dir = logger_params.get('base_log_dir', 'checkpoints/')
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            if 'logger_name' in logger_params:
                log_dir = os.path.join(log_dir, logger_params['logger_name'])
            version = None
        else:
            version = ''
        # Create logger object
        logger_type = logger_params.get('logger_type', 'TensorBoard').lower()
        if logger_type == 'tensorboard':
            self.logger = TensorBoardLogger(save_dir=log_dir,
                                            version=version,
                                            name='')
        elif logger_type == 'wandb':
            self.logger = WandbLogger(name=logger_params.get('project_name', None),
                                      save_dir=log_dir,
                                      version=version,
                                      config=self.config)
        else:
            assert False, f'Unknown logger type \"{logger_type}\"'
        # Save hyperparameters
        log_dir = self.logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, 'hparams.json')):
            os.makedirs(os.path.join(log_dir, 'metrics/'), exist_ok=True)
            with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
        self.log_dir = log_dir

    def __init_model(self, example_input: Any) -> None:

        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        example_input = [example_input] if not isinstance(
            example_input, (list, tuple)) else example_input

        # init model
        variables = self.__run_model_init(example_input, init_rng)
        # create default state. Optimizer will be initialized later
        self.
        state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            rng=model_rng,
            tx=None,
            opt_state=None)

    def __run_model_init(self, example_input: Any, init_rng: Any) -> Dict:

        return self.model.init(init_rng, *example_input, train=True)

    def print_tabulate(self, example_input: Any) -> None:
        print(self.model.tabulate(random.PRNGKey(0), *example_input, train=True))

    def __init_optimizer(self,
                         num_epochs: int,
                         num_step_per_epoch: int) -> None:

        hparams = copy(self.optimizer_hparams)

        # define optimizer
        opt_class = opt.adamw
        lr = hparams.pop("lr", 1e-3)
        warmup = hparams.pop("warmup", 0)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs*num_step_per_epoch),
            end_value=0.01 * lr
        )

        # add gradient clipping
        transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 1.0))]

        optimizer = optax.chain(*transf, opt_class(lr_schedule, **hparams))
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       tx=optimizer,
                                       rng=self.state.rng)

    def create_jitted_functions(self) -> None:
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            print('Skipping jitting due to debug=True')
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    def create_step_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState, Dict]],
                                             Callable[[TrainState, Any], Tuple[TrainState, Dict]]]:
        def train_step(state: TrainState,
                       batch: Any):
            metrics = {}
            return state, metrics

        def eval_step(state: TrainState,
                      batch: Any):
            metrics = {}
            return metrics
        raise NotImplementedError
