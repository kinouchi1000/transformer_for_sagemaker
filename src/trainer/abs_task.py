"""Abstract task module."""
from abc import ABC
from abc import abstractmethod
import argparse
from dataclasses import dataclass
from distutils.version import LooseVersion
import functools
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type
import yaml

from model.abs_asr_model import AbsModel
from model.schedulers.abs_scheduler import AbsScheduler
from model.schedulers.noam_lr import NoamLR
from model.schedulers.warmup_lr import WarmupLR


if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    from torch.multiprocessing.spawn import ProcessContext
else:
    from torch.multiprocessing.spawn import SpawnContext as ProcessContext


optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    # sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)


try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if LooseVersion(torch_optimizer.__version__) < LooseVersion("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(radam=torch_optimizer.RAdam,)
    del torch_optimizer
except ImportError:
    pass

try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None


scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    noamlr=NoamLR,
    warmuplr=WarmupLR,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
)
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and given to DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args, train=True), ...)

        In many cases, you can use our common collate_fn.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as follows,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsModel:
        raise NotImplementedError

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        assert check_argument_types()

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        parser = argparse.ArgumentParser(
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

        parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )
        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )
        group.add_argument(
            "--dry_run",
            type=bool,
            default=False,
            help="Perform process without training",
        )
        group.add_argument(
            "--iterator_type",
            type=str,
            choices=["sequence", "chunk", "task", "none"],
            default="sequence",
            help="Specify iterator type",
        )

        group.add_argument("--output_dir", type=Optional[str], default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
            "This option makes sense only when attention-based model. "
            "We can also disable the attention plot by setting it 0",
        )

        cls.trainer.add_arguments(parser)
        cls.add_task_arguments(parser)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_optimizers(
        cls, args: argparse.Namespace, model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        if cls.num_optimizers != 1:
            raise RuntimeError(
                "build_optimizers() must be overridden if num_optimizers != 1"
            )

        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.sharded_ddp:
            if fairscale is None:
                raise RuntimeError("Requiring fairscale. Do 'pip install fairscale'")
            optim = fairscale.optim.oss.OSS(
                params=model.parameters(), optim=optim_class, **args.optim_conf
            )
        else:
            optim = optim_class(model.parameters(), **args.optim_conf)

        optimizers = [optim]
        return optimizers

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        for k in vars(args):
            if "-" in k:
                raise RuntimeError(f'Use "_" instead of "-": parser.get_parser("{k}")')

        required = ", ".join(
            f"--{a}" for a in args.required if getattr(args, a) is None
        )

        if len(required) != 0:
            parser = cls.get_parser()
            parser.print_help(file=sys.stderr)
            p = Path(sys.argv[0]).name
            print(file=sys.stderr)
            print(
                f"{p}: error: the following arguments are required: " f"{required}",
                file=sys.stderr,
            )
            sys.exit(2)
