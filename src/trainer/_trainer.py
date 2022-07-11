from dataclasses import dataclass
import logging
import torch
import time
from typing import Optional, Iterable, Tuple, List, Dict, Union, Sequence
from pathlib import Path

from typeguard import check_argument_types
import humanfriendly

from model.abs_asr_model import AbsAsrModel
from torch.utils.data.dataloader import DataLoader
from src.schedulers.abs_scheduler import AbsScheduler
from iterators.abs_iter_factory import AbsIterFactory
from utils.set_all_random_seed import set_all_random_seed
from utils.recursive_op import recursive_average

# TODO sagemakerに適用させる。
device = device = torch.device("cuda")


@dataclass
class TrainerOptions:
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int


class Trainer:
    def __init__(self) -> None:
        raise RuntimeError("You cannot make instance of this class")

    @classmethod
    def run(
        self,
        model: AbsAsrModel,
        train_loader: AbsIterFactory,
        valid_loader: AbsIterFactory,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[AbsScheduler],
        options: TrainerOptions,
    ):
        assert check_argument_types()

        output_dir = Path(options.output_dir)
        # make stat epoch manageble
        start_epoch = 0

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, options.max_epoch):

            # count time
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{options.max_epoch}epoch started")
            set_all_random_seed(options.seed + iepoch)

            # train
            all_steps_are_invalid = self.train_one_epoch(
                model=model,
                iterator=train_loader.build_iter(iepoch),
                optimizer=optimizer,
                scheduler=scheduler,
            )
            # validate
            self.validate_one_epoch(
                moded=model,
                iterator=valid_loader.build_iter(iepoch),
            )

            # LR scheduler step
            scheduler.step()

            # 3. Report the results
            # logging.info(reporter.log_message())
            # if trainer_options.use_matplotlib:
            #     reporter.matplotlib_plot(output_dir / "images")
            # if train_summary_writer is not None:
            #     reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
            #     reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
            # if trainer_options.use_wandb:
            #     reporter.wandb_log()

            # 4. save/update the checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    # "reporter": reporter.state_dict(),
                    "optimizers": optimizer.state_dict(),
                    "schedulers": scheduler.state_dict()
                    if scheduler is not None
                    else {},
                    # "scaler": scaler.state_dict() if scaler is not None else None,
                },
                output_dir / "checkpoint.pth",
            )

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break
            # TODO 8. Check early stopping
            # if options.patience is not None:
            #     if reporter.check_early_stopping(
            #         options.patience, *options.early_stopping_criterion
            #     ):
            #         break

        else:
            logging.info(f"The training was finished at {options.max_epoch} epochs ")

    @classmethod
    def train_one_epoch(
        self,
        model: AbsAsrModel,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[AbsScheduler],
        options: TrainerOptions,
    ):

        model.train()

        all_step_are_invalid = True
        iterator_stop = torch.tensor(0).to("cuda")

        for iiter, (utt_id, batch) in enumerate(iterator):
            retval = model(**batch)
            if isinstance(retval, dict):
                loss = retval["loss"]
                stats = retval["stats"]
                weight = retval["weight"]
                optim_idx = retval.get("optim_idx")
            else:
                loss, stats, weight = retval
                optim_idx = None

            stats = {k: v for k, v in stats.items() if v is not None}
            if options.ngpu > 1:
                # Apply weighted averaging for loss and stats
                loss = (loss * weight.type(loss.dtype)).sum()
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, False)
                # Now weight is summation over all workers
                loss /= weight

        pass

    @classmethod
    def validate_one_epoch(
        self,
        model: AbsAsrModel,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        options: TrainerOptions,
    ):

        pass
