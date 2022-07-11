import argparse
from dataclasses import dataclass
import logging
import random
from multiprocessing.spawn import get_command_line, prepare
from typing import Callable, Sequence
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import datetime

import numpy as np
import torch
import fairscale

import humanfriendly
import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type

from model.abs_asr_model import AbsAsrModel
from model.asr_model import ASRModel
from model.encoder.abs_encoder import AbsEncoder
from model.encoder.transformer_encoder import TransformerEncoder
from model.decoder.abs_decoder import AbsDecoder
from model.decoder.transformer_decoder import TransformerDecoder
from model.frontend.abs_frontend import AbsFrontend
from model.frontend.default import DefaultFrontend
from model.preencoder.abs_preencoder import AbsPreEncoder
from model.postencoder.abs_postencoder import AbsPostEncoder
from model.ctc import CTC
from model.specaug.abs_specaug import AbsSpecAug
from model.specaug.specaug import SpecAug
from model.layers.abs_normalize import AbsNormalize
from model.layers.global_mvn import GlobalMVN
from schedulers.abs_scheduler import AbsScheduler

from trainer.preprocessor import CommonPreprocessor

from trainer.classes import optim_classes, scheduler_classes
from trainer.dataset import SequenceDataset
from trainer.trainer import Trainer
from iterators.abs_iter_factory import AbsIterFactory
from iterators.sequence_iter_factory import SequenceIterFactory
from samplers.build_batch_sampler import BATCH_TYPES, build_batch_sampler
from trainer.collate_fn import CommonCollateFn
from utils import utils


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


class ASRTask:

    trainer = Trainer

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def set_logging_config(self, args: argparse.Namespace):
        # logging setting
        logfile = args.logfile_path
        logging.basicConfig(filename=logfile, level=args.log_level)

    @classmethod
    def build_model(self, args: argparse.Namespace) -> AbsAsrModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        frontend = DefaultFrontend(**args.frontend_conf)
        input_size = frontend.output_size()

        # 2. data augmentation for spectrogram
        specaug = SpecAug(**args.specaug_conf)

        # 3. Normalization layer
        normalizer = GlobalMVN(**args.normalize_conf)

        # 4-1. Pre-encoder input block
        preencoder = None

        # 4-2. Encoder
        encoder = TransformerEncoder(**args.encoder_conf)

        # 4-3. Post-encoder block
        postencoder = None

        # 5-1. Decoder
        decoder = TransformerDecoder(**args.decoder_conf)
        joint_network = None
        # 6. CTC
        ctc = CTC(**args.ctc_conf)

        # 7. Build model
        model = ASRModel(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalizer,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            **args.model_conf,
        )
        return model

    @classmethod
    def build_optimizers(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        optim_class = optim_classes.get(args.optim)  # TODO set parser optim
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        else:
            optim = optim_class(model.parameters(), **args.optim_conf)
        return optim

    @classmethod
    def build_scheduler(
        self, args: argparse.Namespace, optim: torch.optim.Optimizer
    ) -> AbsScheduler:

        name = getattr(args, "scheduler")
        conf = getattr(args, "scheduler_conf")
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(f"must be one of {list(scheduler_classes)}: {name}")
            scheduler = cls_(optim, **conf)
        else:
            scheduler = None

        return scheduler

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=None,
                non_linguistic_symbols=None,
                text_cleaner=None,
                g2p_type=None,
                # # NOTE(kamo): Check attribute existence for backward compatibility
                # rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                # rir_apply_prob=args.rir_apply_prob
                # if hasattr(args, "rir_apply_prob")
                # else 1.0,
                # noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                # noise_apply_prob=args.noise_apply_prob
                # if hasattr(args, "noise_apply_prob")
                # else 1.0,
                # noise_db_range=args.noise_db_range
                # if hasattr(args, "noise_db_range")
                # else "13_15",
                # short_noise_thres=args.short_noise_thres
                # if hasattr(args, "short_noise_thres")
                # else 0.5,
                # speech_volume_normalize=args.speech_volume_normalize
                # if hasattr(args, "rir_scp")
                # else None,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_sequence_iter_factory(
        self, args: argparse.Namespace, mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        if mode == "train":
            train_data_path_and_name_and_type = [
                (args.train_dump_path + "/wav.scp", "speech", "sound"),
                (args.train_dump_path + "/text", "text", "text"),
            ]
            data_path_and_name_and_type = train_data_path_and_name_and_type
            preprocess = self.build_preprocess_fn(args, train=True)
            shape_file = args.train_shape_file
            num_iter_per_epoch = args.num_iters_per_epoch

        elif mode == "valid":
            valid_data_path_and_name_and_type = [
                (args.valid_dump_path + "/wav.scp", "speech", "sound"),
                (args.valid_dump_path + "/text", "text", "text"),
            ]
            data_path_and_name_and_type = valid_data_path_and_name_and_type
            preprocess = self.build_preprocess_fn(args, train=False)
            shape_file = args.valid_shape_file
            num_iter_per_epoch = None

        dataset = SequenceDataset(
            data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=preprocess,
            max_cache_size=args.max_cache_size,
            max_cache_fd=args.max_cache_fd,
        )

        utt2category_file = None

        batch_sampler = build_batch_sampler(
            type=args.batch_type,
            shape_files=shape_file,
            fold_lengths=args.fold_length,
            batch_size=args.batch_size,
            batch_bins=args.batch_bins,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
            drop_last=False,
            min_batch_size=1,
            utt2category_file=utt2category_file,
        )

        batches = list(batch_sampler)

        bs_list = [len(batch) for batch in batches]

        logging.info(f"[{mode}] dataset:\n{dataset}")
        logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logging.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=num_iter_per_epoch,
            shuffle=mode == "train",
            num_workers=args.num_workers,
            collate_fn=CommonCollateFn(float_pad_value=0.0, int_pad_value=-1),
            pin_memory=args.ngpu > 0,
        )

    @classmethod
    def main(self, args: argparse.Namespace = None):
        assert check_argument_types()
        self.set_logging_config(args)

        logging.info(get_command_line())
        logging.info(f"args:{args}")

        # 0. set random seed
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        # 1. Build model
        model = self.build_model(args=args)
        if not isinstance(model, AbsAsrModel):
            raise RuntimeError(
                f"model must inherit {AbsAsrModel.__name__}, but got {type(model)}"
            )
        model = model.to(
            dtype=getattr(torch, args.train_dtype),
            device="cuda" if args.ngpu > 0 else "cpu",
        )

        # 3. Build optimizer
        optimizer = self.build_optimizers(args, model=model)

        # 4. Build schedulers
        scheduler = self.build_scheduler(args, optimizer)

        logging.info(utils.pytorch_cudnn_version())
        logging.info(utils.model_summary(model))
        logging.info(f"optimizer:{optimizer}")
        logging.info(f"scheduler:{scheduler}")

        if args.only_show_model is True:
            logging.info("only show model is true!!!")
            return

        # 5. TODO Loads pre-training model

        # 6. build factory
        # TODO make distributed_option

        train_factory = self.build_sequence_iter_factory(args=args, mode="train")

        valid_factory = self.build_sequence_iter_factory(args=args, mode="valid")

        # 7. start training
        options = self.trainer.build_options(args=args)
        self.trainer.run(
            model=model,
            optimizers=[optimizer],
            schedulers=[scheduler],
            train_iter_factory=train_factory,
            valid_iter_factory=valid_factory,
            trainer_options=options,
        )


from sagemaker.estimator import Estimator

Estimator()