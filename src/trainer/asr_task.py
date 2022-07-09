import argparse
import logging
import random
from multiprocessing.spawn import get_command_line
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

from trainer.classes import optim_classes, scheduler_classes
from utils import utils


class ASRTask:
    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def set_logging_config(self, args: argparse.Namespace):
        # logging setting
        logfile = args.logfile_path
        logging.basicConfig(filename=logfile, encoding="utf-8", level=args.log_level)

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
    ) -> List[object]:

        name = getattr(args, f"scheduler")
        conf = getattr(args, f"scheduler_conf")
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(f"must be one of {list(scheduler_classes)}: {name}")
            scheduler = cls_(optim, **conf)
        else:
            scheduler = None

        return scheduler

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

        if args.only_show_model:
            logging.info("only show model is true!!!")
            return
