import argparse
import datetime

from typeguard import check_return_type
from utils.nested_dict_action import NestedDictAction
from trainer.classes import scheduler_classes, optim_classes


class ConfigParserHandler:
    def __init__(self) -> None:
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def get_parser(self) -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser(
            description="base parser for train",
        )

        parser.set_defaults(required=["output_dir"])

        # common conf
        common_group = parser.add_argument_group("Common configuration")

        common_group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )
        common_group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )
        common_group.add_argument(
            "--logfile_path",
            type=str,
            default=f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            help="The logging file path",
        )
        common_group.add_argument(
            "--iterator_type",
            type=str,
            choices=["sequence", "chunk", "task", "none"],
            default="sequence",
            help="Specify iterator type",
        )
        common_group.add_argument("--output_dir", type=str, default=None)
        common_group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        common_group.add_argument("--seed", type=int, default=0, help="Random seed")
        common_group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        common_group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
            "This option makes sense only when attention-based model. "
            "We can also disable the attention plot by setting it 0",
        )
        common_group.add_argument(
            "--only_show_model",
            type=bool,
            default=False,
            help="argument to show only model, don't run trianing",
        )
        asr_group = parser.add_argument_group("asr related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        asr_group.add_argument(
            "--token_list",
            type=str,
            default=None,
            help="A text mapping int-id to token",
        )

        asr_group.add_argument(
            "--token_type",
            type=str,
            default="char",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        asr_group.add_argument(
            "--bpemodel",
            type=str,
            default=None,
            help="The model file of sentencepiece",
        )
        asr_group.add_argument(
            "--model_conf",
            type=NestedDictAction,
            default=dict(),
            help="The keyword arguments for frontend",
        )

        # layer conf
        layer_group = parser.add_argument_group("layer ralated")
        layer_group.add_argument(
            "--frontend_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for frontend",
        )
        layer_group.add_argument(
            "--specaug_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for specaug",
        )
        layer_group.add_argument(
            "--normalize_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for normalize",
        )
        layer_group.add_argument(
            "--encoder_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for encoder",
        )
        layer_group.add_argument(
            "--decoder_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for decoder",
        )
        layer_group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for ctc",
        )

        # optimizer
        optim_group = parser.add_argument_group("optimizer related")
        optim_group.add_argument(
            "--optim",
            type=lambda x: x.lower(),
            default="adam",
            choices=list(optim_classes),
            help="The optimizer type",
        )
        optim_group.add_argument(
            "--optim_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for optimizer",
        )
        optim_group.add_argument(
            "--scheduler",
            type=str,
            default="warmuplr",
            choices=list(scheduler_classes) + [None],
            help="The Learning rate scheduler type",
        )
        optim_group.add_argument(
            "--scheduler_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for lr scheduler",
        )

        # trainer
        trainer_group = parser.add_argument_group("Trainer related")
        trainer_group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        trainer_group.add_argument(
            "--patience",
            type=int,
            default=None,
            help="Number of epochs to wait without improvement "
            "before stopping the training",
        )
        trainer_group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("valid", "loss"),
            help="The criterion used for the value given to the lr scheduler. "
            'Give a pair referring the phase, "train" or "valid",'
            'and the criterion name. The mode specifying "min" or "max" can '
            "be changed by --scheduler_conf",
        )
        trainer_group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("valid", "loss", "min"),
            help="The criterion used for judging of early stopping. "
            'Give a pair referring the phase, "train" or "valid",'
            'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        # trainer_group.add_argument(
        #     "--best_model_criterion",
        #     type=str2triple_str,
        #     nargs="+",
        #     default=[
        #         ("train", "loss", "min"),
        #         ("valid", "loss", "min"),
        #         ("train", "acc", "max"),
        #         ("valid", "acc", "max"),
        #     ],
        #     help="The criterion used for judging of the best model. "
        #     'Give a pair referring the phase, "train" or "valid",'
        #     'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
        # )
        trainer_group.add_argument(
            "--keep_nbest_models",
            type=int,
            nargs="+",
            default=[10],
            help="Remove previous snapshots excluding the n-best scored epochs",
        )
        trainer_group.add_argument(
            "--nbest_averaging_interval",
            type=int,
            default=0,
            help="The epoch interval to apply model averaging and save nbest models",
        )
        trainer_group.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient norm threshold to clip",
        )
        trainer_group.add_argument(
            "--grad_clip_type",
            type=float,
            default=2.0,
            help="The type of the used p-norm for gradient clip. Can be inf",
        )
        trainer_group.add_argument(
            "--grad_noise",
            type=bool,
            default=False,
            help="The flag to switch to use noise injection to "
            "gradients during training",
        )
        trainer_group.add_argument(
            "--accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation",
        )
        trainer_group.add_argument(
            "--no_forward_run",
            type=bool,
            default=False,
            help="Just only iterating data loading without "
            "model forwarding and training",
        )
        trainer_group.add_argument(
            "--resume",
            type=bool,
            default=False,
            help="Enable resuming if checkpoint is existing",
        )
        trainer_group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type for training.",
        )

        assert check_return_type(parser)
        return parser
