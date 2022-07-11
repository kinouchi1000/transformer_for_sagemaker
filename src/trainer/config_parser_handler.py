import argparse
import datetime
import humanfriendly

from typeguard import check_return_type
from utils.nested_dict_action import NestedDictAction
from trainer.classes import scheduler_classes, optim_classes
from samplers.build_batch_sampler import BATCH_TYPES
from utils.types import (
    str2triple_str,
    int_or_none,
    strtobool,
    str_or_none,
    str_or_int,
    str2pair_str,
    str2bool,
)


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
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
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

        # batch sampler related
        batchsampler_group = parser.add_argument_group("BatchSampler related")
        batchsampler_group.add_argument(
            "--num_iters_per_epoch",
            type=int_or_none,
            default=None,
            help="Restrict the number of iterations for training per epoch",
        )
        batchsampler_group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training. Used if batch_type='unsorted',"
            " 'sorted', or 'folded'.",
        )
        batchsampler_group.add_argument(
            "--batch_bins",
            type=int,
            default=1000000,
            help="The number of batch bins. Used if batch_type='length' or 'numel'",
        )
        batchsampler_group.add_argument(
            "--valid_batch_bins",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_bins is used",
        )

        batchsampler_group.add_argument(
            "--train_shape_file", type=str, action="append", default=[]
        )
        batchsampler_group.add_argument(
            "--valid_shape_file", type=str, action="append", default=[]
        )

        group = parser.add_argument_group("Sequence iterator related")
        _batch_type_help = ""
        for key, value in BATCH_TYPES.items():
            _batch_type_help += f'"{key}":\n{value}\n'
        group.add_argument(
            "--batch_type",
            type=str,
            default="folded",
            choices=list(BATCH_TYPES),
            help=_batch_type_help,
        )
        # group.add_argument(
        #     "--valid_batch_type",
        #     type=str_or_none,
        #     default=None,
        #     choices=list(BATCH_TYPES) + [None],
        #     help="If not given, the value of --batch_type is used",
        # )
        group.add_argument("--fold_length", type=int, action="append", default=[])
        group.add_argument(
            "--sort_in_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort the samples in each mini-batches by the sample "
            'lengths. To enable this, "shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort mini-batches by the sample lengths",
        )
        group.add_argument(
            "--multiple_iterator",
            type=str2bool,
            default=False,
            help="Use multiple iterator mode",
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
        trainer_group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
            nargs="+",
            default=[
                ("train", "loss", "min"),
                ("valid", "loss", "min"),
                ("train", "acc", "max"),
                ("valid", "acc", "max"),
            ],
            help="The criterion used for judging of the best model. "
            'Give a pair referring the phase, "train" or "valid",'
            'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
        )
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
        trainer_group.add_argument(
            "--use_amp",
            type=str2bool,
            default=False,
            help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
        )
        trainer_group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in each epochs at the "
            "training phase. If None is given, it is decided according the number "
            "of training samples automatically .",
        )
        trainer_group.add_argument(
            "--use_matplotlib",
            type=str2bool,
            default=True,
            help="Enable matplotlib logging",
        )
        trainer_group.add_argument(
            "--use_tensorboard",
            type=str2bool,
            default=True,
            help="Enable tensorboard logging",
        )
        trainer_group.add_argument(
            "--use_wandb",
            type=str2bool,
            default=False,
            help="Enable wandb logging",
        )
        trainer_group.add_argument(
            "--wandb_project",
            type=str,
            default=None,
            help="Specify wandb project",
        )
        trainer_group.add_argument(
            "--wandb_id",
            type=str,
            default=None,
            help="Specify wandb id",
        )
        trainer_group.add_argument(
            "--wandb_entity",
            type=str,
            default=None,
            help="Specify wandb entity",
        )
        trainer_group.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Specify wandb run name",
        )
        trainer_group.add_argument(
            "--wandb_model_log_interval",
            type=int,
            default=-1,
            help="Set the model log period",
        )
        trainer_group.add_argument(
            "--detect_anomaly",
            type=str2bool,
            default=False,
            help="Set torch.autograd.set_detect_anomaly",
        )

        dataset_group = parser.add_argument_group("Dataset related")
        _data_path_and_name_and_type_help = (
            "Give three words splitted by comma. It's used for the training data. "
            "e.g. '--train_data_path_and_name_and_type some/path/a.scp,foo,sound'. "
            "The first value, some/path/a.scp, indicates the file path, "
            "and the second, foo, is the key name used for the mini-batch data, "
            "and the last, sound, decides the file type. "
            "This option is repeatable, so you can input any number of features "
            "for your task. Supported file types are as follows:\n\n"
        )
        dataset_group.add_argument(
            "--train_dump_path", type=str, default="dump/raw/train_nodup_sp/"
        )
        dataset_group.add_argument(
            "--valid_dump_path", type=str, default="dump/raw/train_dev/"
        )
        dataset_group.add_argument(
            "--max_cache_size",
            type=humanfriendly.parse_size,
            default=0.0,
            help="The maximum cache size for data loader. e.g. 10MB, 20GB.",
        )
        dataset_group.add_argument(
            "--max_cache_fd",
            type=int,
            default=32,
            help="The maximum number of file descriptors to be kept "
            "as opened for ark files. "
            "This feature is only valid when data type is 'kaldi_ark'.",
        )
        assert check_return_type(parser)
        return parser
