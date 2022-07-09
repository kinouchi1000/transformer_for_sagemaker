from trainer.asr_task import ASRTask
from trainer.config_parser_handler import ConfigParserHandler


def main():
    parser = ConfigParserHandler.get_parser()
    args = parser.parse_args()

    ASRTask.main(args=args)


if __name__ == "__main__":
    main()
