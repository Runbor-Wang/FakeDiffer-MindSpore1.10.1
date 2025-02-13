import yaml
import argparse
from trainer import ExpTester


def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/FakeDiffer.yml",
                        help="Specify the path of configuration file to be used.")
    parser.add_argument('--display', '-d', action="store_true", default=False, help='Display some images.')
    return parser.parse_args()


if __name__ == '__main__':
    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    trainer = ExpTester(config, stage="Test")
    trainer.test(display_images=arg.display)
