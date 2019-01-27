"""
The "train -> validate -> predict -> submit" pipline script.
"""

import argparse

import scripts.constants as constants
import scripts.pipe as pipe


def main():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('config_name', type=str, help='name of the config file in s3')

    args = parser.parse_args()
    config_name = args.config_name

    config = pipe.download_config(config_name)


if __name__ == '__main__':
    pass
