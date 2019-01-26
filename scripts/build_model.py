"""
The "train -> validate -> predict -> submit" pipline script.
"""

import json

import jsonschema
import argparse
import yaml
from boto.s3.connection import S3Connection

import scripts.constants as constants

config_schema = """
type: object
properties:
    job_name:
        type: string
    features:
        type: list
        items:
            type: string
    validation:
        type: string
    predict:
        type: object
        properties:
            submit:
                type: boolean
    parameter_tuning:
        type: object
        properties:
            search_type:
                type: string
                enum:
                    - grid
                    - stagewise
            parameters:
                type: list
                items:
                    type: object
                    properties:
                        name:
                            type: string
                        values:
                            type: array
                            uniqueItems: true
"""


def load_config(bucket, config_name):
    try:
        bucket.download_file('dank-defense/configs/' + config_name, '/tmp/'+config_name)
    except:
        print("Failed to fetch config file from s3:", config_name)
        raise

    with open('/tmp/'+config_name) as config_file:
        schema = yaml.load(config_schema)
        config = json.load(config_file)

        jsonschema.validate(schema, config)
        return config


def main():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('config_name', type=str, help='name of the config file in s3')

    args = parser.parse_args()
    config_name = args.config_name

    conn = S3Connection(constants.AWS_KEY, constants.AWS_SECRET)
    bucket = conn.get_bucket(constants.BUCKET)

    config = load_config(bucket)


if __name__ == '__main__':
    pass
