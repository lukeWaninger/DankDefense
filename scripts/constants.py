import os
from pathlib import Path
import yaml

BUCKET = 'dank-defense'
FEATURES_KEY = 'features'
JOBS_KEY = 'jobs'
TAG_KEY = 'Project'
PROJECT_NAME = 'DankDefense'
AMI = 'ami-0c6415e46854ac2d6'
AWS_DEFAULT_REGION = 'us-east-1'
MAX_RETRIES = 5

config_schema = """
job_name:
    type: string
features:
    type: array
    items:
        type: string
        uniqueItems: true
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
            type: array
            items:
                type: object
                properties:
                    name:
                        type: string
                    values:
                        type: array
                        uniqueItems: true
"""
SCHEMA = yaml.load(config_schema)

with open(os.path.join(Path.home(), 'DD_SECRETS'), 'r') as f:
    cf = f.readlines()

    SECRETS = {
        k: v for k, v in [
            list(map(lambda x: x.strip(), l.split('=')))
            for l in cf if not l.startswith('#')
    ]}
