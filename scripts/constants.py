import os
from pathlib import Path
import yaml

BUCKET = 'dank-defense'
FEATURES_KEY = 'features'
JOBS_KEY = 'jobs'
TAG_KEY = 'Project'
PROJECT_NAME = 'DankDefense'
AMI = 'ami-0c6415e46854ac2d6'
AWS_DEFAULT_REGION = 'umport s-east-1'
MAX_RETRIES = 5
DATASET_KEYS = ['train', 'test', 'validate', 'validate_as_train']

config_schema = """
type: object
required:
    - job_name
    - features
    - model
    - task
    - tuning
properties:
    job_name:
        type: string
    features:
        type: array
        items:
            type: string
            uniqueItems: true
    model:
        type: object
        required:
            - name
            - parameters
        properties:
            name:
                type: string
                enum: 
                    - lightgbm
            parameters:
                type: object
    task:
        type: string
        enum: 
            - validate
            - tune
            - submit
            _ validate_submit
            _ tune_submit
    tuning:
        type: object
        required:
            - search_type
            - parameters
        properties:
            search_type:
                type: string
                enum:
                    - grid
                    - stage_wise
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
