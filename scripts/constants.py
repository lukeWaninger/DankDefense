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
DATASET_KEYS = ['train', 'test', 'validate']

config_schema = """
config:
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
                - predict
                - validate_predict
                - tune_predict
        tuning:
            type: object
            required:
                - search_type
                - parameters
                - metric
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
                metric:
                    type: string
                    default: auc
                    enum:
                        - auc
"""
SCHEMA = yaml.load(config_schema)
"""https://lightgbm.readthedocs.io/en/latest/Parameters.html"""

with open(os.path.join(Path.home(), 'DD_SECRETS'), 'r') as f:
    cf = f.readlines()

    SECRETS = {
        k: v for k, v in [
            list(map(lambda x: x.strip(), l.split('=')))
            for l in cf if not l.startswith('#')
    ]}

    for secret in SECRETS:
        os.environ[secret] = SECRETS[secret]
