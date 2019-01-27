import datetime as dt
from io import BytesIO
import json
import multiprocessing as mul
from multiprocessing.dummy import Pool as TPool
import re
import time

import boto3
import jsonschema
import pandas as pd

from scripts.constants import *


def get_feature_names(kwargs={}):
    """get the list of feature names

    Returns:
        [str,]
    """
    client = boto3.client('s3')

    # TODO: Paginate. will fail over 300 features
    response = client.list_objects_v2(
        Bucket=BUCKET,
        Delimiter=',',
        Prefix=configure_prefix(FEATURES_KEY, kwargs),
        MaxKeys=1000,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        features = list({
            re.sub(r'(/|.csv|_test|_train|_validate|' + configure_prefix(FEATURES_KEY, kwargs) + ')+', '', c['Key'])
            for c in contents[1:]
    })
    else:
        features = []

    return features


def validate_name(feature_name, kwargs={}):
    """check whether or not a feature name is valid

    Args:
        feature_name: str - name of feature to verify

    Returns:
        True if unique
    """
    return feature_name not in get_feature_names(kwargs)


def upload_feature(feature_name, paths, overwrite=False, kwargs={}):
    if not validate_name(feature_name, kwargs) and not overwrite:
        raise ValueError(f'Feature name <{feature_name}> already in use')

    if any([not os.path.exists(p) for p in paths]):
        raise FileNotFoundError('Path to feature not found')

    client = boto3.client('s3')
    for path, dataset in zip(paths, ['train', 'test', 'validate']):
        key = f'{configure_prefix(FEATURES_KEY, kwargs)}/{feature_name}_{dataset}.csv'

        with open(path, 'rb') as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=BUCKET,
                Key=key,
                Tagging=TAG_KEY + "=" + TAG_VALUE
            )
            set_acl(client, key)


def set_acl(client, key):
    acl = client.get_bucket_acl(Bucket=BUCKET)
    del acl['ResponseMetadata']

    client.put_object_acl(
        Bucket=BUCKET,
        Key=key,
        AccessControlPolicy=acl
    )


def download_feature(feature_name, kwargs={}):
    if validate_name(feature_name, kwargs):
        raise ValueError(f'Feature <{feature_name}> does not exist')

    client = boto3.client('s3')

    result = {}
    for dataset in ['train', 'test', 'validate']:
        key = f'{configure_prefix(FEATURES_KEY, kwargs)}/{feature_name}_{dataset}.csv'

        obj = client.get_object(
            Bucket=BUCKET,
            Key=key,
        )
        if obj['ContentLength'] > 10:
            bio = BytesIO(obj['Body'].read())

            result[dataset] = pd.read_csv(bio)

            bio.close()
            del bio

    return result


def build_feature_set(feature_names, max_concurrent_conn=-1, kwargs={}):
    max_conn = mul.cpu_count()*5 if max_concurrent_conn == -1 else max_concurrent_conn

    def download_wrapper(args):
        return download_feature(*args)

    pool = TPool(max_conn)
    result = list(pool.map(
        download_wrapper,
        [(feature, kwargs) for feature in feature_names]
    ))

    def recursive_join(frames):
        if len(frames) == 1:
            return frames[0]

        return frames[0].merge(recursive_join(frames[1:]), on='MachineIdentifier', how='outer')

    return dict(
        train=recursive_join([ri['train'] for ri in result]),
        test=recursive_join([ri['test'] for ri in result]),
        validate=recursive_join([ri['validate'] for ri in result])
    )


def download_config(job_name, kwargs={}):
    if job_name not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> has not been prepared')

    client = boto3.client('s3')

    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_config'
    obj = client.get_object(
        Bucket=BUCKET,
        Key=key
    )
    if obj['ContentLength'] > 10:
        bio = BytesIO(obj['Body'].read())
        config = json.load(bio)
        jsonschema.validate(config, SCHEMA)

        bio.close()
        del bio
    else:
        raise Exception(f'Failed to download config for job {job_name}')

    return config


def get_jobs_listing(kwargs={}):
    """get the list of jobs

    Returns:
        [str,]
    """
    client = boto3.client('s3')

    # TODO: Paginate. will fail over 300
    response = client.list_objects_v2(
        Bucket=BUCKET,
        Delimiter=',',
        Prefix=configure_prefix(JOBS_KEY, kwargs),
        MaxKeys=1000,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        jobs_list = list({
            re.sub(r'(/|_config|' + configure_prefix(JOBS_KEY, kwargs) + ')+', '', c['Key'])
            for c in contents[1:]
    })
    else:
        jobs_list = []

    return jobs_list


def prepare_job(config, overwrite=False, kwargs={}):
    if not isinstance(config, dict):
        raise TypeError

    if not overwrite and config['job_name'] in get_jobs_listing(kwargs):
        raise ValueError(f'{config["job_name"]} already in use')

    job_name = config['job_name']
    config = json.dumps(config)
    jsonschema.validate(config, SCHEMA)

    client = boto3.client('s3')
    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_config'

    with BytesIO(bytes(config, encoding='utf-8')) as f:
        response = client.put_object(
            ACL='private',
            Body=f,
            Bucket=BUCKET,
            Key=key,
            Tagging=TAG_KEY + "=" + TAG_VALUE
        )
    set_acl(client, key)

    config = json.loads(config)
    config['submit_time'] = dt.datetime.now().isoformat()
    config['status_code'] = response['ResponseMetadata']['HTTPStatusCode']

    return config


def prepare_init(job_name, kwargs={}):
    if 'test' in kwargs.keys():
        p = os.path.join(os.path.abspath('..'), 'scripts', 'init.sh')
    else:
        p = 'init.sh'

    with open(p, 'r') as f:
        init = f.read()

    init = init.replace(
        "!aws_access_key_id!", SECRETS['AWS_KEY']
    ).replace(
        "!aws_secret_access_key!", SECRETS['AWS_SECRET']
    ).replace(
        "!kaggle_username!", SECRETS['KAGGLE_USERNAME']
    ).replace(
        "!kaggle_key!", SECRETS['KAGGLE_KEY']
    ).replace(
        "!job_name!", job_name
    )

    return init


def run_job(job_name, instance_details, kwargs={}):
    if not isinstance(instance_details, dict):
        raise TypeError
    if any([
        c not in instance_details.keys()
        for c in ['InstanceType']
    ]):
        raise KeyError('instance details must contain InstanceType')

    if job_name not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> not prepared')

    if 'AWS_DEFAULT_REGION' in SECRETS.keys():
        region = SECRETS['AWS_DEFAULT_REGION']
    elif 'region' in instance_details.keys():
        region = instance_details['region']
    else:
        region = AWS_DEFAULT_REGION

    ec2 = boto3.resource('ec2', region_name=region)

    security_group = ec2.create_security_group(

    )

    instance = ec2.create_instances(
        Name=f'DankDefense_{job_name}',
        ImageId=AMI,
        InstanceType=instance_details['InstanceType'],
        UserData=prepare_init(job_name, kwargs=kwargs),
        DisableApiTermination=False,
        EbsOptimized=True,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': TAG_KEY,
                        'Value': TAG_VALUE
                    },
                ]
            },
        ],
        InstanceMarketOptions={
            'MarketType': 'spot',
            'SpotOptions': {
                'SpotInstanceType': 'one-time',
                'InstanceInterruptionBehavior': 'terminate'
            }
        },
        MaxCount=1,
        MinCount=1,
        Monitoring=dict(Enabled=False)
    )

    while get_results(job_name, kwargs=kwargs) is None:
        time.sleep(10)
    print()


def get_results(job_name, kwargs={}):
    pass


def configure_prefix(key, kwargs):
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + key
    else:
        prefix = key

    return prefix


if __name__ == '__main__':
    pass