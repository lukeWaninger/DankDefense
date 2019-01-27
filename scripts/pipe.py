from contextlib import contextmanager
import datetime as dt
from io import BytesIO
import json
import multiprocessing as mul
from multiprocessing.dummy import Pool as TPool
import re
import requests
import time

import boto3
from botocore.exceptions import ClientError
import jsonschema
import pandas as pd
import paramiko

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
                Tagging=TAG_KEY + "=" + PROJECT_NAME
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
            Tagging=TAG_KEY + "=" + PROJECT_NAME
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


def run_job(
        job_name,
        aws_region=None,
        instance_type=None,
        is_spot=True,
        ssh_key=None,
        kwargs={}):
    if job_name not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> not prepared')

    # get the aws_region
    if aws_region is None and 'AWS_DEFAULT_REGION' in SECRETS.keys():
        region = SECRETS['AWS_DEFAULT_REGION']
    else:
        region = AWS_DEFAULT_REGION

    # get the instance type
    if instance_type is None and 'AWS_DEFAULT_INSTANCE_TYPE' in SECRETS.keys():
        instance_type = SECRETS['AWS_DEFAULT_INSTANCE_TYPE']
    else:
        raise ValueError('InstanceType must be defined in SECRETS or instance_details')

    ec2 = boto3.client('ec2', region_name=region)

    try:
        security_group = ec2.create_security_group(
            Description=f'{PROJECT_NAME} inbound ssh',
            GroupName=PROJECT_NAME
        )

        this_ip = requests.get('http://ip.42.pl/raw').text + '/32'
        ec2.authorize_security_group_ingress(
            GroupId=security_group['GroupId'],
            IpPermissions=[
                dict(
                    IpProtocol='tcp',
                    FromPort=22,
                    ToPort=22,
                    IpRanges=[dict(CidrIp=this_ip)]
                )
            ]
        )

        security_group = security_group['GroupId']
    except ClientError as e:
        if 'InvalidGroup.Duplicate' in e.response['Error']['Code']:
            security_group = ec2.describe_security_groups(
                GroupNames=[PROJECT_NAME]
            )
            if len(security_group['SecurityGroups']) > 0:
                security_group = security_group['SecurityGroups'][0]['GroupId']
            else:
                raise e
        else:
            raise e

    iargs = dict(
        ImageId=AMI,
        InstanceType=instance_type,
        SecurityGroupIds=[security_group],
        UserData=prepare_init(job_name, kwargs=kwargs),
        DisableApiTermination=False,
        EbsOptimized=True,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': TAG_KEY,
                        'Value': PROJECT_NAME
                    },
                ]
            },
        ],
        MaxCount=1,
        MinCount=1,
        Monitoring=dict(Enabled=False)
    )

    # verify spot request
    if is_spot:
        iargs['InstanceMarketOptions'] = {
            'MarketType': 'spot',
            'SpotOptions': {
                'SpotInstanceType': 'one-time',
                'InstanceInterruptionBehavior': 'terminate'
            }
        }

    # add the ssh key if it exists
    if not 'AWS_KEY_PAIR_NAME' in SECRETS.keys():
        print('WARNING: no SSH Key found for remote monitoring')
    else:
        ssh_key = SECRETS['AWS_KEY_PAIR_NAME']

    if ssh_key is not None:
        iargs['KeyName'] = ssh_key

    ec2 = boto3.resource('ec2', region_name=region)
    instance = ec2.create_instances(**iargs)[0]

    instance.wait_until_running()
    instance.load()

    try:
        with ec2sftp(instance.public_dns_name) as svr:
            finished = False
            log_file, log = f'{job_name}_log.txt', []

            while not finished:
                svr.get(log_file, log_file)

                with open(log_file, 'r') as f:
                    log_ = f.readlines()

                for line in log_:
                    if line not in log:
                        print(line)
                        log.append(line)
                    else:
                        pass

                if 'job complete' in log[-1]:
                    finished = True
                else:
                    time.sleep(10)

        ec2.terminate_instances(InstanceIds=[instance.instance_id])
    except:
        print('WARNING: you instance is still running')
    print()


@contextmanager
def ec2sftp(public_dns):
    try:
        server = paramiko.SSHClient()
        server.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        server.connect(
            hostname=public_dns,
            username='ubuntu',
            key_filename=SECRETS['AWS_KEY_PAIR_PATH'],
            look_for_keys=False,
        )

        server = server.open_sftp()
        yield server
    except Exception as e:
        pass
    finally:
        server.close()


def upload_results(job_name, results, kwargs={}):
    if job_name not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> has not been prepared')

    client = boto3.client('s3')
    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_results.txt'

    with BytesIO(bytes(results, encoding='utf-8')) as f:
        response = client.put_object(
            ACL='private',
            Body=f,
            Bucket=BUCKET,
            Key=key,
            Tagging=TAG_KEY + "=" + PROJECT_NAME
        )
    set_acl(client, key)


def get_results(job_name, kwargs={}):
    if job_name + '_results.txt' not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> has not reported results')

    client = boto3.client('s3')

    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_results.txt'
    obj = client.get_object(
        Bucket=BUCKET,
        Key=key
    )
    if obj['ContentLength'] > 10:
        with BytesIO(obj['Body'].read()) as f:
            results = f.read()

        return results
    else:
        raise Exception(f'Failed to download results for job {job_name}')


def configure_prefix(key, kwargs):
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + key
    else:
        prefix = key

    return prefix


if __name__ == '__main__':
    pass
