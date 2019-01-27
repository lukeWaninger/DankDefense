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
    """upload a feature to S3

    Args:
        feature_name: (str) name of feature
        paths: ((str, str, str)) file paths to training, test, and validation csv files
        overwrite: (bool) set to True to overwrite an existing feature
        kwargs:

    Returns:
        None
    """
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
    """set access control policy on S3 object

    Args:
        client: (boto3.Client) the active S3 client
        key: (str)

    Returns:
        None
    """
    acl = client.get_bucket_acl(Bucket=BUCKET)
    del acl['ResponseMetadata']

    client.put_object_acl(
        Bucket=BUCKET,
        Key=key,
        AccessControlPolicy=acl
    )


def download_feature(feature_name, kwargs={}):
    """download a single feature

    Args:
        feature_name: (str) name of feature
        kwargs:

    Returns:
        pd.DataFrame
    """
    if validate_name(feature_name, kwargs):
        raise ValueError(f'Feature <{feature_name}> does not exist')

    client = boto3.client('s3')

    result = {}
    for dataset in constants.DATASET_KEYS:
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
    """builds a Pandas DataFrame containing the requested features

    Args:
        feature_names: ([str,]) list of feature names
        max_concurrent_conn: (int) Default-5xCoreCount. Sets the maximum number of features to download simultaneously
        kwargs:

    Returns:
        pd.DataFrame
    """
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

    return dict{
        key: recursive_join([ri[key] for ri in result]) for key in constants.DATASET_KEYS
    }


def download_config(job_name, kwargs={}):
    """download job config from S3

    Args:
        job_name: (str) name of job

    Returns:
        dict
    """
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


def validate_job_name(job_name, kwargs={}):
    """

    Args:
        job_name: (str) name of job
        kwargs:

    Returns:
        True if job_name has not been used
    """
    return job_name not in get_jobs_listing(kwargs)


def prepare_job(config, overwrite=False, kwargs={}):
    """prepare and load job to S3

    Args:
        config: (dict) matching schema defined in constants.py
        overwrite: (bool) set to True to overwrite a job. Logs, results, and predictions will be removed from S3
        kwargs:

    Returns:
        dict
    """
    if not isinstance(config, dict):
        raise TypeError

    if not overwrite and config['job_name'] in get_jobs_listing(kwargs):
        raise ValueError(f'{config["job_name"]} already in use')

    job_name = config['job_name']
    config = json.dumps(config)
    jsonschema.validate(config, SCHEMA)
    prefix = configure_prefix(JOBS_KEY, kwargs)

    client = boto3.client('s3')
    key = f'{prefix}/{job_name}_config'

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

    if overwrite:
        response = client.delete_objects(
            Bucket=BUCKET,
            Delete=dict(
                Objects=[
                    dict(key=f'{prefix}/{job_name}_predictions.csv'),
                    dict(key=f'{prefix}/{job_name}_results.txt'),
                    dict(key=f'{prefix}/{job_name}_log.txt')
                ],
                Quiet=True
            )
        )

    return config


def prepare_init(job_name, kwargs={}):
    """builds an initialization shell script for the EC2 instance

    Args:
        job_name: (str) name of job

    Returns:
        str
    """
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
        ssh_key_name=None,
        ssh_key_path=None,
        kwargs={}):
    """

    Args:
        job_name: (str) - name of job to reference in S3
        aws_region: (str) - AWS region to build instance and security groups
        instance_type: (str) - ie. t3.nane | c4.xlarge | ...
        is_spot: (bool) - set to True to initialize as a spot request
        ssh_key_name: (str) - name of key pair in AWS
        ssh_key_path: (str) - path to private key on local machine
        kwargs:

    Returns:

    """
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

    ec2_client = boto3.client('ec2', region_name=region)

    try:
        security_group = ec2_client.create_security_group(
            Description=f'{PROJECT_NAME} inbound ssh',
            GroupName=PROJECT_NAME
        )

        this_ip = requests.get('http://ip.42.pl/raw').text + '/32'
        ec2_client.authorize_security_group_ingress(
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
            security_group = ec2_client.describe_security_groups(
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
    if ssh_key_name is None and 'AWS_KEY_PAIR_NAME' not in SECRETS.keys():
        print('WARNING: no SSH Key found for remote monitoring')
    elif ssh_key_name is None:
        ssh_key_name = SECRETS['AWS_KEY_PAIR_NAME']

    if ssh_key_name is not None:
        iargs['KeyName'] = ssh_key_name

    start = time.time()
    ec2_resource = boto3.resource('ec2', region_name=region)
    instance = ec2_resource.create_instances(**iargs)[0]
    iid = instance.instance_id

    instance.wait_until_running()
    instance.load()

    try:
        time.sleep(30)
        with ec2sftp(instance.public_dns_name, ssh_key_path) as svr:
            finished, i = False, 0
            log_file, log = f'{job_name}_log.txt', []

            while not finished:
                try:
                    svr.get(log_file, log_file)

                    with open(log_file, 'r') as f:
                        log_ = f.readlines()

                    for line in log_:
                        if line not in log:
                            line = line.rstrip()

                            print(line)
                            log.append(line)
                        else:
                            pass

                    if 'job complete' in log[-1]:
                        finished = True
                    else:
                        time.sleep(10)

                except FileNotFoundError as e:
                    i += 1
                    if i == 10:
                        raise e
                    else:
                        time.sleep(10)

        ec2_client.terminate_instances(InstanceIds=[iid])

        results = get_results(job_name, include_predictions=True, kwargs=kwargs)
        results['instance_id'] = iid
        results['run_time'] = pd.Timedelta(seconds=(time.time()-start))
        return results
    except Exception as e:
        print()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('WARNING: your instance is still running')
        print('---------------------------------------')
        results = dict(
            instance_id=iid,
            run_time='WARNING: your instance is still running',
            exception=e
        )
        return results


@contextmanager
def ec2sftp(public_dns, ssh_key_path=None):
    server = paramiko.SSHClient()
    server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connected, attempts = False, 0

    if ssh_key_path is None and 'AWS_KEY_PAIR_PATH' not in SECRETS.keys():
        raise KeyError('You must provide and AWS key pair name and path to connect to the instance')
    else:
        ssh_key_path = SECRETS['AWS_KEY_PAIR_PATH']

    while not connected:
        try:
            server.connect(
                hostname=public_dns,
                username='ubuntu',
                key_filename=ssh_key_path,
                look_for_keys=False,
                timeout=60
            )
            connected = True

            server = server.open_sftp()

            yield server
            server.close()
        except TimeoutError as e:
            attempts += 1
            if attempts == MAX_RETRIES:
                raise e
        except Exception as e:
            raise e


def upload_results(job_name, result_summary, predictions, kwargs={}):
    """upload result summary and predicted values to S3

    Args:
        job_name: (str) name of job
        result_summary: (str)
        predictions: (pd.DataFrame)
        kwargs:

    Returns:
        None
    """
    if job_name not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> has not been prepared')

    client = boto3.client('s3')

    # upload the result summary
    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_results.txt'
    with BytesIO(bytes(result_summary, encoding='utf-8')) as f:
        response = client.put_object(
            ACL='private',
            Body=f,
            Bucket=BUCKET,
            Key=key,
            Tagging=TAG_KEY + "=" + PROJECT_NAME
        )
    set_acl(client, key)

    # upload the predicted values
    filename = f'{job_name}_predictions.csv'
    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{filename}'

    predictions.to_csv(filename, index=None)
    with open(filename, 'rb') as f:
        response = client.put_object(
            ACL='private',
            Body=f,
            Bucket=BUCKET,
            Key=key,
            Tagging=TAG_KEY + "=" + PROJECT_NAME
        )
    set_acl(client, key)


def get_results(job_name, include_predictions=False, kwargs={}):
    """retrieve validation results from S3

    Args:
        job_name: (str) name of job
        include_predictions: (bool) set to True to include predicted values
        kwargs:

    Returns:
        dict
    """
    if job_name + '_results.txt' not in get_jobs_listing(kwargs):
        raise ValueError(f'Job <{job_name}> has not reported results')

    results = {}
    client = boto3.client('s3')

    # get the config
    results['config'] = download_config(job_name, kwargs)

    # get the results summary
    key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_results.txt'
    obj = client.get_object(
        Bucket=BUCKET,
        Key=key
    )
    if obj['ContentLength'] > 10:
        with BytesIO(obj['Body'].read()) as f:
            summary = f.read()

        results['summary'] = summary
    else:
        results['summary'] = 'Job not ran'

    # get the predictions
    if include_predictions:
        key = f'{configure_prefix(JOBS_KEY, kwargs)}/{job_name}_predictions.csv'
        obj = client.get_object(
            Bucket=BUCKET,
            Key=key
        )
        if obj['ContentLength'] > 10:
            bio = BytesIO(obj['Body'].read())

            results['predictions'] = pd.read_csv(bio)

            bio.close()
            del bio
        else:
            results['predictions'] = None

    return results


def configure_prefix(key, kwargs):
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + key
    else:
        prefix = key

    return prefix


if __name__ == '__main__':
    pass
