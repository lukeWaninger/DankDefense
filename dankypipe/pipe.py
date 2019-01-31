from contextlib import contextmanager
import datetime as dt
from io import BytesIO, StringIO
import json
import multiprocessing as mul
from multiprocessing.dummy import Pool as TPool
import os
import re
import requests
import time

import boto3
from botocore.exceptions import ClientError
import jsonschema
import pandas as pd
import paramiko

import dankypipe.constants as const


def get_feature_names(**kwargs):
    """get the list of feature names

    Returns:
        [str,]
    """
    client = boto3.client('s3')

    # TODO: Paginate. will fail over 300 features
    response = client.list_objects_v2(
        Bucket=const.BUCKET,
        Delimiter=',',
        Prefix=configure_prefix(const.FEATURES_KEY, kwargs),
        MaxKeys=1000,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        prefix = configure_prefix(const.FEATURES_KEY, kwargs)

        features = list({
            re.sub(r'(/|.csv|_' + '|_'.join(const.DATASET_KEYS) + '|' + prefix + ')+', '', c['Key'])
            for c in contents[1:]
    })
    else:
        features = []

    return features


def validate_name(feature_name, **kwargs):
    """check whether or not a feature name is valid

    Args:
        feature_name: str - name of feature to verify

    Returns:
        True if unique
    """
    return feature_name not in get_feature_names(**kwargs)


def upload_feature(feature_name, datasets, overwrite=False, **kwargs):
    """upload a feature to S3

    Args:
        feature_name: (str) name of feature
        datasets: ((str, str, str) | (pd.DataFrame, pd.DataFrame, pd.DataFrame)) file paths or DataFrames to training, test, and validation csv files
        overwrite: (bool) set to True to overwrite an existing feature
        kwargs:

    Returns:
        dict containing ETags of uploaded items by dataset
    """
    if not validate_name(feature_name, **kwargs) and not overwrite:
        raise ValueError(f'Feature name <{feature_name}> already in use')

    if any([not os.path.exists(p) for p in datasets if isinstance(p, str)]):
        raise FileNotFoundError('Path to feature not found')

    client = boto3.client('s3')
    etags = {}

    for feat, dataset in zip(datasets, const.DATASET_KEYS):
        path = f'{feature_name}_{dataset}.pickle'
        key = f'{configure_prefix(const.FEATURES_KEY, kwargs)}/{path}'

        if isinstance(feat, pd.DataFrame):
            if not os.path.exists('tmp'):
                os.mkdir('tmp')

            path = os.path.join('tmp', path)
            feat.to_pickle(path, compression='gzip')

        with open(path, 'rb') as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=const.BUCKET,
                Key=key,
                Tagging=const.TAG_KEY + "=" + const.PROJECT_NAME
            )
        set_acl(client, key)

        etags[dataset] = response['ETag'].replace('"', '')

    return etags


def set_acl(client, key):
    """set access control policy on S3 object

    Args:
        client: (boto3.Client) the active S3 client
        key: (str)

    Returns:
        None
    """
    acl = client.get_bucket_acl(Bucket=const.BUCKET)
    del acl['ResponseMetadata']

    client.put_object_acl(
        Bucket=const.BUCKET,
        Key=key,
        AccessControlPolicy=acl
    )


def download_feature(feature_name, cache=False, **kwargs):
    """download a single feature

    Args:
        feature_name: (str) name of feature
        cache: (bool) set to True if you plan on using the instance for more than one job
        kwargs:

    Returns:
        pd.DataFrame
    """
    if validate_name(feature_name, **kwargs):
        raise ValueError(f'Feature <{feature_name}> does not exist')

    client = boto3.client('s3')

    result = {}
    for dataset in const.DATASET_KEYS:
        prefix = configure_prefix(const.FEATURES_KEY, kwargs)

        if cache and not os.path.exists(prefix):
            os.mkdir(prefix)

        key = f'{prefix}/{feature_name}_{dataset}.pickle'
        if os.path.exists(key):
            result[dataset] = pd.read_pickle(key)

        else:
            obj = client.get_object(
                Bucket=const.BUCKET,
                Key=key,
            )
            if obj['ContentLength'] > 10:
                bio = StringIO(obj['Body'].read())

                result[dataset] = pd.read_pickle(bio.read())

                if cache:
                    result[dataset].to_csv(key, index=None)
                else:
                    pass

                bio.close()
                del bio

    return result


def build_feature_set(feature_names, max_concurrent_conn=-1, **kwargs):
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
        if len(args) > 1:
            return download_feature(args[0], **args[1])
        else:
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

    target = download_feature('Target')

    return {
        key: {
            'x': recursive_join([ri[key] for ri in result]),
            'y': target[key] if key != 'test' else None
        }
        for key in const.DATASET_KEYS
    }


def download_config(job_name, **kwargs):
    """download job config from S3

    Args:
        job_name: (str) name of job

    Returns:
        dict
    """
    if job_name not in get_jobs_listing(**kwargs):
        raise ValueError(f'Job <{job_name}> has not been prepared')

    client = boto3.client('s3')

    key = f'{configure_prefix(const.JOBS_KEY, kwargs)}/{job_name}_config'
    obj = client.get_object(
        Bucket=const.BUCKET,
        Key=key
    )
    if obj['ContentLength'] > 10:
        bio = BytesIO(obj['Body'].read())
        config = json.load(bio)
        jsonschema.validate(config, const.SCHEMA)

        bio.close()
        del bio
    else:
        raise Exception(f'Failed to download config for job {job_name}')

    return config


def get_jobs_listing(**kwargs):
    """get the list of jobs

    Returns:
        [str,]
    """
    client = boto3.client('s3')

    # TODO: Paginate. will fail over 300
    response = client.list_objects_v2(
        Bucket=const.BUCKET,
        Delimiter=',',
        Prefix=configure_prefix(const.JOBS_KEY, kwargs),
        MaxKeys=1000,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        jobs_list = list({
            re.sub(r'(/|_config|' + configure_prefix(const.JOBS_KEY, kwargs) + ')+', '', c['Key'])
            for c in contents[1:]
    })
    else:
        jobs_list = []

    return jobs_list


def validate_job_name(job_name, **kwargs):
    """

    Args:
        job_name: (str) name of job
        kwargs:

    Returns:
        True if job_name has not been used
    """
    return job_name not in get_jobs_listing(**kwargs)


class Ec2Job(object):
    def __init__(self, config,
                 overwrite=False,
                 aws_region=None,
                 instance_type=None,
                 ssh_key_name=None,
                 ssh_key_path=None,
                 **kwargs):
        self.config = config
        self.job_name = config['job_name']
        self.instance = None
        self.iid = None

        # get the aws_region
        if aws_region is None and 'AWS_DEFAULT_REGION' in const.SECRETS.keys():
            self.region = const.SECRETS['AWS_DEFAULT_REGION']
        else:
            self.region = const.AWS_DEFAULT_REGION

        # get the instance type
        if instance_type is not None:
            self.instance_type = instance_type
        elif 'AWS_DEFAULT_INSTANCE_TYPE' in const.SECRETS.keys():
            self.instance_type = const.SECRETS['AWS_DEFAULT_INSTANCE_TYPE']
        else:
            raise ValueError('InstanceType must be defined in SECRETS or instance_details')

        self._client = boto3.client('ec2', region_name=self.region)
        self._resource = boto3.resource('ec2', region_name=self.region)

        # add the ssh key if it exists
        if ssh_key_name is None and 'AWS_KEY_PAIR_NAME' not in const.SECRETS.keys():
            print('WARNING: no SSH Key found for remote monitoring')
        elif ssh_key_name is None:
            self.ssh_key_name = const.SECRETS['AWS_KEY_PAIR_NAME']

        if ssh_key_path is not None:
            self.ssh_key_path = ssh_key_path
        else:
            self.ssh_key_path = const.SECRETS['AWS_KEY_PAIR_PATH']

        self.security_group = set_security_groups(self._client)

        try:
            self.prepare_job(overwrite, **kwargs)
        except ValueError as e:
            if 'already in use' in e.args:
                print(e.args)
                print('Run prepare_jobs again with overwrite to overwrite the existing job')
            else:
                raise e

    def __repr__(self):
        ins = f'{self.instance.instance_id} is {self.instance.state["Name"]}' \
            if self.instance is not None else "no instance attached"
        return f'Ec2Job <({self.config["job_name"]}, {ins})>'

    def __del__(self):
        self.terminate_instance()

    def __prepare_init(self):
        """builds an initialization shell script for the EC2 instance

        Args:
            job_name: (str) name of job

        Returns:
            str
        """
        p = 'init.sh'
        with open(p, 'r') as f:
            init = f.read()

        init = init.replace(
            "!aws_access_key_id!", const.SECRETS['AWS_ACCESS_KEY_ID']
        ).replace(
            "!aws_secret_access_key!", const.SECRETS['AWS_SECRET_ACCESS_KEY']
        ).replace(
            "!kaggle_username!", const.SECRETS['KAGGLE_USERNAME']
        ).replace(
            "!kaggle_key!", const.SECRETS['KAGGLE_KEY']
        ).replace(
            "!job_name!", self.job_name
        )

        return init

    def prepare_job(self, overwrite=False, **kwargs):
        """prepare and load job to S3

        Args:
            overwrite: (bool) set to True to overwrite a job. Logs, results, and predictions will be removed from S3
            kwargs:

        Returns:
            dict
        """
        if not isinstance(self.config, dict):
            raise TypeError

        if not overwrite and self.config['job_name'] in get_jobs_listing(**kwargs):
            raise ValueError(f'{self.config["job_name"]} already in use')

        job_name = self.config['job_name']
        config = json.dumps(self.config)
        jsonschema.validate(config, const.SCHEMA)
        prefix = configure_prefix(const.JOBS_KEY, kwargs)

        client = boto3.client('s3')
        key = f'{prefix}/{job_name}_config'

        with BytesIO(bytes(config, encoding='utf-8')) as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=const.BUCKET,
                Key=key,
                Tagging=const.TAG_KEY + "=" + const.PROJECT_NAME
            )
        set_acl(client, key)

        config = json.loads(config)
        config['submit_time'] = dt.datetime.now().isoformat()
        config['status_code'] = response['ResponseMetadata']['HTTPStatusCode']

        if overwrite:
            response = client.delete_objects(
                Bucket=const.BUCKET,
                Delete=dict(
                    Objects=[
                        dict(Key=f'{prefix}/{job_name}_predictions.csv'),
                        dict(Key=f'{prefix}/{job_name}_results.txt'),
                        dict(Key=f'{prefix}/{job_name}_log.txt')
                    ],
                    Quiet=True
                )
            )

        return config

    def run_job(self, is_spot=True, **kwargs):
        """

        Args:
            is_spot: (bool) - set to True to initialize as a spot request
            kwargs:

        Returns:

        """
        if self.instance is not None:
            self.instance.load()

            with ec2ssh(self.instance.public_dns_name, self.ssh_key_path, sftp=False) as svr:
                cmd = f'python3 runner.py {self.job_name}'
                svr.exec_command(cmd)
        else:
            if self.job_name not in get_jobs_listing(**kwargs):
                raise ValueError(f'Job <{self.job_name}> not prepared')

            iargs = dict(
                ImageId=const.AMI,
                InstanceType=self.instance_type,
                SecurityGroupIds=[self.security_group],
                UserData=self.__prepare_init(),
                DisableApiTermination=False,
                EbsOptimized=True,
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {
                                'Key': const.TAG_KEY,
                                'Value': const.PROJECT_NAME
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

            if self.ssh_key_name is not None:
                iargs['KeyName'] = self.ssh_key_name

            print(f'initializing EC2 instance')
            start = time.time()

            self.instance = self._resource.create_instances(**iargs)[0]
            self.iid = self.instance.instance_id

            self.instance.wait_until_running()

            # wait for instance to run init.sh
            time.sleep(45)
            self.instance.load()

        try:
            self.monitor()

            results = get_results(self.job_name, include_predictions=True, kwargs=kwargs)
            results['instance_id'] = self.iid
            results['run_time'] = pd.Timedelta(seconds=(time.time()-start))

            return results
        except Exception as e:
            results = dict(
                instance_id=self.iid,
                run_time='WARNING: your instance is still running',
                exception=e
            )
            return results

    def monitor(self):
        print(f'establishing connection with {self.instance.public_dns_name}')
        with ec2ssh(self.instance.public_dns_name, self.ssh_key_path) as svr:
            finished, i = False, 0
            log_file, log = f'{self.job_name}_log.txt', []

            while not finished:
                try:
                    svr.get(log_file, log_file)

                    with open(log_file, 'r') as f:
                        log_ = f.readlines()

                    for line in log_:
                        line = line.rstrip()

                        if line not in log:
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

    def terminate_instance(self):
        try:
            response = self._client.terminate_instances(InstanceIds=[self.iid])
            return response
        except Exception as e:
            print()
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('!! WARNING: your instance is still running !!')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(e.args)
            print()

    def reset_config(self, config):
        self.config = config
        self.job_name = config['job_name']
        return self


def get_results(job_name, include_predictions=False, **kwargs):
    """retrieve validation results from S3

    Args:
        job_name: (str) name of job
        include_predictions: (bool) set to True to include predicted values
        kwargs:

    Returns:
        dict
    """
    if job_name + '_results.txt' not in get_jobs_listing(**kwargs):
        raise ValueError(f'Job <{job_name}> has not reported results')

    results = {}
    client = boto3.client('s3')

    # get the config
    results['config'] = download_config(job_name, **kwargs)

    # get the results summary
    key = f'{configure_prefix(const.JOBS_KEY, **kwargs)}/{job_name}_results.txt'
    obj = client.get_object(
        Bucket=const.BUCKET,
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
        key = f'{configure_prefix(const.JOBS_KEY, kwargs)}/{job_name}_predictions.csv'
        obj = client.get_object(
            Bucket=const.BUCKET,
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


def set_ingress(security_group, client):
    try:
        this_ip = requests.get('http://ip.42.pl/raw').text + '/32'
        client.authorize_security_group_ingress(
            GroupId=security_group,
            IpPermissions=[
                dict(
                    IpProtocol='tcp',
                    FromPort=22,
                    ToPort=22,
                    IpRanges=[dict(CidrIp=this_ip)]
                )
            ]
        )
    except ClientError as e:
        if 'InvalidGroup.Duplicate' in e.response['Error']['Code']:
            pass


def set_security_groups(client):
    try:
        security_group = client.create_security_group(
            Description=f'{const.PROJECT_NAME} inbound ssh',
            GroupName=const.PROJECT_NAME
        )

        security_group = security_group['GroupId']
        set_ingress(security_group, client)

        print(f'Created security group with id: {security_group}')
        return security_group
    except ClientError as e:
        if 'InvalidGroup.Duplicate' in e.response['Error']['Code']:
            security_group = client.describe_security_groups(
                GroupNames=[const.PROJECT_NAME]
            )
            if len(security_group['SecurityGroups']) > 0:
                security_group = security_group['SecurityGroups'][0]['GroupId']
                set_ingress(security_group, client)

                return security_group
            else:
                raise e
        else:
            raise e


@contextmanager
def ec2ssh(public_dns, ssh_key_path, sftp=True):
    server = paramiko.SSHClient()
    server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connected, attempts = False, 0

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

            if sftp:
                server = server.open_sftp()
            else:
                pass

            yield server
            server.close()
        except TimeoutError as e:
            attempts += 1
            if attempts == const.MAX_RETRIES:
                raise e
        except Exception as e:
            raise e


def upload_results(job_name, result_summary, predictions, **kwargs):
    """upload result summary and predicted values to S3

    Args:
        job_name: (str) name of job
        result_summary: (str)
        predictions: (pd.DataFrame)
        kwargs:

    Returns:
        None
    """
    if job_name not in get_jobs_listing(**kwargs):
        raise ValueError(f'Job <{job_name}> has not been prepared')

    client = boto3.client('s3')

    # upload the result summary
    if result_summary is not None:
        key = f'{configure_prefix(const.JOBS_KEY, kwargs)}/{job_name}_results.txt'
        with BytesIO(bytes(result_summary, encoding='utf-8')) as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=const.BUCKET,
                Key=key,
                Tagging=const.TAG_KEY + "=" + const.PROJECT_NAME
            )
        set_acl(client, key)

    # upload the predicted values
    if predictions is not None:
        filename = f'{job_name}_predictions.csv'
        key = f'{configure_prefix(const.JOBS_KEY, kwargs)}/{filename}'

        predictions.to_csv(filename, index=None)
        with open(filename, 'rb') as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=const.BUCKET,
                Key=key,
                Tagging=const.TAG_KEY + "=" + const.PROJECT_NAME
            )
        set_acl(client, key)


def configure_prefix(key, kwargs):
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + key
    else:
        prefix = key

    return prefix


if __name__ == '__main__':
    pass
