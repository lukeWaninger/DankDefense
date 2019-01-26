from io import BytesIO
import multiprocessing as mul
from multiprocessing.dummy import Pool as TPool
import os
import re

import boto3
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
        Prefix=features_prefix(kwargs),
        MaxKeys=1000,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        features = list({
            re.sub(r'(/|.csv|_test|_train|_validate|'+features_prefix(kwargs)+')+', '', c['Key'])
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
        key = f'{features_prefix(kwargs)}/{feature_name}_{dataset}.csv'

        with open(path, 'rb') as f:
            response = client.put_object(
                ACL='private',
                Body=f,
                Bucket=BUCKET,
                Key=key,
                Tagging="Project=DankDefense"
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
        key = f'{features_prefix(kwargs)}/{feature_name}_{dataset}.csv'

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


def features_prefix(kwargs):
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + FEATURES_KEY
    else:
        prefix = FEATURES_KEY

    return prefix


if __name__ == '__main__':
    pass