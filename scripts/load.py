import re

import boto3

from scripts.constants import *


def get_feature_names(kwargs={}):
    """get the list of feature names

    Returns:
        [str,]
    """
    prefix = 'features'
    if 'test' in kwargs.keys():
        prefix = 'unit-testing/' + prefix
    client = boto3.client('s3')

    response = client.list_objects(
        Bucket='dank-defense',
        Delimiter=',',
        Prefix=prefix,
        MaxKeys=500,
    )

    contents = response.get('Contents')
    if contents is not None and len(contents) > 1:
        features = [
            re.sub(r'(/|.csv|'+prefix+')+', '', c['Key'])
            for c in contents[1:]
        ]
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


def upload_feature(feature_name, path):


if __name__ == '__main__':
    pass