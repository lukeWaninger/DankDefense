#!/bin/bash

export AWS_ACCESS_KEY_ID="!aws_access_key_id!"
export AWS_SECRET_ACCESS_KEY="!aws_secret_access_key!"
export KAGGLE_USERNAME="!kaggle_username!"
export KAGGLE_KEY="!kaggle_key!"
export JOB="!job_name!"

export LOGFILE=$JOB\_log.txt

apt-get -y install python3-pip
pip3 install awscli

log_message()
{
    echo "`date`: " $1 &>> $LOGFILE
}

cd /home/ubuntu

log_message $"starting job"
log_message $"downloading base data"
# mkdir data
# aws s3 cp s3://dank-defense/data data --recursive

log_message $"installing the dank pipe"
pip3 install git+https://github.com/lukeWaninger/DankDefense

log_message $"executing runner"
python3 runner.py $JOB &>> $LOGFILE

log_message $"uploading logs"
aws s3 cp $LOGFILE s3://dank-defense/jobs/$LOGFILE

log_message $"job complete"