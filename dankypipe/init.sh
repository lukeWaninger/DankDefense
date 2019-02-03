#!/bin/bash

export AWS_ACCESS_KEY_ID="!aws_access_key_id!"
export AWS_SECRET_ACCESS_KEY="!aws_secret_access_key!"
export KAGGLE_USERNAME="!kaggle_username!"
export KAGGLE_KEY="!kaggle_key!"
export JOB="!job_name!"

export LOGFILE=$JOB\_log.txt

log_message()
{
    echo "`date`: " $1 &>> $LOGFILE
}
log_message $"starting job"

cd /home/ubuntu

log_message $"installing python3-pip"
apt-get update
apt-get -y install python3-pip

log_message $"installing awcli"
pip3 install awscli

log_message $"installing the dank pipe"
pip3 install git+https://github.com/lukeWaninger/DankDefense &>> $LOGFILE

log message $"installing requirements"
wget https://raw.githubusercontent.com/lukeWaninger/DankDefense/master/dankypipe/requirements.txt
pip3 install -r requirements.txt

log_message $"executing runner"
wget https://raw.githubusercontent.com/lukeWaninger/DankDefense/master/dankypipe/runner.py
python3 runner.py $JOB

log_message $"uploading logs"
aws s3 cp $LOGFILE s3://dank-defense/jobs/$LOGFILE

log_message $"job complete"