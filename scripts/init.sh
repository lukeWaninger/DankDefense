#!/usr/bin/env bash

export AWS_ACCESS_KEY_ID="!aws_access_key_id!"
export AWS_SECRET_ACCESS_KEY="!aws_secret_access_key!"
export KAGGLE_USERNAME="!kaggle_username!"
export KAGGLE_KEY="!kaggle_key!"
export JOB="!job_name!"

export LOGFILE=$JOB\_log.txt

pip install awscli

log_message()
{
    echo "`date`: " $1 &>> $LOGFILE
}

cd /home/ubuntu

log_message $"starting job"
log_message $"downloading base data"
mkdir data
aws s3 cp s3://dank-defense/data data --recursive

log_message $"downloading pipeline scripts"
mkdir pipe
aws s3 cp s3://dank-defense/scripts scripts --recursive

log_message $"installing Python requirements"
pip install --user -r scripts/requirements.txt &>> $LOGFILE

log_message $"running validation script"
python scripts/validate.py --job=$JOB &>>$LOGFILE

log_message $"uploading logs"
aws s3 cp $LOGFILE s3://dank-defense/jobs/$LOGFILE

log_message $"job complete"
