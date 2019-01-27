#!/usr/bin/env bash

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
log_message $"downloading base data"
mkdir data
aws cp s3://dank-defense/data data --recursive &>> $LOGFILE

log_message $"downloading pipeline scripts"
mkdir pipe
aws cp s3://dank-defense/pipe pipe --recursive &>>$LOGFILE

log_message $"installing Python requirements"
pip install --user -r pipe/requirements.txt &>> $LOGFILE

log_message $"running validation script"
python pipe/validate.py --job=$JOB &>>$LOGFILE

log_message $"uploading results"
python pipe/pipe.py --upload-results

sudo