from __future__ import print_function

import json
import urllib
import boto3

print('Loading message function...')
topc_arn = 'arn:aws:sns:eu-central-1:348499957229:parquet-file'
sns = boto3.client('sns')

base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/'
years = range(2009, 2023)
months = range(1,13)
is_dry_run = True # TODO: set to false


def publish(subject, msg):
    sns.publish(
        TopicArn=topc_arn,
        Subject=subject,
        Message=msg
    )

def iterate_files():
    for year in years:
        for month in months:
            if year == 2023 and month > 2: # 2023-02 is the latest dataset.
                return
            formatted_month = "{:02d}".format(month) # Add leading zeroes
            dataset_name = "yellow_tripdata_{}-{}".format(year, formatted_month)
            filename = "{}.parquet".format(dataset_name)
            url = "{}{}".format(base_url, filename)
            if not is_dry_run:
                publish(filename, url)
            print('Sent: ', filename, url, topc_arn)

def lambda_handler(event, context):
    iterate_files()
    if is_dry_run:
        print ('Dry run complete. No messages published.')
        return
    print ('Sent all messages for yellow taxi to SNS.')
    

