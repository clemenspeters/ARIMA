from __future__ import print_function

import json
import urllib
import boto3

print('Loading message function...')
topc_arn = 'arn:aws:sns:eu-central-1:348499957229:download-file'
sns = boto3.client('sns')

base_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'
years = range(2009, 2020)
months = range(1,13)
is_dry_run = False


def publish(subject, msg):
    sns.publish(
        TopicArn=topc_arn,
        Subject=subject,
        Message=msg
    )

def iterate_files():
    for year in years:
        for month in months:
            if year == 2019 and month > 6: # 2019-06 is the latest dataset.
                return
            formatted_month = "{:02d}".format(month) # Add leading zeroes
            dataset_name = "yellow_tripdata_{}-{}".format(year, formatted_month)
            filename = "{}.csv".format(dataset_name)
            url = "{}{}".format(base_url, filename)
            msg = 'https://s3.amazonaws.com/nyc-tlc/trip+data/{}'.format(filename)
            if not is_dry_run:
                publish(filename, msg)
            print('Sent: ', filename, url, topc_arn)

def lambda_handler(event, context):
    iterate_files()
    if is_dry_run:
        print ('Dry run complete. No messages published.')
        return
    print ('Sent all messages for yellow taxi to SNS.')
    

