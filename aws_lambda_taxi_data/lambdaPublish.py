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
            # dry_run_download(url)
            msg = 'https://s3.amazonaws.com/nyc-tlc/trip+data/{}'.format(filename)
            publish(filename, msg)
            print('Sent: ', filename, url, topc_arn)

def lambda_handler(event, context):

    # This function receives JSON input with three fields: the ARN of an SNS topic,
    # a string with the subject of the message, and a string with the body of the message.
    # The message is then sent to the SNS topic.
    #
    # Example:
    #   {
    #       "topic": "arn:aws:sns:REGION:123456789012:MySNSTopic",
    #       "subject": "This is the subject of the message.",
    #       "message": "This is the body of the message."
    #   }

    # filename = 'yellow_tripdata_2009-03.csv'
    # msg = 'https://s3.amazonaws.com/nyc-tlc/trip+data/{}'.format(filename)
    # publish(filename, msg)
    # return ('Sent a message to an Amazon SNS topic.')

    iterate_files()
    return ('Sent all messages for yellow taxi to SNS.')
    