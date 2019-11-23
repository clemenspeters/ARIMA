import json
import urllib.parse
import boto3
from smart_open import open
import datetime
# import pandas as pd
import terminalColors as tc
import taxiDataFileProcessor

print('Loading function')

s3 = boto3.client('s3')


def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    s3bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    dataset = key.split('/')[1].split('.')[0]
    print('dataset', dataset)

    yellowTaxiProcessor = taxiDataFileProcessor.TaxiDataFileProcessor(
        s3bucket = s3bucket,
        key = key,
        pickup_index = 1,
        passenger_index = 3
    )

    yellowTaxiProcessor.process_file()

    # lineCount = 0
    # for line in open('s3://{}/{}'.format(bucket, key)):
    #     print(lineCount)
    #     print(repr(line))
    #     lineCount += 1
    #     if lineCount > 10:
    #         break

    # try:
    #     response = s3.get_object(Bucket=bucket, Key=key)
    #     print('key', key)
    #     print('dataset', dataset)
    #     print("CONTENT TYPE: " + response['ContentType'])
    #     return response['ContentType']
    # except Exception as e:
    #     print(e)
    #     print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
    #     raise e
