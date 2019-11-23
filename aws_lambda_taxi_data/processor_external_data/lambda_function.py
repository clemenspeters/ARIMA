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
s3bucket = 'nyc-tlc' # bucket from where we download

def start_processor(filename):
    s3file = 'trip+data/{}'.format(filename)
    key = urllib.parse.unquote_plus(s3file, encoding='utf-8')
    dataset = key.split('/')[1].split('.')[0]
    print('dataset', dataset)

    yellowTaxiProcessor = taxiDataFileProcessor.TaxiDataFileProcessor(
        s3bucket = s3bucket,
        key = key,
        pickup_index = 1,
        passenger_index = 3
    )

    yellowTaxiProcessor.process_file()

def lambda_handler(event, context):
    url = event['Records'][0]['Sns']['Message']
    filename = event['Records'][0]['Sns']['Subject']
    print("From SNS url: " + url)
    print("From SNS filename: " + filename)
    start_processor(filename)