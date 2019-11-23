import json
import urllib.parse
import boto3
import botocore.vendored.requests.packages.urllib3 as urllib3
import urllib.request

print('Loading function')
base_url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/'
s3 = boto3.client('s3')


def download_url(url, filename):
    try:
        with urllib.request.urlopen(url) as response:
            if(response.code == 200):
                print(response.code, "downloading", url)
                # urllib.request.urlretrieve(url, filename)
                store_to_s3(url, filename)
                print("Completed", url)
    except urllib.error.HTTPError as err:
        print(err.code, url)

def store_to_s3(url, filename):
    bucket = 'timeseries-anomaly' #your s3 bucket
    key = 'data/{}'.format(filename) #your desired s3 path or filename

    s3=boto3.client('s3')
    http=urllib3.PoolManager()
    s3.upload_fileobj(http.request('GET', url,preload_content=False), bucket, key)


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))
    url = event['Records'][0]['Sns']['Message']
    filename = event['Records'][0]['Sns']['Subject']
    print("From SNS url: " + url)
    print("From SNS filename: " + filename)
    download_url(url, filename)
