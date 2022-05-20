# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
import boto3
import csv
import botocore
from time import gmtime, strftime
from boto3.session import Session
import json
import os
import numpy as np
from io import BytesIO

sagemaker = boto3.client('sagemaker')
code_pipeline = boto3.client('codepipeline')

# ARN of IAM role Amazon SageMaker can assume to access model artifacts and docker image for deployment
SageMakerRole = os.environ['SageMakerExecutionRole']

#use json to send data to model and get back the prediction.
JSON_CONTENT_TYPE = "text/csv"

def lambda_handler(event, context):
    try:
        
        # Read In CodePipeline Data 
        #    - Previous Event Step Information = Resources created in the previous step (Ex. Hosting Endpoint)
        #    - User Parameters: This function accepts the following User Parameters from CodePipeline
        #         { "env": "Dev"}
        #             where: 
        #                  env = Environment, Valid Values (Dev, Test) 
        #                       
        previousStepEvent = read_job_info(event)
        endpointName = previousStepEvent['endpoint']
        print("[INFO]Endpoint:", endpointName)

        evalText = event['CodePipeline.job']['data']['actionConfiguration']['configuration']['UserParameters']
        test_info = json.loads(evalText)
            
        environment = test_info["env"]
        print("[INFO]ENVIRONMENT:", environment)
        
        # Environment variable containing S3 bucket for data used for validation and/or test
        data_bucket = os.environ['S3DataBucket']
        print("[INFO]DATA_BUCKET:", data_bucket)
        
         
        if environment == 'Dev':
            xkey = 'train/x_test1.npy'
            ykey = 'train/y_test1.npy'
            print("[INFO]Dev Test Info:"+ environment + " S3 Data Bucket: " + data_bucket + " S3 Prefix/Key: " + xkey+"/"+ykey)
            dev_eval = evaluate_model(data_bucket,xkey,ykey,endpointName)
            print('[SUCCESS] Dev Test Complete')
            write_job_info_s3(event)
            put_job_success(event)
        
        elif environment == 'Test':
            xkey = 'train/x_test2.npy'
            ykey = 'train/y_test2.npy'
            print("[INFO]Full Test Info:"+ environment + " S3 Data Bucket: " + data_bucket + " S3 Prefix/Key: " + xkey+"/"+ykey)
            test_eval = evaluate_model(data_bucket,xkey,ykey,endpointName)
            print('[SUCCESS] Full Test Complete')
            write_job_info_s3(event)
            put_job_success(event)
    
    except Exception as e:
        print(e)
        print('[ERROR]Unable to successfully invoke endpoint')
        event['message'] = str(e)
        put_job_failure(event)

    return event 

#Get test/validation data
def evaluate_model(data_bucket, xkey, ykey, endpointName):
    # Get the object from s3
    
    s3_client = boto3.client('s3')  # low-level functional API
    #s3_resource = boto3.resource('s3')  # high-level object-oriented API
    # subsitute this for your s3 bucket name.
    #my_bucket = s3_resource.Bucket(data_bucket)
    x_obj = s3_client.get_object(Bucket=data_bucket, Key=xkey)
    x_test = np.load(BytesIO(x_obj['Body'].read()))

    y_obj = s3_client.get_object(Bucket=data_bucket, Key=ykey)
    y_test = np.load(BytesIO(y_obj['Body'].read()))

    print("[INFO]Test data array loaded")

    try:
        #Use sagemaker runtime to make predictions after getting data
        runtime_client = boto3.client('runtime.sagemaker')
        EndpointInput=endpointName

        print ("[INFO]Endpoint Version:", endpointName)

        label_value = y_test.tolist()
        body=x_test
        payload = json.dumps(body.tolist())
        
        response = runtime_client.invoke_endpoint(
            ContentType="application/json",
            Body=payload,
            EndpointName=EndpointInput
            )
        
        print("[INFO]InvokeEndpoint Response:", response)
        #Check for successful return code (200)
        return_code = response['ResponseMetadata']['HTTPStatusCode']
        print("[INFO]InvokeEndpoint return_code:", return_code)
              
        if return_code != 200:
            #event['message'] = str(return_code)
            print("[FAIL] Test")
            #put_job_failure(event)
            return 'failed'
        elif return_code == 200:
            print('[INFO]Predictions Processed')
            result = json.loads(response['Body'].read().decode())
            actual_response = result['predictions']
            print('[INFO]Actual_response', actual_response)
            basic_metric = process_prediction(label_value, actual_response)
                
    except botocore.exceptions.ClientError as e:
        print(e)
        print('[ERRORUnable to get predictions')
        #event['message'] = str(e)
        put_job_failure(e)
        
    return return_code
            
def process_prediction(label_list, actual_list):
    #PostProcessing - Because we chose binary:logistic as our objective metric, our result for the payload will be
    # an output probability. We are going to use the same optimal cutoff detailed in the example notebook; however, this
    # would be configurable based on post-processing evaluation of impact. 
    #This processing could alternatively be setup as a post-processing container behind the hosted endpoint using
    #inference pipeline capabilities.
    
    for i in range(len(label_list)):
        label_value=label_list[i][0]
        actual_response=actual_list[i][0]

        response_cutoff = 0.46   
        #label = float(label_value)
        #print("Label:", label)
        predict_value = float(actual_response) 
        print("predict_value", predict_value)
        
        if predict_value > response_cutoff:
            prediction = '1'
            print('[INFO]Prediction is:', prediction)
        else:
            prediction = '0'
            print('[INFO]Prediction is:', prediction)

        label_value=str(round(label_value))
        labeltype = type(label_value)
        print("LabelType", labeltype)
        predictiontype = type(prediction)
        print("PredictionType", predictiontype)
        
        if label_value == '0' and prediction == '0':
            # True Negative
            basic_metric = 'TN'
        elif label_value == '0' and prediction == '1':
            # False Positive
            basic_metric = 'FP'
        elif label_value == '1' and prediction == '0':
            # False Negative
            basic_metric = 'FN'
        else:
            # True Positive
            basic_metric = 'TP'
            
        print("[INFO] Label:" + label_value + "| Prediction:" + prediction + " | Metric Response:" + basic_metric)
                    
        return basic_metric
                           
def write_job_info_s3(event):
    
    KMSKeyIdSSEIn = os.environ['SSEKMSKeyIdIn']
    
    objectKey = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['objectKey']
    bucketname = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['bucketName']

    artifactCredentials = event['CodePipeline.job']['data']['artifactCredentials']
    artifactName = event['CodePipeline.job']['data']['outputArtifacts'][0]['name']

    json_data = json.dumps(event)
    
    print(json_data)

    session = Session(aws_access_key_id=artifactCredentials['accessKeyId'],
                  aws_secret_access_key=artifactCredentials['secretAccessKey'],
                  aws_session_token=artifactCredentials['sessionToken'])
   

    s3 = session.resource("s3")
    object = s3.Object(bucketname, objectKey + '/event.json')
    object = s3.Object(bucketname, objectKey)
    print(object)
    object.put(Body=json_data, ServerSideEncryption='aws:kms', SSEKMSKeyId=KMSKeyIdSSEIn)

def read_job_info(event):

    print("[DEBUG]EVENT IN:", event)
    bucketname = event['CodePipeline.job']['data']['inputArtifacts'][0]['location']['s3Location']['bucketName']
    print("[INFO]Previous Job Info Bucket:", bucketname)
    
    objectKey = event['CodePipeline.job']['data']['inputArtifacts'][0]['location']['s3Location']['objectKey']
    print("[INFO]Previous Job Info Object:", objectKey)

    artifactCredentials = event['CodePipeline.job']['data']['artifactCredentials']

    session = Session(aws_access_key_id=artifactCredentials['accessKeyId'],
                  aws_secret_access_key=artifactCredentials['secretAccessKey'],
                  aws_session_token=artifactCredentials['sessionToken'])
   
 
    s3 = session.resource('s3')

    obj = s3.Object(bucketname,objectKey)
  
    item = json.loads(obj.get()['Body'].read().decode('utf-8'))
    
    print("[INFO]Previous CodePipeline Job Info Sucessfully Read:", item)
    return item

def put_job_success(event):
    #print('Putting job success')
    print("[PASS] Test")
    #print(event['message'])
    code_pipeline.put_job_success_result(jobId=event['CodePipeline.job']['id'])
  
def put_job_failure(event):
    
    print('Putting job failure')
    print(event['message'])
    event['successful_inferences'] = 'Inferences Successfully Passed Test'
    code_pipeline.put_job_failure_result(jobId=event['CodePipeline.job']['id'], failureDetails={'message': event['message'], 'type': 'JobFailed'})

def load_s3data(bucket, key):
    s3 = boto3.resource('s3')
    try:
        s3.Bucket(bucket).download_file(
            key, key)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
