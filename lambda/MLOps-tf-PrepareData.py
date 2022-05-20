# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: MIT-0
import os
import boto3
import numpy as np
from numpy import save
import pandas as pd
import json
from datetime import datetime as dt
from boto3.session import Session

region = boto3.session.Session().region_name

code_pipeline = boto3.client('codepipeline')

class DataPreparer:
    """
    A class to prepare raw data for machine learning training.

    Attributes
    ----------
    None

    Methods
    -------
    
    """

    def __init__(self):
        """
        Initialize an instance of DataPreparer.

        Attributes
        ----------
        
        """

        # AWS Resources
        self.bucket_name = None

        # Raw DataFrames
        self.train_df = None
        self.test_df = None

        #Sequence length
        self.sequence_length = 30
   
        # Output array
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test1 = None
        self.y_test1 = None
        self.x_test2 = None
        self.y_test2 = None

        # Data Resources
        self.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6',
                        's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
        self.num_cols = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']

    def load_raw_data(self):
        """Loads the raw train and test data into DataFrames for processing.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # read raw train data
        for i in range(1, 5):
            self.train_df = pd.read_csv(
                f's3://{self.bucket_name}/raw/train_FD00{i}.txt', delimiter=' ', header=None)

            self.train_df.drop(
                self.train_df.columns[[26, 27]], axis=1, inplace=True)
            self.train_df.columns = self.columns
        
        # read raw test data
        for i in range(1, 5):
            self.test_df = pd.read_csv(
                f's3://{self.bucket_name}/raw/test_FD00{i}.txt', delimiter=' ', header=None)

            self.test_df.drop(
                self.test_df.columns[[26, 27]], axis=1, inplace=True)
            self.test_df.columns = self.columns


    @staticmethod
    def generate_target(df):
        """Aggregate daily data into weekly data.

        Parameters
        ----------
        df (DataFrame): raw data

        Returns
        -------
        df (DataFrame): data with target

        """
        #Normalize data
        eps = 0.000001  # for floating point issues during normalization
        df.iloc[:, 2:] = (df.iloc[:, 2:]-df.iloc[:, 2:].min()+eps) / \
            (df.iloc[:, 2:].max()-df.iloc[:, 2:].min()+eps)

        # Data Labeling - generate target.
        rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        df = df.merge(rul, on=['id'], how='left')
        df['RUL'] = df['max'] - df['cycle']
        df.drop('max', axis=1, inplace=True)
        df['target'] = df['RUL'].apply(lambda x: 1 if x <= 14 else 0)

        return df

    # function to reshape features into (samples, time steps, features)

    def get_sequence(self,df, seq_length, feature_cols):
        data_array = df[feature_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]


    def get_targets(self,df, seq_length, target):
        data_array = df[target].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]
    
    def reconstruct_data_singleID(self,dataset, target, start_index, end_index, history_size,
                                  target_size, single_step=True):
        """
        reshape data for deep learning input.

        Parameters
        ----------
        dataset (array): raw data
        target (array): raw label
        start_index (int): starting index number
        history_size (int): window size of the sequence
        target_size (int): number of future steps to be predicted
        single_step (boolean): indicator of single predciton or multiple predictions

        Returns
        -------
        data (array): predictor varaibles
        labels (array): target label

        """
        data = []
        labels = []
    
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
       
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            data.append(dataset[indices])
       
            if single_step:
                labels.append([target[i+target_size]])
            else:
                labels.append([target[i:i+target_size]])
       
        return np.array(data), np.array(labels)
       
    # generator for the sequences for all IDs
    def reconstruct_data_mutipleID(self, raw_df, num_cols,history_size,target_size):
        """
        reshape data for multiple IDs.

        Parameters
        ----------
        raw_df (dataframe): raw data with label
        num_cols (list): numerical column names
        history_size (int): window size of the sequence
        target_size (int): number of future steps to be predicted
        Returns
        -------
        fea_data (array): predictor varaibles
        target (array): target label
        """
        y_data = []
        x_data = []
        for id in raw_df['id'].unique():
            df = raw_df[raw_df['id'] == id]
            x_, y_ = self.reconstruct_data_singleID(df[num_cols].values, df['target'].values, 0, None, history_size,
                                               target_size, single_step=True)
            x_data.append(x_)
            y_data.append(y_)
        target = np.concatenate(y_data).astype(np.float32)
        fea_data = np.concatenate(x_data).astype(np.float32)
        return fea_data, target


    def gen_sequence_list(self, df, sequence_length, num_cols):

        # get id that has record more than the sequence_length
        id_count1 = df.groupby("id").count()
        id_count1 = id_count1.reset_index()
        df_id = id_count1[id_count1['s1'] > sequence_length]
        df = df[df['id'].isin(df_id['id'])]
        
        # generator for the sequences for all IDs
        seq_gen1 = (list(self.get_sequence(df[df['id'] == id], sequence_length, num_cols)) for id in df['id'].unique())
        # generate sequences and convert to numpy array
        x_df = np.concatenate(list(seq_gen1)).astype(np.float32)


        # generate targets
        target1 = [self.get_targets(df[df['id'] == id], sequence_length, ['target'])
                   for id in df['id'].unique()]
        y_df = np.concatenate(target1).astype(np.float32)

        return x_df, y_df

    def train_data_prep(self):
        """
        Prepares training data

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        print("[INFO] Start training data preparation.")

        #generate target
        self.train_df = self.generate_target(self.train_df)
        self.test_df = self.generate_target(self.train_df)
        
        #split feature and target
        self.x_train, self.y_train = self.reconstruct_data_mutipleID(
            self.train_df, self.num_cols, self.sequence_length, 0 )

        self.x_val, self.y_val = self.reconstruct_data_mutipleID(
            self.test_df, self.num_cols, self.sequence_length, 0)
        
        #get small portion data for testing
        self.y_test2 = self.y_val[-30:-10]
        self.x_test2 = self.x_val[-30:-10]
        self.y_test1 = self.y_val[-20:-5]
        self.x_test1 = self.x_val[-20:-5]

        print("Finished preparing training data.")

    def save_npy(self,df, bucket, fname):
        """
        save array to S3 bucket
        """
        s3_resource = boto3.resource('s3')

        save(f'/tmp/{fname}.npy', df)
        filename = f'train/{fname}.npy'
        s3_resource.meta.client.upload_file(
            f'/tmp/{fname}.npy', bucket, filename)

    def write_output_to_s3(self):
        """
        Writes prepared data set to S3 for machine learning use.

        Parameters
        ----------
        self

        Returns
        -------
        None

        """
        self.save_npy(self.x_train, self.bucket_name, 'x_train')
        self.save_npy(self.y_train, self.bucket_name, 'y_train')
        self.save_npy(self.x_val, self.bucket_name, 'x_val')
        self.save_npy(self.y_val, self.bucket_name, 'y_val')
        self.save_npy(self.x_test1, self.bucket_name, 'x_test1')
        self.save_npy(self.y_test1, self.bucket_name, 'y_test1')
        self.save_npy(self.x_test2, self.bucket_name, 'x_test2')
        self.save_npy(self.y_test2, self.bucket_name, 'y_test2')
    

def lambda_handler(event, context):

    print(event)
    print('[INFO] Preparing data...')

    dataPreparer = DataPreparer()
    dataPreparer.bucket_name = os.environ['S3DataBucket']

    try:
        # Retrieve data from S3
        dataPreparer.load_raw_data()
        # Process Data
        dataPreparer.train_data_prep()
        #Save Data
        dataPreparer.write_output_to_s3()
        
        event['message'] = 'Raw data processed'
        
        put_job_success(event)
        write_job_info_s3(event)
    
    except Exception as e:
        print(e)
        print('[ERROR] Unable to prepare data job.')
        event['message'] = str(e)
        put_job_failure(event)

    return event

def put_job_success(event):

    print('[SUCCESS]Training Data Saved to S3 ')
    print(event['message'])
    code_pipeline.put_job_success_result(jobId=event['CodePipeline.job']['id'])


def put_job_failure(event):

    print('[FAILURE]Putting job failure')
    print(event['message'])
    code_pipeline.put_job_failure_result(jobId=event['CodePipeline.job']['id'], failureDetails={
                                         'message': event['message'], 'type': 'JobFailed'})
    return event


def write_job_info_s3(event):
    print(event)

    objectKey = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['objectKey']

    bucketname = event['CodePipeline.job']['data']['outputArtifacts'][0]['location']['s3Location']['bucketName']

    artifactCredentials = event['CodePipeline.job']['data']['artifactCredentials']

    artifactName = event['CodePipeline.job']['data']['outputArtifacts'][0]['name']

    # S3 Managed Key for Encryption
    S3SSEKey = os.environ['SSEKMSKeyIdIn']

    json_data = json.dumps(event)
    print(json_data)

    session = Session(aws_access_key_id=artifactCredentials['accessKeyId'],
                      aws_secret_access_key=artifactCredentials['secretAccessKey'],
                      aws_session_token=artifactCredentials['sessionToken'])

    s3 = session.resource("s3")
    #object = s3.Object(bucketname, objectKey + '/event.json')
    object = s3.Object(bucketname, objectKey)
    print(object)
    object.put(Body=json_data, ServerSideEncryption='aws:kms',
               SSEKMSKeyId=S3SSEKey)
    print('event written to s3')
