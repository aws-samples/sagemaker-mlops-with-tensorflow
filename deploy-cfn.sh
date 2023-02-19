#!/bin/bash

# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

set -e

# This script will create an S3 seedcode bucket ad lambda layer files.
# It will zip files and load to the S3 bucket.
# It will also deploy the CloudFormation template that will provision AWS services.   

# This script assumes a Linux or MacOSX environment and relies on the following software packages being installed:
# . - AWS Command Line Interface (CLI)
# . - sed
# . - Python 3 / pip3
# . - zip
# . - Docker

date=`date +"%Y-%m-%d"`
today=$(date +"%Y-%m-%d")
echo "${today}"

#Provide your own parameters
SEED_BUCKET_NAME="tf-cicd-lambda-seed-${today}"
UPLOAD_DIR="upload_to_s3"
CFN_TEMPLATE="CF-MLOps-TensorFlow-Pipeline.yml"
DEPLOYMENT_REGION="us-east-1"

STACK_NAME="MLOps-TensorFlow-FrameWork-CICD"
#3-letter unique id. INPUT YOUR OWN ID
Unique_ID="<YOUR ID>"
KmsKey="YOUR KMS KEY"

UPLOAD_LST="MLOps-tf-EvaluateModel.py.zip MLOps-tf-DeployModel.py.zip MLOps-tf-GetStatus.py.zip MLOps-tf-PrepareData.py.zip MLOps-tf-TrainModel.py.zip pandas_layer.zip seedcode.zip"

echo "Create s3 bucket"
# Check that S3 bucket exists, if not create a new one
if aws s3 ls s3://${SEED_BUCKET_NAME} 2>&1 | grep NoSuchBucket
then
    echo Creating Amazon S3 bucket ${SEED_BUCKET_NAME}
    aws s3 mb s3://${SEED_BUCKET_NAME} --region $DEPLOYMENT_REGION
fi

echo "Create lambda layer files using docker"
#Use docker to create lambda layer file
docker run -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.7" /bin/sh -c "pip install -r requirements.txt -t lambda_layer/python/lib/python3.7/site-packages/; exit"

echo "Zip files"
mkdir -p ${UPLOAD_DIR}
(cd seedcode ; zip -r ../${UPLOAD_DIR}/seedcode.zip . )
(cd lambda_layer ; zip -r ../${UPLOAD_DIR}/pandas_layer.zip python )

(cd lambda
for fname in *.py;
do
    zip ../${UPLOAD_DIR}/${fname}.zip ${fname}  
done
)

# push files to S3, note this does not 'package' the templates
echo "Copying LAMBDA AND SEED files to S3: ${UPLOAD_LST}" 
for fname in ${UPLOAD_LST};
do
    aws s3 cp ${UPLOAD_DIR}/${fname} s3://${SEED_BUCKET_NAME}/${fname}  
done

echo "LAMBDA AND SEED CODE UPLOADED" 

aws cloudformation deploy \
        --template-file ${CFN_TEMPLATE} \
        --stack-name ${STACK_NAME} \
        --capabilities CAPABILITY_NAMED_IAM \
        --parameter-overrides "UniqueID"=${Unique_ID} "LambdaSeedBucket"=${SEED_BUCKET_NAME} "KmsKey"=${KmsKey}

echo ==================================================
echo "Deoployment complete."
