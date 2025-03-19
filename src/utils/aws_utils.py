"""
AWS Utilities Module for EV Adoption Analysis.
Provides utilities for interacting with AWS services.
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import io

logger = logging.getLogger("ev_adoption_analysis")

class AWSHandler:
    """
    Class for handling AWS interactions.
    """
    
    def __init__(self, aws_access_key=None, aws_secret_key=None, region_name='us-west-2'):
        """
        Initialize the AWSHandler.
        
        Args:
            aws_access_key (str, optional): AWS access key.
                                          If None, uses environment variables.
            aws_secret_key (str, optional): AWS secret key.
                                          If None, uses environment variables.
            region_name (str, optional): AWS region name.
                                      Defaults to 'us-west-2'.
        """
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region_name = region_name
        
        # Initialize AWS clients
        self.s3 = None
        self.rds = None
        
        logger.debug(f"Initialized AWSHandler with region: {region_name}")
    
    def _get_s3_client(self):
        """
        Get or create an S3 client.
        
        Returns:
            boto3.client: S3 client.
        """
        if self.s3 is None:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region_name
            )
        return self.s3
    
    def _get_rds_client(self):
        """
        Get or create an RDS client.
        
        Returns:
            boto3.client: RDS client.
        """
        if self.rds is None:
            self.rds = boto3.client(
                'rds',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region_name
            )
        return self.rds
    
    def upload_to_s3(self, file_path, bucket_name, object_key=None):
        """
        Upload a file to an S3 bucket.
        
        Args:
            file_path (str): Path to the file to upload.
            bucket_name (str): Name of the S3 bucket.
            object_key (str, optional): S3 object key (path in the bucket).
                                       If None, uses the file name.
                                       
        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        if object_key is None:
            object_key = os.path.basename(file_path)
            
        logger.info(f"Uploading {file_path} to S3 bucket {bucket_name} as {object_key}")
        
        s3_client = self._get_s3_client()
        
        try:
            s3_client.upload_file(file_path, bucket_name, object_key)
            logger.info(f"Successfully uploaded {file_path} to S3")
            return True
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False
    
    def download_from_s3(self, bucket_name, object_key, file_path=None):
        """
        Download a file from an S3 bucket.
        
        Args:
            bucket_name (str): Name of the S3 bucket.
            object_key (str): S3 object key (path in the bucket).
            file_path (str, optional): Path to save the file.
                                     If None, saves to a temporary file.
                                     
        Returns:
            str: Path to the downloaded file, or None if download failed.
        """
        if file_path is None:
            file_path = f"/tmp/{object_key.replace('/', '_')}"
            
        logger.info(f"Downloading {object_key} from S3 bucket {bucket_name} to {file_path}")
        
        s3_client = self._get_s3_client()
        
        try:
            s3_client.download_file(bucket_name, object_key, file_path)
            logger.info(f"Successfully downloaded {object_key} from S3")
            return file_path
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            return None
    
    def read_csv_from_s3(self, bucket_name, object_key):
        """
        Read a CSV file directly from S3 into a pandas DataFrame.
        
        Args:
            bucket_name (str): Name of the S3 bucket.
            object_key (str): S3 object key (path in the bucket).
            
        Returns:
            pandas.DataFrame: DataFrame containing the CSV data.
        """
        logger.info(f"Reading CSV {object_key} from S3 bucket {bucket_name}")
        
        s3_client = self._get_s3_client()
        
        try:
            # Get the object from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            
            # Read the CSV data directly into a pandas DataFrame
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            
            logger.info(f"Successfully read CSV with {len(df)} rows from S3")
            return df
        except ClientError as e:
            logger.error(f"Error reading CSV from S3: {str(e)}")
            return None
    
    def save_dataframe_to_s3(self, df, bucket_name, object_key):
        """
        Save a pandas DataFrame directly to a CSV file in S3.
        
        Args:
            df (pandas.DataFrame): DataFrame to save.
            bucket_name (str): Name of the S3 bucket.
            object_key (str): S3 object key (path in the bucket).
            
        Returns:
            bool: True if the save was successful, False otherwise.
        """
        logger.info(f"Saving DataFrame with {len(df)} rows to {object_key} in S3 bucket {bucket_name}")
        
        s3_client = self._get_s3_client()
        
        try:
            # Convert DataFrame to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Upload to S3
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=csv_buffer.getvalue()
            )
            
            logger.info(f"Successfully saved DataFrame to S3")
            return True
        except ClientError as e:
            logger.error(f"Error saving DataFrame to S3: {str(e)}")
            return False
    
    def list_s3_objects(self, bucket_name, prefix=''):
        """
        List objects in an S3 bucket with optional prefix.
        
        Args:
            bucket_name (str): Name of the S3 bucket.
            prefix (str, optional): Prefix to filter objects.
                                  Defaults to ''.
                                  
        Returns:
            list: List of object keys in the bucket.
        """
        logger.info(f"Listing objects in S3 bucket {bucket_name} with prefix '{prefix}'")
        
        s3_client = self._get_s3_client()
        
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            # Extract object keys
            if 'Contents' in response:
                object_keys = [obj['Key'] for obj in response['Contents']]
                logger.info(f"Found {len(object_keys)} objects in S3 bucket")
                return object_keys
            else:
                logger.info("No objects found in S3 bucket")
                return []
        except ClientError as e:
            logger.error(f"Error listing objects in S3 bucket: {str(e)}")
            return []
    
    def upload_model_to_s3(self, model_path, bucket_name, model_key=None):
        """
        Upload a trained model to S3.
        
        Args:
            model_path (str): Path to the saved model file.
            bucket_name (str): Name of the S3 bucket.
            model_key (str, optional): S3 object key for the model.
                                     If None, uses the model file name.
                                     
        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        if model_key is None:
            model_key = f"models/{os.path.basename(model_path)}"
            
        logger.info(f"Uploading model {model_path} to S3 bucket {bucket_name} as {model_key}")
        
        return self.upload_to_s3(model_path, bucket_name, model_key)
    
    def get_rds_instance_status(self, db_instance_identifier):
        """
        Get the status of an RDS database instance.
        
        Args:
            db_instance_identifier (str): RDS instance identifier.
            
        Returns:
            str: RDS instance status, or None if error.
        """
        logger.info(f"Getting status of RDS instance {db_instance_identifier}")
        
        rds_client = self._get_rds_client()
        
        try:
            response = rds_client.describe_db_instances(
                DBInstanceIdentifier=db_instance_identifier
            )
            
            # Extract instance status
            if 'DBInstances' in response and len(response['DBInstances']) > 0:
                status = response['DBInstances'][0]['DBInstanceStatus']
                logger.info(f"RDS instance status: {status}")
                return status
            else:
                logger.warning(f"No RDS instance found with identifier {db_instance_identifier}")
                return None
        except ClientError as e:
            logger.error(f"Error getting RDS instance status: {str(e)}")
            return None
    
    def create_ec2_client(self):
        """
        Create an EC2 client.
        
        Returns:
            boto3.client: EC2 client.
        """
        return boto3.client(
            'ec2',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region_name
        )
    
    def get_ec2_instance_status(self, instance_id):
        """
        Get the status of an EC2 instance.
        
        Args:
            instance_id (str): EC2 instance ID.
            
        Returns:
            str: EC2 instance status, or None if error.
        """
        logger.info(f"Getting status of EC2 instance {instance_id}")
        
        ec2_client = self.create_ec2_client()
        
        try:
            response = ec2_client.describe_instance_status(
                InstanceIds=[instance_id]
            )
            
            # Extract instance status
            if 'InstanceStatuses' in response and len(response['InstanceStatuses']) > 0:
                status = response['InstanceStatuses'][0]['InstanceState']['Name']
                logger.info(f"EC2 instance status: {status}")
                return status
            else:
                logger.warning(f"No EC2 instance found with ID {instance_id}")
                return None
        except ClientError as e:
            logger.error(f"Error getting EC2 instance status: {str(e)}")
            return None 