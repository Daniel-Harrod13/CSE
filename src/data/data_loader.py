"""
Data Loader Module for EV Adoption Analysis.
Handles loading data from various sources.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger("ev_adoption_analysis")

class DataLoader:
    """
    Class for loading EV adoption and charging station data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the DataLoader with a data directory.
        
        Args:
            data_dir (str, optional): Directory containing the data files.
                                      If None, uses the default data directory.
        """
        if data_dir is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Get the parent directory (src)
            src_dir = os.path.dirname(current_dir)
            # Get the parent directory (project root)
            project_dir = os.path.dirname(src_dir)
            # Set the data directory
            self.data_dir = os.path.join(project_dir, "data")
        else:
            self.data_dir = data_dir
            
        logger.debug(f"Data directory set to: {self.data_dir}")
    
    def load_ev_data(self, filename="ev_adoption_data.csv"):
        """
        Load EV adoption data from CSV file.
        
        Args:
            filename (str, optional): Name of the file to load.
                                     Defaults to "ev_adoption_data.csv".
                                     
        Returns:
            pandas.DataFrame: DataFrame containing the EV adoption data.
        """
        file_path = os.path.join(self.data_dir, "raw", filename)
        logger.info(f"Loading EV adoption data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from {filename}")
            logger.debug(f"EV data columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_charging_station_data(self, filename="charging_stations_geo.csv"):
        """
        Load charging station geographic data from CSV file.
        
        Args:
            filename (str, optional): Name of the file to load.
                                     Defaults to "charging_stations_geo.csv".
                                     
        Returns:
            pandas.DataFrame: DataFrame containing the charging station data.
        """
        file_path = os.path.join(self.data_dir, "raw", filename)
        logger.info(f"Loading charging station data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from {filename}")
            logger.debug(f"Charging station data columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_from_aws_s3(self, bucket_name, key, aws_access_key=None, aws_secret_key=None):
        """
        Load data from AWS S3 bucket.
        This is a placeholder method that demonstrates AWS integration capability.
        
        Args:
            bucket_name (str): Name of the S3 bucket.
            key (str): Key of the object to load.
            aws_access_key (str, optional): AWS access key. 
                                          If None, uses environment variables.
            aws_secret_key (str, optional): AWS secret key.
                                          If None, uses environment variables.
                                          
        Returns:
            pandas.DataFrame: DataFrame containing the loaded data.
        """
        logger.info(f"Loading data from S3 bucket: {bucket_name}, key: {key}")
        
        try:
            # This is a placeholder. In a real implementation, we would use boto3
            # to load data from S3.
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            
            # Download the file to a temporary location
            tmp_file = "/tmp/s3_data.csv"
            s3.download_file(bucket_name, key, tmp_file)
            
            # Load the data
            df = pd.read_csv(tmp_file)
            os.remove(tmp_file)  # Clean up
            
            logger.info(f"Successfully loaded {len(df)} records from S3")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from S3: {str(e)}")
            raise
    
    def load_from_database(self, connection_string, query):
        """
        Load data from a database.
        This is a placeholder method that demonstrates database integration.
        
        Args:
            connection_string (str): Database connection string.
            query (str): SQL query to execute.
            
        Returns:
            pandas.DataFrame: DataFrame containing the query results.
        """
        logger.info("Loading data from database")
        
        try:
            # This is a placeholder. In a real implementation, we would use
            # SQLAlchemy to connect to the database and execute the query.
            from sqlalchemy import create_engine
            
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            
            logger.info(f"Successfully loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise 