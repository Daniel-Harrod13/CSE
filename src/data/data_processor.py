"""
Data Processor Module for EV Adoption Analysis.
Handles data cleaning, transformation, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ev_adoption_analysis")

class DataProcessor:
    """
    Class for processing and transforming EV adoption data.
    """
    
    def __init__(self):
        """
        Initialize the DataProcessor.
        """
        self.scaler = StandardScaler()
        
    def process(self, ev_data, charging_data):
        """
        Process and transform the raw data.
        
        Args:
            ev_data (pandas.DataFrame): Raw EV adoption data.
            charging_data (pandas.DataFrame): Raw charging station data.
            
        Returns:
            pandas.DataFrame: Processed and transformed data.
        """
        logger.info("Processing and transforming data")
        
        # 1. Clean the data
        ev_cleaned = self._clean_ev_data(ev_data)
        charging_cleaned = self._clean_charging_data(charging_data)
        
        # 2. Transform the data
        ev_transformed = self._transform_ev_data(ev_cleaned)
        
        # 3. Feature engineering
        enriched_data = self._engineer_features(ev_transformed, charging_cleaned)
        
        # 4. Normalize/standardize the data
        final_data = self._normalize_data(enriched_data)
        
        return final_data
    
    def _clean_ev_data(self, df):
        """
        Clean the EV adoption data.
        
        Args:
            df (pandas.DataFrame): Raw EV adoption data.
            
        Returns:
            pandas.DataFrame: Cleaned EV adoption data.
        """
        logger.info("Cleaning EV adoption data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        logger.debug(f"Missing values before cleaning: {missing_values}")
        
        # Fill missing values (if any)
        if missing_values.sum() > 0:
            # For numeric columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.debug(f"Filled {col} missing values with median: {median_val}")
            
            # For categorical columns, fill with mode
            cat_cols = df_clean.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    logger.debug(f"Filled {col} missing values with mode: {mode_val}")
        
        # Check for duplicates
        duplicate_count = df_clean.duplicated().sum()
        if duplicate_count > 0:
            logger.debug(f"Found {duplicate_count} duplicate rows")
            df_clean = df_clean.drop_duplicates()
            logger.debug(f"Removed {duplicate_count} duplicate rows")
        
        logger.info(f"EV data cleaning complete: {len(df)} rows -> {len(df_clean)} rows")
        return df_clean
    
    def _clean_charging_data(self, df):
        """
        Clean the charging station data.
        
        Args:
            df (pandas.DataFrame): Raw charging station data.
            
        Returns:
            pandas.DataFrame: Cleaned charging station data.
        """
        logger.info("Cleaning charging station data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        logger.debug(f"Missing values before cleaning: {missing_values}")
        
        # Fill missing values (if any)
        if missing_values.sum() > 0:
            # For numeric columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    logger.debug(f"Filled {col} missing values with median: {median_val}")
            
            # For categorical columns, fill with mode
            cat_cols = df_clean.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    logger.debug(f"Filled {col} missing values with mode: {mode_val}")
        
        # Parse installation date
        df_clean['installation_date'] = pd.to_datetime(df_clean['installation_date'])
        
        logger.info(f"Charging data cleaning complete: {len(df)} rows -> {len(df_clean)} rows")
        return df_clean
    
    def _transform_ev_data(self, df):
        """
        Transform the EV adoption data.
        
        Args:
            df (pandas.DataFrame): Cleaned EV adoption data.
            
        Returns:
            pandas.DataFrame: Transformed EV adoption data.
        """
        logger.info("Transforming EV adoption data")
        
        # Make a copy
        df_transform = df.copy()
        
        # Calculate EV adoption rate (%)
        df_transform['adoption_rate'] = df_transform['ev_sales'] / df_transform['total_car_sales'] * 100
        logger.debug("Added 'adoption_rate' column")
        
        # Calculate year-over-year growth
        df_transform = df_transform.sort_values(['region', 'year'])
        df_transform['previous_ev_sales'] = df_transform.groupby('region')['ev_sales'].shift(1)
        df_transform['yoy_growth'] = (df_transform['ev_sales'] - df_transform['previous_ev_sales']) / df_transform['previous_ev_sales'] * 100
        # Fill NaN values for the first year in each region
        df_transform['yoy_growth'] = df_transform['yoy_growth'].fillna(0)
        logger.debug("Added 'yoy_growth' column")
        
        # Calculate the ratio of charging stations to EVs sold
        df_transform['charging_ev_ratio'] = df_transform['charging_stations'] / df_transform['ev_sales'] * 1000
        logger.debug("Added 'charging_ev_ratio' column")
        
        logger.info("EV data transformation complete")
        return df_transform
    
    def _engineer_features(self, ev_df, charging_df):
        """
        Engineer new features from the transformed data.
        
        Args:
            ev_df (pandas.DataFrame): Transformed EV adoption data.
            charging_df (pandas.DataFrame): Cleaned charging station data.
            
        Returns:
            pandas.DataFrame: Data with engineered features.
        """
        logger.info("Engineering new features")
        
        # Make a copy
        df = ev_df.copy()
        
        # Calculate the gas to electricity price ratio (proxy for cost savings)
        df['gas_electricity_ratio'] = df['avg_gas_price'] / df['electricity_price']
        logger.debug("Added 'gas_electricity_ratio' column")
        
        # Calculate affordability index (median income / average EV price)
        # Note: We don't have EV price in our data, so we'll use a proxy
        # For simplicity, let's assume avg EV price is $40k in 2018 with 5% inflation per year
        base_price = 40000
        inflation_rate = 0.05
        for year in df['year'].unique():
            years_since_base = year - 2018
            price_factor = (1 + inflation_rate) ** years_since_base
            df.loc[df['year'] == year, 'avg_ev_price'] = base_price * price_factor
        
        df['affordability_index'] = df['median_income'] / df['avg_ev_price']
        logger.debug("Added 'affordability_index' column")
        
        # Add charging infrastructure data
        # Calculate chargers per region and year
        charging_df['year'] = pd.DatetimeIndex(charging_df['installation_date']).year
        chargers_by_region_year = charging_df.groupby(['region', 'year']).size().reset_index(name='new_chargers')
        
        # Calculate charging speed metrics
        charging_speeds = charging_df.groupby(['region', 'year']).agg({
            'power_level': ['mean', 'median', 'max'],
            'available_ports': 'sum',
            'avg_daily_usage': 'mean'
        }).reset_index()
        
        # Flatten multi-index columns
        charging_speeds.columns = ['region', 'year', 'avg_power', 'median_power', 'max_power', 'total_ports', 'avg_usage']
        
        # Merge with main dataset
        df = pd.merge(df, chargers_by_region_year, on=['region', 'year'], how='left')
        df = pd.merge(df, charging_speeds, on=['region', 'year'], how='left')
        
        # Fill any missing values from the merge
        df.fillna({
            'new_chargers': 0,
            'avg_power': df['avg_power'].mean(),
            'median_power': df['median_power'].mean(),
            'max_power': df['max_power'].mean(),
            'total_ports': df['total_ports'].mean(),
            'avg_usage': df['avg_usage'].mean()
        }, inplace=True)
        
        logger.debug("Added charging infrastructure metrics")
        
        # Calculate charger utilization rate
        df['charger_utilization'] = df['avg_usage'] * df['total_ports'] / (df['charging_stations'] * 24)
        logger.debug("Added 'charger_utilization' column")
        
        # Create binary feature for states with incentives
        df['has_incentive'] = (df['incentive_amount'] > 0).astype(int)
        logger.debug("Added 'has_incentive' column")
        
        logger.info("Feature engineering complete")
        return df
    
    def _normalize_data(self, df):
        """
        Normalize or standardize the data.
        
        Args:
            df (pandas.DataFrame): Data with engineered features.
            
        Returns:
            pandas.DataFrame: Normalized data.
        """
        logger.info("Normalizing data")
        
        # Make a copy
        df_norm = df.copy()
        
        # Select numeric columns to normalize
        numeric_cols = df_norm.select_dtypes(include=np.number).columns.tolist()
        
        # Remove columns we don't want to normalize
        cols_to_exclude = ['year', 'ev_sales', 'total_car_sales', 'previous_ev_sales', 'has_incentive']
        numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        # Fit and transform the data
        logger.debug(f"Normalizing columns: {numeric_cols}")
        df_norm[numeric_cols] = self.scaler.fit_transform(df_norm[numeric_cols])
        
        logger.info("Data normalization complete")
        return df_norm
    
    def save_processed_data(self, df, filename="processed_ev_data.csv"):
        """
        Save the processed data to a CSV file.
        
        Args:
            df (pandas.DataFrame): Processed data to save.
            filename (str, optional): Name of the file to save to.
                                    Defaults to "processed_ev_data.csv".
        """
        # Get the current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the src directory
        src_dir = os.path.dirname(current_dir)
        # Get the project root directory
        project_dir = os.path.dirname(src_dir)
        # Set the data directory
        data_dir = os.path.join(project_dir, "data", "processed")
        
        # Ensure the directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Set the full file path
        file_path = os.path.join(data_dir, filename)
        
        logger.info(f"Saving processed data to {file_path}")
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved {len(df)} records to {filename}")
        
        # Also save a copy with the scaler
        if hasattr(self, 'scaler') and self.scaler is not None:
            import joblib
            scaler_path = os.path.join(data_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
            
        return file_path 