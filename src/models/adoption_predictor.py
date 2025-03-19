"""
Machine Learning Model Module for EV Adoption Prediction.
"""

import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ev_adoption_analysis")

class AdoptionPredictor:
    """
    Class for predicting EV adoption rates using machine learning models.
    """
    
    def __init__(self, model_type="random_forest"):
        """
        Initialize the AdoptionPredictor with a specified model type.
        
        Args:
            model_type (str, optional): Type of model to use.
                                        Choices: 'random_forest', 'gradient_boosting',
                                        'linear', 'ridge', 'lasso'.
                                        Defaults to "random_forest".
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.test_data = None
        self.test_predictions = None
        
        logger.debug(f"Initialized AdoptionPredictor with model_type={model_type}")
    
    def train(self, df, target_col='adoption_rate', test_size=0.2, random_state=42):
        """
        Train the machine learning model.
        
        Args:
            df (pandas.DataFrame): Processed data to train on.
            target_col (str, optional): Target column to predict.
                                       Defaults to 'adoption_rate'.
            test_size (float, optional): Proportion of data to use for testing.
                                        Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility.
                                         Defaults to 42.
                                         
        Returns:
            dict: Dictionary of model evaluation metrics.
        """
        logger.info(f"Training {self.model_type} model to predict {target_col}")
        
        # Prepare the data
        X, y, X_train, X_test, y_train, y_test = self._prepare_data(df, target_col, test_size, random_state)
        
        # Initialize and train the model
        self.model = self._get_model()
        
        # If using grid search for hyperparameter tuning
        if self.model_type == "random_forest":
            self._train_with_grid_search(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate the model
        self.metrics = self._evaluate_model(X_train, y_train, X_test, y_test)
        
        # Save test data and predictions for later use
        self.test_data = (X_test, y_test)
        self.test_predictions = self.model.predict(X_test)
        
        # Record feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 features: {top_features}")
        
        logger.info(f"Model training complete. Test R²: {self.metrics['test_r2']:.4f}")
        
        return self.metrics
    
    def _prepare_data(self, df, target_col, test_size, random_state):
        """
        Prepare the data for model training.
        
        Args:
            df (pandas.DataFrame): Input data.
            target_col (str): Target column name.
            test_size (float): Proportion of data for testing.
            random_state (int): Random state for reproducibility.
            
        Returns:
            tuple: (X, y, X_train, X_test, y_train, y_test)
        """
        logger.debug("Preparing data for model training")
        
        # Make a copy
        df_model = df.copy()
        
        # Separate features and target
        y = df_model[target_col]
        
        # Drop columns not useful for prediction
        cols_to_drop = [
            target_col,  # Target column
            'region',    # Categorical identifier
            'year',      # Temporal identifier
            'ev_sales',  # Raw count (used to create target)
            'total_car_sales',  # Raw count (used to create target)
            'previous_ev_sales'  # Used to create other features
        ]
        
        # Only drop columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in df_model.columns]
        
        X = df_model.drop(columns=cols_to_drop)
        
        # Check for any remaining categorical columns and convert them
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            logger.debug(f"One-hot encoding categorical columns: {cat_cols}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.debug(f"Data prepared: X shape={X.shape}, y shape={y.shape}")
        logger.debug(f"Train/test split: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X, y, X_train, X_test, y_train, y_test
    
    def _get_model(self):
        """
        Get the appropriate model based on model_type.
        
        Returns:
            object: Initialized model.
        """
        logger.debug(f"Initializing {self.model_type} model")
        
        if self.model_type == "random_forest":
            return RandomForestRegressor(random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(random_state=42)
        elif self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(random_state=42)
        elif self.model_type == "lasso":
            return Lasso(random_state=42)
        else:
            logger.warning(f"Unknown model type: {self.model_type}. Using default RandomForestRegressor.")
            return RandomForestRegressor(random_state=42)
    
    def _train_with_grid_search(self, X_train, y_train):
        """
        Train the model with grid search for hyperparameter tuning.
        
        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.
        """
        logger.info("Training model with grid search")
        
        if self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            # Default to a smaller grid for other models
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5]
            }
        
        # Create grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters from grid search: {grid_search.best_params_}")
    
    def _evaluate_model(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.
            X_test (pandas.DataFrame): Testing features.
            y_test (pandas.Series): Testing target.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        # Log the metrics
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (pandas.DataFrame): Features to predict on.
            
        Returns:
            numpy.ndarray: Predicted values.
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        logger.debug(f"Making predictions on data with shape {X.shape}")
        return self.model.predict(X)
    
    def predict_future(self, df, years_ahead=5):
        """
        Predict future EV adoption rates based on existing data.
        
        Args:
            df (pandas.DataFrame): Current data.
            years_ahead (int, optional): Number of years to predict.
                                        Defaults to 5.
                                        
        Returns:
            pandas.DataFrame: DataFrame with future predictions.
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        logger.info(f"Predicting adoption rates {years_ahead} years ahead")
        
        # Make a copy of the dataframe
        future_df = df.copy()
        
        # Get the regions and latest year for each region
        regions = future_df['region'].unique()
        max_year = future_df['year'].max()
        
        # Create a list to hold future data
        future_data = []
        
        for region in regions:
            # Get the latest data for this region
            latest_data = future_df[(future_df['region'] == region) & (future_df['year'] == max_year)].copy()
            
            # Project future years
            for i in range(1, years_ahead + 1):
                # Create a new row for this future year
                future_row = latest_data.copy()
                future_row['year'] = max_year + i
                
                # Update features that would naturally change over time
                # This is a simplistic approach; a real model would be more sophisticated
                
                # Assume median income grows by 2% per year
                future_row['median_income'] = latest_data['median_income'] * (1.02 ** i)
                
                # Assume electricity price grows by 1% per year
                future_row['electricity_price'] = latest_data['electricity_price'] * (1.01 ** i)
                
                # Assume gas price grows by 3% per year
                future_row['avg_gas_price'] = latest_data['avg_gas_price'] * (1.03 ** i)
                
                # Assume carbon intensity decreases by 2% per year
                future_row['carbon_intensity'] = latest_data['carbon_intensity'] * (0.98 ** i)
                
                # Assume charging stations increase by 20% per year
                future_row['charging_stations'] = latest_data['charging_stations'] * (1.2 ** i)
                
                # Update any derivative features
                future_row['gas_electricity_ratio'] = future_row['avg_gas_price'] / future_row['electricity_price']
                
                # Add to future data
                future_data.append(future_row)
        
        # Combine into a DataFrame
        future_df = pd.concat(future_data, ignore_index=True)
        
        # Prepare future data for prediction (similar to _prepare_data)
        # Here we need to ensure the columns match what the model was trained on
        X_future = future_df.drop(columns=['region', 'year', 'ev_sales', 'total_car_sales', 'previous_ev_sales', 'adoption_rate'])
        
        # Make predictions
        future_df['predicted_adoption_rate'] = self.model.predict(X_future)
        
        # Estimate EV sales based on predicted adoption rate
        # Assume total car sales decrease by 1% per year
        for i, year in enumerate(range(max_year + 1, max_year + years_ahead + 1)):
            year_mask = future_df['year'] == year
            future_df.loc[year_mask, 'total_car_sales'] = future_df.loc[year_mask, 'total_car_sales'] * (0.99 ** (i + 1))
            future_df.loc[year_mask, 'predicted_ev_sales'] = future_df.loc[year_mask, 'predicted_adoption_rate'] * future_df.loc[year_mask, 'total_car_sales'] / 100
        
        logger.info(f"Future predictions generated for {len(regions)} regions, {years_ahead} years ahead")
        return future_df
    
    def save_model(self, filename="adoption_predictor_model.joblib"):
        """
        Save the trained model to disk.
        
        Args:
            filename (str, optional): Filename to save the model.
                                    Defaults to "adoption_predictor_model.joblib".
                                    
        Returns:
            str: Path where the model was saved.
        """
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("No model to save. Train a model first.")
        
        # Get the current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the src directory
        src_dir = os.path.dirname(current_dir)
        # Get the project root directory
        project_dir = os.path.dirname(src_dir)
        # Set the models directory
        models_dir = os.path.join(project_dir, "models")
        
        # Ensure the directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Set the full file path
        model_path = os.path.join(models_dir, filename)
        
        # Save the model
        logger.info(f"Saving model to {model_path}")
        
        # Create a dictionary with all the model data
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved successfully to {model_path}")
        
        return model_path
    
    def load_model(self, filename="adoption_predictor_model.joblib"):
        """
        Load a trained model from disk.
        
        Args:
            filename (str, optional): Filename of the saved model.
                                    Defaults to "adoption_predictor_model.joblib".
                                    
        Returns:
            object: The loaded model.
        """
        # Get the current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the src directory
        src_dir = os.path.dirname(current_dir)
        # Get the project root directory
        project_dir = os.path.dirname(src_dir)
        # Set the models directory
        models_dir = os.path.join(project_dir, "models")
        
        # Set the full file path
        model_path = os.path.join(models_dir, filename)
        
        logger.info(f"Loading model from {model_path}")
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.metrics = model_data['metrics']
            self.feature_importance = model_data['feature_importance']
            
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 