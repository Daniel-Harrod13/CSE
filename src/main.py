#!/usr/bin/env python3
"""
Main script for EV Adoption Analysis Project.
This script orchestrates the entire data analysis pipeline.
"""

import os
import sys
import logging
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.adoption_predictor import AdoptionPredictor
from src.visualization.visualizer import Visualizer
from src.utils.logger import setup_logger

def main():
    """
    Main function to run the EV adoption analysis pipeline.
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/run_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger("ev_adoption_analysis", log_file)
    
    logger.info("Starting EV Adoption Analysis Pipeline")
    
    try:
        # 1. Load the data
        logger.info("Loading data...")
        data_loader = DataLoader()
        ev_data = data_loader.load_ev_data()
        charging_data = data_loader.load_charging_station_data()
        
        # 2. Process the data
        logger.info("Processing data...")
        data_processor = DataProcessor()
        processed_data = data_processor.process(ev_data, charging_data)
        data_processor.save_processed_data(processed_data)
        
        # 3. Train prediction models
        logger.info("Training prediction models...")
        predictor = AdoptionPredictor()
        predictor.train(processed_data)
        predictor.save_model()
        
        # 4. Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = Visualizer()
        visualizer.create_adoption_trend_charts(processed_data)
        visualizer.create_correlation_heatmap(processed_data)
        visualizer.create_charging_station_map(charging_data)
        visualizer.create_prediction_plots(predictor, processed_data)
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise
    
if __name__ == "__main__":
    main() 