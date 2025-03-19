# EV Adoption Analysis Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and visualization for the EV Adoption Analysis project.

## Notebooks

- `EV_Adoption_EDA.ipynb` - Initial exploratory data analysis of EV adoption data
- `Charging_Station_Analysis.ipynb` - Analysis of charging station data and infrastructure impacts
- `Prediction_Model_Development.ipynb` - Development and testing of ML models for predicting adoption rates

## Usage

These notebooks can be viewed and run using:

```bash
jupyter notebook
```

Or with JupyterLab:

```bash
jupyter lab
```

## Dependencies

The notebooks require the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- folium
- geopandas

All dependencies are listed in the root `requirements.txt` file.

## Data Sources

The notebooks access data from the `data/` directory in the project root.
- Raw data: `data/raw/`
- Processed data: `data/processed/` 