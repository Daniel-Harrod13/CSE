# EV Adoption Analysis Project

## Overview
This project analyzes electric vehicle (EV) adoption patterns across different regions, examining the relationship between EV adoption rates, charging infrastructure availability, economic factors, and environmental impact. The analysis provides insights into effective strategies for accelerating clean transportation transitions.

## Project Structure
- `data/` - Raw and processed datasets
  - `raw/` - Original data files
  - `processed/` - Cleaned and transformed data
- `notebooks/` - Jupyter notebooks for exploratory analysis
- `src/` - Python modules for data processing, analysis, and visualization
  - `data/` - Data loading and processing modules
  - `models/` - Machine learning models
  - `visualization/` - Visualization tools
  - `utils/` - Utility functions and helpers
- `models/` - Trained ML models
- `dashboard/` - Interactive Streamlit web dashboard
  - `assets/` - Visualization assets for the dashboard
- `docs/` - Documentation and methodology reports

## Features
- **Data Processing**: Automated ETL pipelines for cleaning and transforming EV adoption data
- **Statistical Analysis**: Comprehensive analysis of factors influencing EV adoption
- **Machine Learning**: Models for predicting adoption trends based on multiple factors
- **Geospatial Analysis**: Mapping of charging infrastructure distribution
- **Interactive Dashboard**: Streamlit-based web application for exploring the data
- **AWS Integration**: Cloud storage and processing capabilities

## Key Insights
- Charging infrastructure availability shows strong correlation with EV adoption rates
- Economic factors, particularly median income, significantly impact adoption patterns
- State incentives demonstrate effectiveness in accelerating EV adoption
- Environmental benefits vary by region due to differences in electricity generation sources

## Technologies Used
- **Python**: Core programming language
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning implementations
- **Matplotlib/Seaborn**: Data visualization
- **Geopandas/Folium**: Geospatial analysis and mapping
- **Streamlit**: Interactive dashboard development
- **AWS**: Cloud integration (S3, EC2, RDS)

## Getting Started
1. Clone the repository:
```
git clone https://github.com/yourusername/ev-adoption-analysis.git
cd ev-adoption-analysis
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the analysis pipeline:
```
python src/main.py
```

4. Launch the dashboard:
```
cd dashboard
streamlit run app.py
```

## Dashboard
The interactive dashboard provides visualization and exploration capabilities:
- Regional EV adoption trends
- Charging infrastructure analysis
- Economic and environmental factor relationships
- Predictive modeling and scenario analysis

## Data Sources
This project uses synthetic data that simulates real-world EV adoption patterns. In a real implementation, data could be sourced from:
- Vehicle registration databases
- Charging station networks
- Economic indicators from government sources
- Environmental impact metrics

## Future Work
- Integration with real-time data sources
- Expansion to more granular geographic analysis
- Enhanced predictive models using additional features
- Policy impact simulations

## License
This project is available under the MIT license.

## Contact
For more information, please contact: your.email@example.com 