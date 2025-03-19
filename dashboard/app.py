#!/usr/bin/env python3
"""
Streamlit Dashboard for EV Adoption Analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.models.adoption_predictor import AdoptionPredictor

# Set page config
st.set_page_config(
    page_title="EV Adoption Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for the dashboard
def load_data():
    """
    Load the processed data.
    
    Returns:
        pandas.DataFrame: Processed EV adoption data.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_dir = os.path.dirname(current_dir)
    # Set the data directory
    data_dir = os.path.join(project_dir, "data", "processed")
    
    # Load the data
    data_path = os.path.join(data_dir, "processed_ev_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Processed data file not found. Please run the analysis pipeline first.")
        return None

def load_charging_station_data():
    """
    Load the charging station data.
    
    Returns:
        pandas.DataFrame: Charging station data.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_dir = os.path.dirname(current_dir)
    # Set the data directory
    data_dir = os.path.join(project_dir, "data", "raw")
    
    # Load the data
    data_path = os.path.join(data_dir, "charging_stations_geo.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Charging station data file not found.")
        return None

def load_model():
    """
    Load the trained model.
    
    Returns:
        AdoptionPredictor: Loaded model.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_dir = os.path.dirname(current_dir)
    # Set the models directory
    models_dir = os.path.join(project_dir, "models")
    
    # Load the model
    model_path = os.path.join(models_dir, "adoption_predictor_model.joblib")
    if os.path.exists(model_path):
        try:
            predictor = AdoptionPredictor()
            predictor.load_model()
            return predictor
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.warning("Model file not found. Please run the analysis pipeline first to train the model.")
        return None

def create_charging_station_map(df):
    """
    Create an interactive map showing charging station locations.
    
    Args:
        df (pandas.DataFrame): Charging station data with geographic coordinates.
        
    Returns:
        folium.Map: Interactive map.
    """
    # Create a base map centered on the US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='cartodbpositron')
    
    # Create a feature group for each type of charger
    dc_fast = folium.FeatureGroup(name="DC Fast Chargers")
    level_2 = folium.FeatureGroup(name="Level 2 Chargers")
    
    # Add markers for each charging station
    for _, row in df.iterrows():
        # Create popup content
        popup_content = f"""
            <b>{row['station_id']}</b><br>
            <b>Type:</b> {row['charger_type']}<br>
            <b>Power:</b> {row['power_level']} kW<br>
            <b>Installed:</b> {row['installation_date']}<br>
            <b>Operator:</b> {row['operator']}<br>
            <b>Ports:</b> {row['available_ports']}<br>
            <b>Avg Daily Usage:</b> {row['avg_daily_usage']} sessions
        """
        
        # Create a marker with popup
        if row['charger_type'] == 'DC Fast Charger':
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=int(row['power_level']/30) + 3,  # Size based on power level
                color='#e53935',
                fill=True,
                fill_color='#e53935',
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(dc_fast)
        else:  # Level 2
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=int(row['power_level']/5) + 2,  # Size based on power level
                color='#1e88e5',
                fill=True,
                fill_color='#1e88e5',
                fill_opacity=0.7,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(level_2)
    
    # Add feature groups to the map
    dc_fast.add_to(m)
    level_2.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    """
    Main function for the Streamlit dashboard.
    """
    # Load CSS
    # Customize the dashboard with CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            color: #333;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: #555;
        }
        .insight-box {
            background-color: #f1f7ff;
            border-left: 5px solid #1E88E5;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .highlight {
            color: #1E88E5;
            font-weight: bold;
        }
        .footer {
            font-size: 0.8rem;
            color: #888;
            text-align: center;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>EV Adoption Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
        <div class="insight-box">
        This dashboard presents an analysis of electric vehicle (EV) adoption patterns across different US regions.
        The analysis examines relationships between EV adoption rates, charging infrastructure, economic factors,
        and environmental impact. It provides insights for accelerating clean transportation transitions.
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    charging_data = load_charging_station_data()
    model = load_model()
    
    if data is None:
        st.stop()
    
    # Sidebar with filters
    st.sidebar.title("Filters & Settings")
    
    # Region filter
    regions = sorted(data['region'].unique())
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=regions,
        default=regions
    )
    
    # Year range filter
    years = sorted(data['year'].unique())
    min_year, max_year = min(years), max(years)
    selected_year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Filter data
    filtered_data = data[
        (data['region'].isin(selected_regions)) &
        (data['year'] >= selected_year_range[0]) &
        (data['year'] <= selected_year_range[1])
    ]
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Adoption Trends", 
        "Charging Infrastructure", 
        "Economic & Environmental Factors",
        "Predictions & Insights"
    ])
    
    # Tab 1: Adoption Trends
    with tab1:
        st.markdown("<h2 class='sub-header'>EV Adoption Trends</h2>", unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        # Latest year metrics
        latest_year = filtered_data['year'].max()
        latest_data = filtered_data[filtered_data['year'] == latest_year]
        
        with col1:
            total_ev_sales = latest_data['ev_sales'].sum()
            st.metric(
                label=f"Total EV Sales ({latest_year})",
                value=f"{total_ev_sales:,}"
            )
        
        with col2:
            avg_adoption_rate = latest_data['adoption_rate'].mean()
            st.metric(
                label=f"Average Adoption Rate ({latest_year})",
                value=f"{avg_adoption_rate:.2f}%"
            )
        
        with col3:
            avg_yoy_growth = latest_data['yoy_growth'].mean()
            st.metric(
                label=f"Average YoY Growth ({latest_year})",
                value=f"{avg_yoy_growth:.2f}%"
            )
        
        # Adoption trend charts
        st.markdown("<h3 class='section-header'>Adoption Rate Trends by Region</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/adoption_rate_trend.png")
        
        st.markdown("<h3 class='section-header'>Year-over-Year Growth</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/yoy_growth_trend.png")
        
        st.markdown("<h3 class='section-header'>EV Sales Volume by Region</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/ev_sales_by_region.png")
    
    # Tab 2: Charging Infrastructure
    with tab2:
        st.markdown("<h2 class='sub-header'>Charging Infrastructure Analysis</h2>", unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        # Latest year metrics
        with col1:
            total_stations = latest_data['charging_stations'].sum()
            st.metric(
                label=f"Total Charging Stations ({latest_year})",
                value=f"{total_stations:,}"
            )
        
        with col2:
            stations_per_ev = total_stations / (total_ev_sales or 1) * 1000
            st.metric(
                label=f"Stations per 1,000 EVs ({latest_year})",
                value=f"{stations_per_ev:.2f}"
            )
        
        with col3:
            avg_growth = (latest_data['charging_stations'].sum() / 
                         filtered_data[filtered_data['year'] == latest_year-1]['charging_stations'].sum() - 1) * 100
            st.metric(
                label=f"YoY Charging Infrastructure Growth",
                value=f"{avg_growth:.2f}%"
            )
        
        # Infrastructure trend charts
        st.markdown("<h3 class='section-header'>Charging Infrastructure Growth</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/charging_infrastructure_trend.png")
        
        st.markdown("<h3 class='section-header'>Relationship Between Charging Infrastructure and Adoption</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/charging_vs_adoption_scatter.png")
        
        # Map of charging stations
        if charging_data is not None:
            st.markdown("<h3 class='section-header'>Interactive Map of Charging Stations</h3>", unsafe_allow_html=True)
            
            # Filter charging data if needed
            filtered_charging = charging_data
            if selected_regions and len(selected_regions) < len(regions):
                filtered_charging = charging_data[charging_data['region'].isin(selected_regions)]
            
            # Create and display the map
            map_obj = create_charging_station_map(filtered_charging)
            folium_static(map_obj, width=1000, height=600)
        
    # Tab 3: Economic & Environmental Factors
    with tab3:
        st.markdown("<h2 class='sub-header'>Economic & Environmental Factors</h2>", unsafe_allow_html=True)
        
        # Correlation heatmap
        st.markdown("<h3 class='section-header'>Correlation Between Key Factors</h3>", unsafe_allow_html=True)
        st.image("dashboard/assets/correlation_heatmap.png")
        
        # Key factor analysis
        st.markdown("<h3 class='section-header'>Factor Analysis</h3>", unsafe_allow_html=True)
        
        # Select factors to plot
        factors = [
            'median_income', 'electricity_price', 'avg_gas_price', 
            'carbon_intensity', 'population_density', 'incentive_amount'
        ]
        factor_names = {
            'median_income': 'Median Income ($)',
            'electricity_price': 'Electricity Price ($/kWh)',
            'avg_gas_price': 'Gas Price ($/gallon)',
            'carbon_intensity': 'Carbon Intensity (g CO2/kWh)',
            'population_density': 'Population Density (people/sq mi)',
            'incentive_amount': 'State Incentive Amount ($)'
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_factor = st.selectbox(
                "X-Axis Factor",
                options=factors,
                format_func=lambda x: factor_names.get(x, x),
                index=0
            )
        
        with col2:
            y_factor = st.selectbox(
                "Y-Axis Factor",
                options=factors,
                format_func=lambda x: factor_names.get(x, x),
                index=5
            )
        
        # Plot the relationship
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for region in selected_regions:
            region_data = filtered_data[filtered_data['region'] == region]
            ax.scatter(
                region_data[x_factor], 
                region_data[y_factor],
                label=region,
                alpha=0.7,
                s=region_data['adoption_rate'] * 10  # Size based on adoption rate
            )
        
        ax.set_xlabel(factor_names.get(x_factor, x_factor), fontsize=12)
        ax.set_ylabel(factor_names.get(y_factor, y_factor), fontsize=12)
        ax.set_title(f"Relationship between {factor_names.get(x_factor, x_factor)} and {factor_names.get(y_factor, y_factor)}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Region")
        
        st.pyplot(fig)
        
        # Key insights
        st.markdown("<h3 class='section-header'>Key Insights</h3>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="insight-box">
            <p><span class="highlight">Economic factors impact adoption:</span> Regions with higher median income tend to have higher EV adoption rates, suggesting affordability is a key factor.</p>
            <p><span class="highlight">Gas-to-electricity price ratio matters:</span> Areas with a higher ratio between gas and electricity prices show accelerated EV adoption as cost savings increase.</p>
            <p><span class="highlight">Incentives boost adoption:</span> State incentives show a strong positive correlation with adoption rates, highlighting the effectiveness of policy support.</p>
            <p><span class="highlight">Environmental impact varies:</span> Regions with lower electricity carbon intensity see greater environmental benefits from EV adoption.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Tab 4: Predictions & Insights
    with tab4:
        st.markdown("<h2 class='sub-header'>Predictions & Insights</h2>", unsafe_allow_html=True)
        
        if model is not None:
            # Model performance
            st.markdown("<h3 class='section-header'>Model Performance</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("dashboard/assets/actual_vs_predicted.png")
            
            with col2:
                st.image("dashboard/assets/feature_importance.png")
            
            # Projections
            st.markdown("<h3 class='section-header'>Future Projections</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("dashboard/assets/adoption_rate_projections.png")
            
            with col2:
                st.image("dashboard/assets/ev_sales_projections.png")
            
            # Interactive prediction
            st.markdown("<h3 class='section-header'>Interactive Prediction Tool</h3>", unsafe_allow_html=True)
            
            st.info("""
                This tool lets you simulate different scenarios by adjusting key factors and seeing their projected impact on EV adoption rates.
                Adjust the sliders to create your scenario and see the predicted adoption rate.
            """)
            
            # Get a reference region to modify
            reference_region = st.selectbox(
                "Select Reference Region",
                options=regions
            )
            
            # Get the latest data for the reference region
            reference_data = filtered_data[
                (filtered_data['region'] == reference_region) & 
                (filtered_data['year'] == max_year)
            ].copy()
            
            if not reference_data.empty:
                # Create sliders for adjustable factors
                col1, col2 = st.columns(2)
                
                with col1:
                    charging_pct = st.slider(
                        "Charging Station Growth (%)",
                        min_value=-50,
                        max_value=200,
                        value=20,
                        step=10
                    )
                    
                    income_pct = st.slider(
                        "Median Income Growth (%)",
                        min_value=-20,
                        max_value=50,
                        value=5,
                        step=5
                    )
                    
                    elec_price_pct = st.slider(
                        "Electricity Price Change (%)",
                        min_value=-30,
                        max_value=50,
                        value=0,
                        step=5
                    )
                
                with col2:
                    gas_price_pct = st.slider(
                        "Gas Price Change (%)",
                        min_value=-30,
                        max_value=100,
                        value=10,
                        step=5
                    )
                    
                    incentive_change = st.slider(
                        "Incentive Amount Change ($)",
                        min_value=-int(reference_data['incentive_amount'].values[0]),
                        max_value=10000,
                        value=0,
                        step=500
                    )
                    
                    carbon_pct = st.slider(
                        "Carbon Intensity Change (%)",
                        min_value=-50,
                        max_value=20,
                        value=-10,
                        step=5
                    )
                
                # Apply changes to the reference data
                scenario_data = reference_data.copy()
                scenario_data['charging_stations'] *= (1 + charging_pct/100)
                scenario_data['median_income'] *= (1 + income_pct/100)
                scenario_data['electricity_price'] *= (1 + elec_price_pct/100)
                scenario_data['avg_gas_price'] *= (1 + gas_price_pct/100)
                scenario_data['incentive_amount'] += incentive_change
                scenario_data['carbon_intensity'] *= (1 + carbon_pct/100)
                
                # Update derived features
                scenario_data['gas_electricity_ratio'] = scenario_data['avg_gas_price'] / scenario_data['electricity_price']
                
                # Make prediction
                try:
                    # Prepare data for prediction (similar to model training)
                    X_pred = scenario_data.drop(columns=[
                        'region', 'year', 'ev_sales', 'total_car_sales', 
                        'previous_ev_sales', 'adoption_rate', 'yoy_growth'
                    ])
                    
                    # Predict adoption rate
                    predicted_rate = model.predict(X_pred)[0]
                    current_rate = reference_data['adoption_rate'].values[0]
                    
                    # Display the prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Current Adoption Rate",
                            value=f"{current_rate:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Predicted Adoption Rate",
                            value=f"{predicted_rate:.2f}%",
                            delta=f"{predicted_rate - current_rate:.2f}%"
                        )
                    
                    with col3:
                        change_pct = (predicted_rate / current_rate - 1) * 100
                        st.metric(
                            label="Percentage Change",
                            value=f"{change_pct:.1f}%"
                        )
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            
            # Policy recommendations
            st.markdown("<h3 class='section-header'>Policy Recommendations</h3>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class="insight-box">
                <p><span class="highlight">Expand charging infrastructure:</span> Increase public investment in charging stations, with focus on underserved areas and fast chargers along highways.</p>
                <p><span class="highlight">Optimize incentive programs:</span> Target financial incentives to middle-income households where adoption barriers are highest.</p>
                <p><span class="highlight">Grid decarbonization:</span> Coordinate EV adoption with renewable energy expansion to maximize environmental benefits.</p>
                <p><span class="highlight">Electricity pricing policies:</span> Implement time-of-use rates to encourage off-peak charging and reduce grid impacts.</p>
                <p><span class="highlight">Education campaigns:</span> Address common EV misconceptions, especially regarding range anxiety and charging availability.</p>
                </div>
            """, unsafe_allow_html=True)
        
        else:
            st.warning("Model not loaded. Please run the analysis pipeline first to train the model.")
    
    # Footer
    st.markdown("""
        <div class="footer">
        EV Adoption Analysis Dashboard | Created using Streamlit | Data updated as of 2022
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 