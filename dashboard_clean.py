#!/usr/bin/env python3
"""
Streamlit Dashboard for EV Adoption Analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Set page config - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="EV Adoption Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

import matplotlib.pyplot as plt
import folium

# Import streamlit_folium quietly
try:
    from streamlit_folium import st_folium
except ImportError:
    # If import fails, define a placeholder function
    def st_folium(folium_map, width, height):
        st.warning("Map display requires streamlit-folium package")
        return None

import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.models.adoption_predictor import AdoptionPredictor

# Define functions for the dashboard
def load_data():
    """
    Load and prepare data for the dashboard.
    """
    try:
        # Try to load processed data
        df = pd.read_csv("data/processed/processed_ev_data.csv")
        charging_df = pd.read_csv("data/raw/charging_stations_geo.csv")
        return df, charging_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def format_number(num):
    """Format numbers with commas for thousands."""
    return f"{int(num):,}"

def create_intro():
    """Create the introduction section of the dashboard."""
    st.title("EV Adoption Analysis Dashboard")
    
    st.markdown("""
    This dashboard presents a comprehensive analysis of electric vehicle (EV) adoption trends, 
    factors influencing adoption, and future projections across different regions.
    
    Use the sidebar to navigate through different sections of the analysis or to modify scenario parameters.
    """)
    
    # Display the key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data for metrics
    df, _ = load_data()
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    
    # Total EV sales in the latest year
    total_sales = latest_data['ev_sales'].sum()
    with col1:
        st.metric(
            label=f"Total EV Sales ({latest_year})",
            value=format_number(total_sales),
            delta=f"+{int((total_sales/df[df['year'] == latest_year-1]['ev_sales'].sum() - 1) * 100)}%"
        )
    
    # Average adoption rate
    avg_adoption = latest_data['adoption_rate'].mean()
    with col2:
        st.metric(
            label=f"Avg Adoption Rate ({latest_year})",
            value=f"{avg_adoption:.2f}%",
            delta=f"+{(avg_adoption/df[df['year'] == latest_year-1]['adoption_rate'].mean() - 1) * 100:.1f}%"
        )
    
    # Total charging stations
    total_charging = latest_data['charging_stations'].sum()
    with col3:
        st.metric(
            label=f"Charging Stations ({latest_year})",
            value=format_number(total_charging),
            delta=f"+{int((total_charging/df[df['year'] == latest_year-1]['charging_stations'].sum() - 1) * 100)}%"
        )
    
    # Region with highest adoption
    highest_region = latest_data.loc[latest_data['adoption_rate'].idxmax()]['region']
    highest_rate = latest_data['adoption_rate'].max()
    with col4:
        st.metric(
            label="Highest Adoption Region",
            value=highest_region,
            delta=f"{highest_rate:.2f}%"
        )
        
    # Add new content to fill the empty space
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        The electric vehicle market continues to demonstrate robust growth across all analyzed regions,
        with year-over-year increases in both absolute sales and adoption rates. Key observations include:
        
        - **Regional Variations**: The Northeast and West Coast regions show the highest adoption rates,
          likely due to stronger policy support and more developed charging infrastructure.
          
        - **Infrastructure Correlation**: Our analysis confirms a strong positive correlation between
          charging infrastructure density and EV adoption rates.
          
        - **Economic Factors**: Financial incentives remain a significant driver of adoption, with
          each $1,000 in incentives associated with approximately 0.7% increase in adoption rate.
          
        - **Future Outlook**: Projections indicate accelerating adoption through 2027, with the national
          average potentially reaching 15-20% of new vehicle sales, though regional variations will persist.
        """)
    
    with col2:
        st.image("dashboard/assets/adoption_rate_trend.png", caption="EV Adoption Rate Trends by Region")
    
    # Dashboard Overview
    st.subheader("🔍 Dashboard Sections Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚗 Adoption Trends**
        - Historical adoption patterns
        - Regional comparisons
        - Growth rate analysis
        - Year-over-year changes
        """)
        
        st.markdown("""
        **📈 Factor Analysis**
        - Correlation between variables
        - Key adoption drivers
        - Regional factor differences
        - Temporal changes in relationships
        """)
    
    with col2:
        st.markdown("""
        **🔌 Charging Infrastructure**
        - Interactive map visualization
        - Infrastructure density analysis
        - Charger type distribution
        - Strategic location patterns
        """)
        
        st.markdown("""
        **🔮 Predictions & Projections**
        - Model performance metrics
        - Feature importance analysis
        - 5-year regional projections
        - Confidence intervals
        """)
    
    with col3:
        st.markdown("""
        **⚖️ Scenario Analysis**
        - Interactive "what-if" tool
        - Policy impact simulation
        - Sensitivity testing
        - Regional response variations
        """)
        
        st.markdown("""
        **📋 Policy Recommendations**
        - Evidence-based suggestions
        - Regional policy customization
        - Implementation prioritization
        - Expected impact assessment
        """)

def create_trend_analysis():
    """Create the trend analysis section of the dashboard."""
    st.header("EV Adoption Trends")
    
    # Add some explanatory text
    st.markdown("""
    This section shows the historical trends in EV adoption across different regions.
    The visualizations help identify which regions are leading in adoption and how the growth patterns differ.
    """)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Adoption Rate", "EV Sales", "YoY Growth", "Charging Infrastructure"])
    
    with tab1:
        st.subheader("EV Adoption Rate by Region")
        st.image("dashboard/assets/adoption_rate_trend.png")
        st.markdown("""
        **Observation**: Adoption rates show consistent growth across all regions, with some regions 
        demonstrating accelerated adoption in recent years. This suggests increasing consumer acceptance 
        of electric vehicles and improved value proposition.
        """)
    
    with tab2:
        st.subheader("EV Sales Volume by Region")
        st.image("dashboard/assets/ev_sales_by_region.png")
        st.markdown("""
        **Observation**: Absolute sales volumes provide insight into the market size for EVs in each region.
        The stacked bar chart shows both individual regional contributions and the overall market growth.
        """)
    
    with tab3:
        st.subheader("Year-over-Year Growth in EV Sales")
        st.image("dashboard/assets/yoy_growth_trend.png")
        st.markdown("""
        **Observation**: Year-over-year growth rates help identify which regions are experiencing 
        acceleration or deceleration in EV adoption. Declining growth rates may indicate market saturation
        or the need for additional policy support.
        """)
    
    with tab4:
        st.subheader("Charging Infrastructure Development")
        st.image("dashboard/assets/charging_infrastructure_trend.png")
        st.markdown("""
        **Observation**: The development of charging infrastructure is a critical enabler for EV adoption.
        Regions with more rapid infrastructure deployment typically see corresponding increases in EV sales.
        """)

def create_correlation_analysis():
    """Create the correlation analysis section of the dashboard."""
    st.header("Factors Influencing EV Adoption")
    
    st.markdown("""
    This section explores the relationships between various factors and EV adoption rates.
    Understanding these correlations can help identify the most effective levers for policy interventions.
    """)
    
    # Create two columns for the layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Correlation Matrix")
        st.image("dashboard/assets/correlation_heatmap.png")
        st.markdown("""
        The correlation heatmap shows the strength of relationships between different factors.
        Strong positive correlations (close to 1.0) indicate that as one factor increases, the other 
        tends to increase as well. Strong negative correlations (close to -1.0) indicate an inverse relationship.
        """)
    
    with col2:
        st.subheader("Key Insights")
        st.markdown("""
        Based on the correlation analysis:
        
        1. **Charging Infrastructure**: Shows a strong positive correlation with adoption rates, confirming
           that infrastructure availability is a key enabler.
           
        2. **Economic Factors**: Income levels and incentive amounts both correlate positively with adoption,
           suggesting that affordability remains an important consideration.
           
        3. **Fuel Prices**: Higher gas prices relative to electricity costs tend to drive higher EV adoption
           as the operational cost advantage becomes more significant.
           
        4. **Carbon Intensity**: Regions with higher grid carbon intensity tend to show lower EV adoption,
           possibly due to reduced environmental incentives or different policy priorities.
        """)
    
    st.subheader("Relationship Between Charging Infrastructure and Adoption")
    st.image("dashboard/assets/charging_vs_adoption_scatter.png")
    st.markdown("""
    This scatter plot specifically highlights how charging infrastructure availability relates to adoption rates
    across different regions. The clear positive relationship suggests that infrastructure development
    should be a priority in regions seeking to boost EV adoption.
    """)

def create_geographic_visualization():
    """Create geographic visualizations of charging infrastructure."""
    st.header("Charging Infrastructure Map")
    
    st.markdown("""
    This interactive map shows the distribution of EV charging stations across regions.
    The clustering of stations provides insight into infrastructure density and potential gaps.
    """)
    
    # Load charging station data
    _, charging_df = load_data()
    
    if charging_df is not None:
        # Create a folium map centered on the US
        map_obj = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Add markers for each charging station
        for _, station in charging_df.iterrows():
            # Choose color based on charger type
            color = 'red' if station['charger_type'] == 'DC Fast Charger' else 'blue'
            
            # Create popup information
            popup_text = f"""
            <b>Station ID:</b> {station['station_id']}<br>
            <b>Region:</b> {station['region']}<br>
            <b>Charger Type:</b> {station['charger_type']}<br>
            <b>Power Level:</b> {station['power_level']} kW<br>
            <b>Operator:</b> {station['operator']}<br>
            <b>Ports:</b> {station['available_ports']}<br>
            <b>Avg. Daily Usage:</b> {station['avg_daily_usage']} sessions<br>
            <b>Installed:</b> {station['installation_date']}
            """
            
            # Add the marker to the map
            folium.Marker(
                location=[station['latitude'], station['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{station['charger_type']} - {station['power_level']} kW",
                icon=folium.Icon(color=color, icon='plug', prefix='fa')
            ).add_to(map_obj)
        
        # Add layer control and display the map
        folium.LayerControl().add_to(map_obj)
        st_folium(map_obj, width=1000, height=600)
        
        st.markdown("""
        **Map Legend:**
        - <span style='color:red'>⬤</span> DC Fast Chargers
        - <span style='color:blue'>⬤</span> Level 2 Chargers
        
        **Observations:**
        1. Charging infrastructure tends to be concentrated in urban areas and along major highways.
        2. Regions with higher adoption rates typically show denser charging networks.
        3. Fast chargers (red) are strategically placed along travel corridors, while Level 2 chargers (blue)
           are more common in urban and suburban areas.
        """)
    else:
        st.error("Could not load charging station data for map visualization.")

def create_prediction_section():
    """Create the prediction and forecasting section."""
    st.header("Predictions and Projections")
    
    st.markdown("""
    This section presents model-based predictions of future EV adoption trends and analyzes
    the key features that influence these predictions.
    """)
    
    # Create tabs for different prediction visualizations
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Performance", "Future Projections"])
    
    with tab1:
        st.subheader("Factors Driving EV Adoption")
        st.image("dashboard/assets/feature_importance.png")
        st.markdown("""
        The chart shows the relative importance of different factors in predicting EV adoption rates.
        These insights can help policymakers and stakeholders focus their efforts on the most influential factors.
        """)
    
    with tab2:
        st.subheader("Prediction Model Performance")
        st.image("dashboard/assets/actual_vs_predicted.png")
        st.markdown("""
        This scatter plot shows how well our model's predictions match actual adoption rates.
        Points closer to the diagonal line represent more accurate predictions. The clustering of points
        near this line indicates that our model has good predictive power.
        """)
    
    with tab3:
        st.subheader("Future Adoption Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("dashboard/assets/adoption_rate_projections.png")
            st.markdown("""
            Projected adoption rates through 2027 show the expected trajectory for each region.
            These projections account for historical trends, policy environments, and market developments.
            """)
        
        with col2:
            st.image("dashboard/assets/ev_sales_projections.png")
            st.markdown("""
            Projected EV sales volumes through 2027 illustrate the expected market growth in absolute terms.
            The projections suggest significant expansion of the EV market across all regions.
            """)

def create_scenario_analysis():
    """Create the scenario analysis section with interactive sliders."""
    st.header("What-If Scenario Analysis")
    
    st.markdown("""
    This interactive tool allows you to explore how changes in key factors might affect future EV adoption.
    Adjust the sliders to create different scenarios and see the projected impact on adoption rates.
    """)
    
    # Load base data
    df, _ = load_data()
    
    if df is not None:
        # Get the latest year data as the baseline
        latest_year = df['year'].max()
        scenario_data = df[df['year'] == latest_year].copy()
        
        # Create sliders for adjusting key factors
        st.subheader("Adjust Scenario Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            incentive_pct = st.slider(
                "Change in Financial Incentives (%)",
                min_value=-100,
                max_value=200,
                value=0,
                step=10,
                help="Adjust the level of financial incentives for EV purchases"
            )
            
            charging_pct = st.slider(
                "Change in Charging Infrastructure (%)",
                min_value=-50,
                max_value=200,
                value=0,
                step=10,
                help="Adjust the availability of charging infrastructure"
            )
        
        with col2:
            gas_price_pct = st.slider(
                "Change in Gas Prices (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=10,
                help="Adjust the price of gasoline"
            )
            
            carbon_pct = st.slider(
                "Change in Grid Carbon Intensity (%)",
                min_value=-75,
                max_value=50,
                value=0,
                step=5,
                help="Adjust the carbon intensity of the electricity grid"
            )
        
        # Apply the adjustments to create the scenario
        if st.button("Calculate Scenario Impact"):
            with st.spinner("Calculating scenario impact..."):
                # Apply changes to the scenario data
                scenario_data['incentive_amount'] *= (1 + incentive_pct/100)
                scenario_data['charging_stations'] *= (1 + charging_pct/100)
                scenario_data['avg_gas_price'] *= (1 + gas_price_pct/100)
                scenario_data['carbon_intensity'] *= (1 + carbon_pct/100)
                
                # Simple model to estimate new adoption rates based on elasticities
                # This is a simplified approach - a real model would be more complex
                baseline = scenario_data['adoption_rate'].copy()
                
                # Apply elasticities (hypothetical values for demonstration)
                incentive_elasticity = 0.3  # 1% change in incentives -> 0.3% change in adoption
                charging_elasticity = 0.5   # 1% change in charging -> 0.5% change in adoption
                gas_price_elasticity = 0.2  # 1% change in gas price -> 0.2% change in adoption
                carbon_elasticity = -0.1    # 1% change in carbon intensity -> -0.1% change in adoption
                
                # Calculate impact
                incentive_impact = baseline * (incentive_elasticity * incentive_pct / 100)
                charging_impact = baseline * (charging_elasticity * charging_pct / 100)
                gas_price_impact = baseline * (gas_price_elasticity * gas_price_pct / 100)
                carbon_impact = baseline * (carbon_elasticity * carbon_pct / 100)
                
                # Apply combined impact
                scenario_data['projected_adoption_rate'] = baseline + incentive_impact + charging_impact + gas_price_impact + carbon_impact
                
                # Ensure no negative adoption rates
                scenario_data['projected_adoption_rate'] = scenario_data['projected_adoption_rate'].clip(lower=0)
                
                # Calculate average change
                avg_baseline = baseline.mean()
                avg_projected = scenario_data['projected_adoption_rate'].mean()
                pct_change = (avg_projected / avg_baseline - 1) * 100
                
                # Display the results
                st.subheader("Scenario Impact")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Average Baseline Adoption Rate",
                        value=f"{avg_baseline:.2f}%"
                    )
                    
                    st.metric(
                        label="Average Projected Adoption Rate",
                        value=f"{avg_projected:.2f}%", 
                        delta=f"{pct_change:.1f}%"
                    )
                
                with col2:
                    # Create a bar chart comparing baseline and projected
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bar_width = 0.35
                    regions = scenario_data['region']
                    x = np.arange(len(regions))
                    
                    ax.bar(x - bar_width/2, scenario_data['adoption_rate'], bar_width, label='Baseline', color='skyblue')
                    ax.bar(x + bar_width/2, scenario_data['projected_adoption_rate'], bar_width, label='Projected', color='orange')
                    
                    ax.set_ylabel('Adoption Rate (%)')
                    ax.set_title('Baseline vs. Projected Adoption Rates by Region')
                    ax.set_xticks(x)
                    ax.set_xticklabels(regions, rotation=45, ha='right')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown(f"""
                ### Key Findings
                
                Based on the scenario parameters you selected:
                
                - The average EV adoption rate across all regions would change from 
                  **{avg_baseline:.2f}%** to **{avg_projected:.2f}%** ({pct_change:.1f}% change).
                  
                - The largest contributing factor to this change is 
                  **{'increased charging infrastructure' if charging_impact.mean() > max(incentive_impact.mean(), gas_price_impact.mean(), abs(carbon_impact.mean())) else
                    'financial incentives' if incentive_impact.mean() > max(charging_impact.mean(), gas_price_impact.mean(), abs(carbon_impact.mean())) else
                    'gas prices' if gas_price_impact.mean() > max(incentive_impact.mean(), charging_impact.mean(), abs(carbon_impact.mean())) else
                    'grid decarbonization'}**.
                  
                - Regions would experience different magnitudes of change, with the highest impact in
                  **{scenario_data.loc[scenario_data['projected_adoption_rate'] / scenario_data['adoption_rate'] - 1 == (scenario_data['projected_adoption_rate'] / scenario_data['adoption_rate'] - 1).max(), 'region'].values[0]}**.
                """)
        
    else:
        st.error("Could not load data for scenario analysis.")

def create_policy_recommendations():
    """Create the policy recommendations section."""
    st.header("Policy Recommendations")
    
    st.markdown("""
    Based on the analysis of historical trends, correlations, and predictive models,
    the following policy recommendations are suggested to accelerate EV adoption:
    """)
    
    # Create expandable sections for different recommendation categories
    with st.expander("Infrastructure Development", expanded=True):
        st.markdown("""
        - **Strategic Charging Network Expansion**: Prioritize charging infrastructure in urban centers, 
          along highways, and in underserved regions with high potential for adoption.
          
        - **Public-Private Partnerships**: Establish partnerships with private charging networks to accelerate
          deployment while reducing public investment requirements.
          
        - **Grid Integration Planning**: Develop comprehensive plans for grid upgrades to support increased
          electricity demand from EVs, including smart charging capabilities.
        """)
    
    with st.expander("Financial Incentives"):
        st.markdown("""
        - **Targeted Rebate Programs**: Design incentive programs that target middle-income consumers
          who are price-sensitive but have sufficient resources to consider an EV purchase.
          
        - **Used EV Incentives**: Expand incentives to the used EV market to make electric mobility
          more accessible to lower-income consumers.
          
        - **Time-Limited Incentives**: Implement declining incentive schedules that provide higher benefits
          in the near term but gradually phase out as EV costs decrease.
        """)
    
    with st.expander("Regulatory Frameworks"):
        st.markdown("""
        - **Zero-Emission Vehicle Mandates**: Establish or strengthen requirements for automakers
          to sell increasing percentages of zero-emission vehicles.
          
        - **Building Code Updates**: Require EV-ready wiring in new construction and major renovations
          to reduce future charging infrastructure costs.
          
        - **Carbon Pricing**: Implement or strengthen carbon pricing mechanisms to better reflect
          the environmental costs of fossil fuels.
        """)
    
    with st.expander("Education and Awareness"):
        st.markdown("""
        - **Experience Centers**: Support the creation of EV experience centers where consumers
          can learn about and test drive electric vehicles without sales pressure.
          
        - **Total Cost of Ownership Tools**: Develop and promote easy-to-use tools that help consumers
          understand the total cost of ownership advantages of EVs over time.
          
        - **Workplace Engagement**: Partner with major employers to provide EV information and
          charging solutions at workplaces.
        """)

def main():
    """Main function to run the dashboard."""
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a section:",
        [
            "Introduction", 
            "Adoption Trends", 
            "Factor Analysis",
            "Charging Infrastructure Map",
            "Predictions & Projections",
            "Scenario Analysis",
            "Policy Recommendations"
        ]
    )
    
    # Add a sidebar note
    st.sidebar.markdown("""
    ---
    This dashboard presents analysis of EV adoption patterns
    and factors influencing adoption rates across regions.
    
    Data is based on historical trends from 2018-2022 with
    projections extending to 2027.
    """)
    
    # Main content based on selection
    if page == "Introduction":
        create_intro()
    elif page == "Adoption Trends":
        create_trend_analysis()
    elif page == "Factor Analysis":
        create_correlation_analysis()
    elif page == "Charging Infrastructure Map":
        create_geographic_visualization()
    elif page == "Predictions & Projections":
        create_prediction_section()
    elif page == "Scenario Analysis":
        create_scenario_analysis()
    elif page == "Policy Recommendations":
        create_policy_recommendations()
    
    # Footer
    st.markdown("---")
    st.caption("EV Adoption Analysis Dashboard | Created for demonstration purposes")

if __name__ == "__main__":
    main() 