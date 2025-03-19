"""
Visualization Module for EV Adoption Analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import logging
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger("ev_adoption_analysis")

class Visualizer:
    """
    Class for creating visualizations of EV adoption analysis.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations.
                                       If None, uses the default visualizations directory.
        """
        if output_dir is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Get the parent directory (src)
            src_dir = os.path.dirname(current_dir)
            # Get the parent directory (project root)
            project_dir = os.path.dirname(src_dir)
            # Set the output directory
            self.output_dir = os.path.join(project_dir, "dashboard", "assets")
        else:
            self.output_dir = output_dir
            
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set a custom color palette
        self.colors = ['#1e88e5', '#ff5722', '#43a047', '#ffc107', '#5e35b1', '#e53935']
        self.color_palette = sns.color_palette(self.colors)
        
        # Modern plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        logger.debug(f"Visualizer initialized with output directory: {self.output_dir}")
    
    def create_adoption_trend_charts(self, df):
        """
        Create charts showing EV adoption trends over time.
        
        Args:
            df (pandas.DataFrame): Processed data with adoption metrics.
            
        Returns:
            list: List of paths to saved visualization files.
        """
        logger.info("Creating adoption trend charts")
        
        saved_files = []
        
        # Ensure the data is sorted by year
        df = df.sort_values(['region', 'year'])
        
        # 1. EV Adoption Rate by Region Over Time
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df,
            x='year',
            y='adoption_rate',
            hue='region',
            palette=self.colors,
            marker='o',
            linewidth=2.5
        )
        plt.title('EV Adoption Rate by Region (2018-2022)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Adoption Rate (%)', fontsize=14)
        plt.legend(title='Region', fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis to show percentages
        plt.gca().set_yticklabels([f'{x:.1f}%' for x in plt.gca().get_yticks()])
        
        # Annotate the last point of each line with the value
        for region in df['region'].unique():
            region_data = df[df['region'] == region].sort_values('year')
            last_point = region_data.iloc[-1]
            plt.annotate(
                f"{last_point['adoption_rate']:.1f}%",
                (last_point['year'], last_point['adoption_rate']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        # Save the chart
        file_path = os.path.join(self.output_dir, "adoption_rate_trend.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(file_path)
        logger.debug(f"Saved adoption rate trend chart to {file_path}")
        
        # 2. YoY Growth Rate by Region
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df,
            x='year',
            y='yoy_growth',
            hue='region',
            palette=self.colors,
            marker='o',
            linewidth=2.5
        )
        plt.title('Year-over-Year EV Sales Growth by Region', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('YoY Growth (%)', fontsize=14)
        plt.legend(title='Region', fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis to show percentages
        plt.gca().set_yticklabels([f'{x:.0f}%' for x in plt.gca().get_yticks()])
        
        # Save the chart
        file_path = os.path.join(self.output_dir, "yoy_growth_trend.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(file_path)
        logger.debug(f"Saved YoY growth trend chart to {file_path}")
        
        # 3. EV Sales Volume by Region
        plt.figure(figsize=(12, 8))
        
        # Create a stacked bar chart
        regions = df['region'].unique()
        years = sorted(df['year'].unique())
        data = {}
        
        for region in regions:
            region_data = df[df['region'] == region].set_index('year')
            data[region] = [region_data.loc[year, 'ev_sales'] if year in region_data.index else 0 for year in years]
        
        bottom = np.zeros(len(years))
        for i, region in enumerate(regions):
            plt.bar(years, data[region], bottom=bottom, width=0.8, label=region, color=self.colors[i % len(self.colors)])
            bottom += np.array(data[region])
        
        plt.title('EV Sales Volume by Region (2018-2022)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of EVs Sold', fontsize=14)
        plt.legend(title='Region', fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add total values on top of each bar
        for i, year in enumerate(years):
            total = sum(data[region][i] for region in regions)
            plt.text(year, total + (total*0.02), f'{total:,}', ha='center', fontsize=12, fontweight='bold')
        
        # Format y-axis with commas for thousands
        plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Save the chart
        file_path = os.path.join(self.output_dir, "ev_sales_by_region.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(file_path)
        logger.debug(f"Saved EV sales volume chart to {file_path}")
        
        # 4. Charging Infrastructure Growth
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=df,
            x='year',
            y='charging_stations',
            hue='region',
            palette=self.colors,
            marker='o',
            linewidth=2.5
        )
        plt.title('Charging Infrastructure Growth by Region (2018-2022)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Charging Stations', fontsize=14)
        plt.legend(title='Region', fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis with commas for thousands
        plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        # Save the chart
        file_path = os.path.join(self.output_dir, "charging_infrastructure_trend.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(file_path)
        logger.debug(f"Saved charging infrastructure trend chart to {file_path}")
        
        # 5. Scatterplot of Charging Stations vs. Adoption Rate
        plt.figure(figsize=(12, 8))
        
        # Get the latest year data
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        # Create a scatterplot
        sns.scatterplot(
            data=latest_data,
            x='charging_stations',
            y='adoption_rate',
            size='total_car_sales', 
            hue='region',
            palette=self.colors,
            sizes=(100, 1000),
            alpha=0.8
        )
        
        plt.title(f'Relationship Between Charging Infrastructure and EV Adoption Rate ({latest_year})', fontsize=16)
        plt.xlabel('Number of Charging Stations', fontsize=14)
        plt.ylabel('EV Adoption Rate (%)', fontsize=14)
        plt.legend(title='Region', fontsize=12, title_fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add labels for each point
        for _, row in latest_data.iterrows():
            plt.annotate(
                row['region'],
                (row['charging_stations'], row['adoption_rate']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        # Format y-axis to show percentages
        plt.gca().set_yticklabels([f'{x:.1f}%' for x in plt.gca().get_yticks()])
        
        # Save the chart
        file_path = os.path.join(self.output_dir, "charging_vs_adoption_scatter.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(file_path)
        logger.debug(f"Saved charging vs. adoption scatterplot to {file_path}")
        
        logger.info(f"Created {len(saved_files)} adoption trend charts")
        return saved_files
    
    def create_correlation_heatmap(self, df):
        """
        Create a heatmap showing correlations between variables.
        
        Args:
            df (pandas.DataFrame): Processed data with adoption metrics.
            
        Returns:
            str: Path to saved heatmap file.
        """
        logger.info("Creating correlation heatmap")
        
        # Select numeric columns for correlation
        numeric_cols = [
            'adoption_rate', 'charging_stations', 'median_income', 
            'electricity_price', 'avg_gas_price', 'carbon_intensity',
            'population_density', 'incentive_amount', 'charging_ev_ratio',
            'yoy_growth', 'gas_electricity_ratio'
        ]
        
        # Only include columns that exist in the dataframe
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('blue_red', ['#1e88e5', 'white', '#e53935'])
        
        # Create the heatmap
        plt.figure(figsize=(14, 12))
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            annot=True,
            fmt='.2f',
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .8}
        )
        
        plt.title('Correlation Matrix of EV Adoption Factors', fontsize=18, pad=20)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Save the heatmap
        file_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved correlation heatmap to {file_path}")
        return file_path
    
    def create_charging_station_map(self, df):
        """
        Create an interactive map showing charging station locations.
        
        Args:
            df (pandas.DataFrame): Charging station data with geographic coordinates.
            
        Returns:
            str: Path to saved map file.
        """
        logger.info("Creating charging station map")
        
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
        
        # Save the map
        file_path = os.path.join(self.output_dir, "charging_stations_map.html")
        m.save(file_path)
        
        logger.debug(f"Saved charging station map to {file_path}")
        return file_path
    
    def create_prediction_plots(self, predictor, df):
        """
        Create plots showing model predictions and future projections.
        
        Args:
            predictor (AdoptionPredictor): Trained predictor model.
            df (pandas.DataFrame): Processed data.
            
        Returns:
            list: List of paths to saved visualization files.
        """
        logger.info("Creating prediction plots")
        
        saved_files = []
        
        # 1. Actual vs. Predicted Adoption Rate
        if hasattr(predictor, 'test_data') and predictor.test_data is not None:
            X_test, y_test = predictor.test_data
            
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, predictor.test_predictions, alpha=0.7, s=80, color='#1e88e5')
            
            # Add the perfect prediction line
            max_val = max(y_test.max(), predictor.test_predictions.max()) * 1.1
            min_val = min(y_test.min(), predictor.test_predictions.min()) * 0.9
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
            
            plt.xlabel('Actual Adoption Rate (%)', fontsize=14)
            plt.ylabel('Predicted Adoption Rate (%)', fontsize=14)
            plt.title('Model Performance: Actual vs. Predicted EV Adoption Rate', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add R² value to the plot
            if 'test_r2' in predictor.metrics:
                r2 = predictor.metrics['test_r2']
                plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
            
            # Save the plot
            file_path = os.path.join(self.output_dir, "actual_vs_predicted.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(file_path)
            logger.debug(f"Saved actual vs. predicted plot to {file_path}")
        
        # 2. Feature Importance Plot
        if predictor.feature_importance is not None:
            # Sort features by importance
            sorted_features = sorted(predictor.feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features)
            
            # Select top 15 features at most
            n_features = min(15, len(features))
            features = features[:n_features]
            importance = importance[:n_features]
            
            plt.figure(figsize=(12, 10))
            sns.barplot(x=list(importance), y=list(features), palette='viridis')
            
            plt.title('Feature Importance for EV Adoption Prediction', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7, axis='x')
            
            # Save the plot
            file_path = os.path.join(self.output_dir, "feature_importance.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(file_path)
            logger.debug(f"Saved feature importance plot to {file_path}")
        
        # 3. Future Projections
        try:
            # Predict 5 years into the future
            future_df = predictor.predict_future(df, years_ahead=5)
            
            # Group by region and year
            future_summary = future_df.groupby(['region', 'year'])[['adoption_rate', 'predicted_adoption_rate', 'predicted_ev_sales']].mean().reset_index()
            
            # Plot projected adoption rates for each region
            plt.figure(figsize=(12, 8))
            
            # Get the current max year from the data
            current_max_year = df['year'].max()
            
            # Plot historical data with solid lines
            for region in future_summary['region'].unique():
                region_data = future_summary[future_summary['region'] == region]
                historical = region_data[region_data['year'] <= current_max_year]
                future = region_data[region_data['year'] > current_max_year]
                
                # Plot historical data
                if not historical.empty:
                    plt.plot(historical['year'], historical['adoption_rate'], 
                            marker='o', linewidth=2.5, label=f"{region} (Historical)")
                
                # Plot projected data with dashed lines and different color
                if not future.empty:
                    plt.plot(future['year'], future['predicted_adoption_rate'], 
                            marker='x', linestyle='--', linewidth=2, label=f"{region} (Projected)")
            
            plt.title('EV Adoption Rate Projections by Region', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Adoption Rate (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a vertical line at the transition from historical to projected data
            plt.axvline(x=current_max_year, color='gray', linestyle='-', alpha=0.5)
            plt.text(current_max_year + 0.1, plt.ylim()[1]*0.98, 'Projections', 
                    fontsize=12, color='gray', ha='left', va='top')
            
            # Format y-axis to show percentages
            plt.gca().set_yticklabels([f'{x:.1f}%' for x in plt.gca().get_yticks()])
            
            # Adjust legend to be more compact
            plt.legend(title='Region', fontsize=10, title_fontsize=12, ncol=2)
            
            # Save the plot
            file_path = os.path.join(self.output_dir, "adoption_rate_projections.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(file_path)
            logger.debug(f"Saved adoption rate projections plot to {file_path}")
            
            # 4. Projected EV Sales
            plt.figure(figsize=(12, 8))
            
            # Stacked bar chart for historical data
            years = sorted(df['year'].unique())
            historical_years = [year for year in years if year <= current_max_year]
            projected_years = [year for year in sorted(future_summary['year'].unique()) if year > current_max_year]
            all_years = historical_years + projected_years
            
            # Get historical EV sales by region and year
            historical_data = {}
            for region in df['region'].unique():
                region_df = df[df['region'] == region].set_index('year')
                historical_data[region] = [region_df.loc[year, 'ev_sales'] if year in region_df.index else 0 
                                        for year in historical_years]
            
            # Get projected EV sales by region and year
            projected_data = {}
            for region in future_summary['region'].unique():
                region_future = future_summary[(future_summary['region'] == region) & 
                                            (future_summary['year'] > current_max_year)].set_index('year')
                projected_data[region] = [region_future.loc[year, 'predicted_ev_sales'] if year in region_future.index else 0 
                                        for year in projected_years]
            
            # Combine historical and projected data
            combined_data = {}
            for region in df['region'].unique():
                if region in historical_data and region in projected_data:
                    combined_data[region] = historical_data[region] + projected_data[region]
            
            # Plot the stacked bar chart
            bottom = np.zeros(len(all_years))
            regions = list(combined_data.keys())
            for i, region in enumerate(regions):
                plt.bar(all_years[:len(historical_years)], combined_data[region][:len(historical_years)], 
                        bottom=bottom[:len(historical_years)], width=0.8, 
                        label=f"{region} (Historical)", color=self.colors[i % len(self.colors)])
                
                plt.bar(all_years[len(historical_years):], combined_data[region][len(historical_years):], 
                        bottom=bottom[len(historical_years):], width=0.8, 
                        label=f"{region} (Projected)", color=self.colors[i % len(self.colors)], alpha=0.6, hatch='/')
                
                bottom += np.array(combined_data[region])
            
            plt.title('EV Sales Projection by Region', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of EVs Sold', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add a vertical line at the transition from historical to projected data
            plt.axvline(x=current_max_year + 0.5, color='gray', linestyle='-', alpha=0.5)
            plt.text(current_max_year + 0.6, plt.ylim()[1]*0.98, 'Projections', 
                    fontsize=12, color='gray', ha='left', va='top')
            
            # Format y-axis with commas for thousands
            plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
            
            # Adjust legend to be more compact
            plt.legend(title='Region', fontsize=10, title_fontsize=12, ncol=2)
            
            # Save the plot
            file_path = os.path.join(self.output_dir, "ev_sales_projections.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(file_path)
            logger.debug(f"Saved EV sales projections plot to {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating future projections: {str(e)}")
        
        logger.info(f"Created {len(saved_files)} prediction plots")
        return saved_files 