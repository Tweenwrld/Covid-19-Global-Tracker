import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback

# Page configuration
st.set_page_config(
    page_title="COVID-19 Global Tracker",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0e7cc0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chart-header {
        font-size: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🌍 COVID-19 Global Tracker</h1>", unsafe_allow_html=True)
st.markdown("Track cases, deaths, and vaccinations worldwide using data from Our World in Data")

# Load data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    """
    Load COVID-19 data from Our World in Data and perform initial preprocessing.
    Returns a pandas DataFrame with COVID-19 data.
    """
    try:
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'])
        
        # Data preprocessing
        for col in ['new_cases', 'new_deaths', 'total_cases', 'total_deaths']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Calculate rolling averages
        df['cases_7day_avg'] = df.groupby('location')['new_cases'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())
        df['deaths_7day_avg'] = df.groupby('location')['new_deaths'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean())

        # Calculate per capita metrics if population data is available
        if 'population' in df.columns:
            df['cases_per_million'] = df['total_cases'] * 1000000 / df['population']
            df['deaths_per_million'] = df['total_deaths'] * 1000000 / df['population']
        
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        # Return empty DataFrame with expected columns to prevent downstream errors
        return pd.DataFrame(columns=['date', 'location', 'iso_code', 'new_cases', 'new_deaths', 
                                    'total_cases', 'total_deaths', 'population'])

try:
    with st.spinner('Loading COVID-19 data from Our World in Data...'):
        df = load_data()
        
        # Check if data was loaded successfully
        if df.empty:
            st.error("No data was loaded. Please check your internet connection and try again.")
            st.stop()
            
        # Display data info
        st.sidebar.info(f"Data ranges from {df['date'].min().date()} to {df['date'].max().date()}")
        st.sidebar.info(f"Number of countries/regions: {df['location'].nunique()}")
        
        # Get list of countries, excluding continents and world
        continents = ['World', 'Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
        countries = sorted([c for c in df['location'].unique() if c not in continents])
        
        # Sidebar filters
        st.sidebar.header("📊 Data Filters")
        
        # Country selection (multiselect for comparison)
        default_countries = ["United States", "United Kingdom", "India"]
        # Make sure default countries exist in the data
        default_countries = [c for c in default_countries if c in countries]
        if not default_countries and countries:
            default_countries = [countries[0]]
            
        selected_countries = st.sidebar.multiselect(
            "Select Countries", 
            countries,
            default=default_countries
        )
        
        # Date range
        default_end_date = df['date'].max()
        default_start_date = default_end_date - timedelta(days=180)  # Default to last 6 months
        
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=default_start_date.date(),
            min_value=df['date'].min().date(),
            max_value=default_end_date.date()
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=default_end_date.date(),
            min_value=start_date,
            max_value=default_end_date.date()
        )
        
        # Metric selection
        metric_options = {
            "New Cases": "new_cases",
            "New Deaths": "new_deaths",
            "Total Cases": "total_cases",
            "Total Deaths": "total_deaths",
            "New Cases (7-day avg)": "cases_7day_avg",
            "New Deaths (7-day avg)": "deaths_7day_avg"
        }
        
        # Add vaccination metrics if available
        if 'people_vaccinated_per_hundred' in df.columns:
            metric_options.update({
                "Vaccination Rate (%)": "people_vaccinated_per_hundred",
                "Fully Vaccinated (%)": "people_fully_vaccinated_per_hundred"
            })
        
        selected_metric = st.sidebar.selectbox("Select Metric", list(metric_options.keys()))
        metric_column = metric_options[selected_metric]
        
        # Normalize by population option
        per_capita = st.sidebar.checkbox("Show per million people", value=False)
        
        # Filter data based on selections
        if not selected_countries:
            st.warning("Please select at least one country.")
            st.stop()
        
        filtered_df = df[
            (df['location'].isin(selected_countries)) &
            (df['date'] >= pd.Timestamp(start_date)) &
            (df['date'] <= pd.Timestamp(end_date))
        ]
        
        # Check if we have data after filtering
        if filtered_df.empty:
            st.warning("No data available for the selected countries and date range.")
            st.stop()
        
        # Calculate per capita values if requested
        if per_capita and 'population' in df.columns:
            display_metric = f"{metric_column}_per_million"
            metric_title = f"{selected_metric} per Million People"
            
            # Create per million columns for each country
            for country in selected_countries:
                country_data = filtered_df[filtered_df['location'] == country]
                # Get population safely
                population = country_data['population'].iloc[0] if (len(country_data) > 0 and 
                                                                  'population' in country_data.columns and 
                                                                  not country_data['population'].isna().all()) else np.nan
                
                if not np.isnan(population) and population > 0:
                    # Create per million column for the selected metric
                    filtered_df.loc[filtered_df['location'] == country, display_metric] = \
                        filtered_df.loc[filtered_df['location'] == country, metric_column] * 1000000 / population
        else:
            display_metric = metric_column
            metric_title = selected_metric
            
        # If per capita was selected but we don't have the column, alert the user
        if per_capita and display_metric not in filtered_df.columns:
            st.warning("Population data is not available for per million calculation. Showing absolute values instead.")
            display_metric = metric_column
            metric_title = selected_metric
        
        # Main dashboard area
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🗺️ Map View", "📊 Country Comparison"])
        
        with tab1:
            st.markdown(f"<h3 class='chart-header'>{metric_title} Over Time</h3>", unsafe_allow_html=True)
            
            # Line chart for trends
            try:
                fig = px.line(
                    filtered_df, 
                    x='date', 
                    y=display_metric, 
                    color='location',
                    title=f"{metric_title} Trends",
                    labels={'date': 'Date', display_metric: metric_title, 'location': 'Country'}
                )
                
                fig.update_layout(
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating trends chart: {e}")
            
            # Summary statistics
            st.markdown(f"<h3 class='chart-header'>Key Statistics</h3>", unsafe_allow_html=True)
            
            # Create columns for country metrics
            cols = st.columns(len(selected_countries))
            
            for i, country in enumerate(selected_countries):
                country_data = filtered_df[filtered_df['location'] == country].sort_values('date')
                
                if not country_data.empty:
                    latest = country_data.iloc[-1]
                    
                    with cols[i]:
                        st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                        st.subheader(country)
                        
                        if 'total_cases' in df.columns:
                            total_cases = latest.get('total_cases', 0)
                            if not pd.isna(total_cases):
                                st.metric("Total Cases", f"{int(total_cases):,}")
                        
                        if 'total_deaths' in df.columns:
                            total_deaths = latest.get('total_deaths', 0)
                            if not pd.isna(total_deaths):
                                st.metric("Total Deaths", f"{int(total_deaths):,}")
                        
                        if 'people_vaccinated_per_hundred' in df.columns:
                            vax_rate = latest.get('people_vaccinated_per_hundred', None)
                            if vax_rate is not None and not pd.isna(vax_rate):
                                st.metric("Vaccination Rate", f"{vax_rate:.1f}%")
                        
                        # Calculate case fatality rate
                        if ('total_cases' in df.columns and 'total_deaths' in df.columns and 
                            latest.get('total_cases', 0) > 0 and not pd.isna(latest.get('total_cases', 0))):
                            cfr = (latest.get('total_deaths', 0) / latest.get('total_cases', 0)) * 100
                            st.metric("Case Fatality Rate", f"{cfr:.2f}%")
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                    
        with tab2:
            st.markdown("<h3 class='chart-header'>Global Map View</h3>", unsafe_allow_html=True)
            
            # Get latest date data for all countries for map
            latest_date = df['date'].max()
            map_data = df[df['date'] == latest_date].copy()
            
            # Choose map metric
            map_metric_options = {
                "Total Cases": "total_cases",
                "Total Deaths": "total_deaths",
                "Total Cases per Million": "cases_per_million",
                "Total Deaths per Million": "deaths_per_million"
            }
            
            # Add vaccination metrics if available
            if 'people_vaccinated_per_hundred' in df.columns:
                map_metric_options.update({
                    "Vaccination Rate (%)": "people_vaccinated_per_hundred",
                    "Fully Vaccinated (%)": "people_fully_vaccinated_per_hundred"
                })
                
            map_metric = st.selectbox("Select Map Metric", list(map_metric_options.keys()))
            map_column = map_metric_options[map_metric]
            
            # Check if this metric exists
            if map_column not in map_data.columns:
                st.warning(f"Selected metric '{map_metric}' is not available in the dataset.")
            else:
                # Create choropleth map
                try:
                    # Filter out rows with missing data for this metric or iso_code
                    valid_map_data = map_data.dropna(subset=['iso_code', map_column])
                    
                    if not valid_map_data.empty:
                        fig_map = px.choropleth(
                            valid_map_data,
                            locations="iso_code",
                            color=map_column,
                            hover_name="location",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title=f"Global COVID-19 {map_metric}"
                        )
                        
                        fig_map.update_layout(
                            height=600,
                            coloraxis_colorbar=dict(title=map_metric),
                            geo=dict(showframe=False, showcoastlines=True)
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Show top 10 countries for this metric
                        st.markdown(f"<h3 class='chart-header'>Top 10 Countries by {map_metric}</h3>", 
                                   unsafe_allow_html=True)
                        
                        top10 = valid_map_data.sort_values(map_column, ascending=False).head(10)
                        top10_fig = px.bar(
                            top10,
                            x='location',
                            y=map_column,
                            color='location',
                            labels={map_column: map_metric, 'location': 'Country'}
                        )
                        
                        top10_fig.update_layout(
                            height=400,
                            xaxis={'categoryorder':'total descending'},
                            xaxis_title="Country",
                            yaxis_title=map_metric
                        )
                        
                        st.plotly_chart(top10_fig, use_container_width=True)
                    else:
                        st.warning(f"No valid data available for '{map_metric}' visualization.")
                except Exception as e:
                    st.error(f"Error creating map visualization: {e}")
                    st.error(traceback.format_exc())
        
        with tab3:
            st.markdown("<h3 class='chart-header'>Multi-Country Comparison</h3>", unsafe_allow_html=True)
            
            if len(selected_countries) < 2:
                st.info("Please select at least two countries in the sidebar to compare data.")
            else:
                # Compare metrics
                comparison_options = {
                    "Time Series": "Show trend comparison",
                    "Bar Chart": "Compare latest values",
                    "Case Fatality Rate": "Compare mortality rate"
                }
                
                comparison_type = st.selectbox(
                    "Comparison Type", 
                    list(comparison_options.keys()),
                    format_func=lambda x: f"{x}: {comparison_options[x]}"
                )
                
                if comparison_type == "Time Series":
                    # Already shown in tab1, but we could show a different metric here
                    st.markdown("<h4>Trend Comparison</h4>", unsafe_allow_html=True)
                    
                    # Let user pick a different metric for comparison
                    comp_metric = st.selectbox(
                        "Select Comparison Metric", 
                        list(metric_options.keys()),
                        index=list(metric_options.keys()).index(selected_metric)
                    )
                    comp_column = metric_options[comp_metric]
                    
                    # Handle per capita option
                    if per_capita and 'population' in df.columns:
                        comp_display = f"{comp_column}_per_million"
                        comp_title = f"{comp_metric} per Million People"
                        
                        # Create per million columns if needed
                        for country in selected_countries:
                            country_data = filtered_df[filtered_df['location'] == country]
                            population = country_data['population'].iloc[0] if not country_data.empty else np.nan
                            
                            if not np.isnan(population) and population > 0:
                                filtered_df.loc[filtered_df['location'] == country, comp_display] = \
                                    filtered_df.loc[filtered_df['location'] == country, comp_column] * 1000000 / population
                    else:
                        comp_display = comp_column
                        comp_title = comp_metric
                    
                    # Create the comparison chart
                    try:
                        fig_comp = px.line(
                            filtered_df,
                            x='date',
                            y=comp_display,
                            color='location',
                            title=f"{comp_title} Comparison",
                            labels={'date': 'Date', comp_display: comp_title, 'location': 'Country'}
                        )
                        
                        fig_comp.update_layout(
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating comparison chart: {e}")
                
                elif comparison_type == "Bar Chart":
                    st.markdown("<h4>Current Values Comparison</h4>", unsafe_allow_html=True)
                    
                    # Get latest values for each country
                    latest_values = []
                    
                    for country in selected_countries:
                        country_data = filtered_df[filtered_df['location'] == country].sort_values('date')
                        if not country_data.empty:
                            latest_values.append(country_data.iloc[-1])
                    
                    if latest_values:
                        latest_df = pd.DataFrame(latest_values)
                        
                        # Create bar chart comparison
                        try:
                            # Let user select metrics to compare
                            compare_metrics = st.multiselect(
                                "Select Metrics to Compare", 
                                list(metric_options.keys()),
                                default=[selected_metric]
                            )
                            
                            if compare_metrics:
                                compare_columns = [metric_options[m] for m in compare_metrics]
                                
                                # Create a tall format dataframe for multi-metric comparison
                                plot_data = []
                                for _, row in latest_df.iterrows():
                                    for metric_name, metric_col in zip(compare_metrics, compare_columns):
                                        if metric_col in row:
                                            plot_data.append({
                                                'Country': row['location'],
                                                'Metric': metric_name,
                                                'Value': row[metric_col]
                                            })
                                
                                plot_df = pd.DataFrame(plot_data)
                                
                                if not plot_df.empty:
                                    fig_bar = px.bar(
                                        plot_df,
                                        x='Country',
                                        y='Value',
                                        color='Metric',
                                        barmode='group',
                                        title="Multi-Metric Comparison"
                                    )
                                    
                                    fig_bar.update_layout(
                                        height=500,
                                        xaxis_title="Country",
                                        yaxis_title="Value",
                                        legend_title="Metric"
                                    )
                                    
                                    st.plotly_chart(fig_bar, use_container_width=True)
                                else:
                                    st.warning("No data available for selected metrics.")
                            else:
                                st.warning("Please select at least one metric to compare.")
                        except Exception as e:
                            st.error(f"Error creating bar chart: {e}")
                
                elif comparison_type == "Case Fatality Rate":
                    st.markdown("<h4>Case Fatality Rate Comparison</h4>", unsafe_allow_html=True)
                    
                    try:
                        # Calculate CFR for each country
                        cfr_data = []
                        
                        for country in selected_countries:
                            country_data = filtered_df[filtered_df['location'] == country].sort_values('date')
                            if not country_data.empty:
                                latest = country_data.iloc[-1]
                                
                                if ('total_cases' in latest and 'total_deaths' in latest and 
                                    latest['total_cases'] > 0 and not pd.isna(latest['total_cases'])):
                                    cfr = (latest['total_deaths'] / latest['total_cases']) * 100
                                    cfr_data.append({
                                        'Country': country,
                                        'Case Fatality Rate (%)': cfr
                                    })
                        
                        if cfr_data:
                            cfr_df = pd.DataFrame(cfr_data)
                            
                            # Create CFR comparison chart
                            fig_cfr = px.bar(
                                cfr_df,
                                x='Country',
                                y='Case Fatality Rate (%)',
                                color='Country',
                                title="Case Fatality Rate Comparison"
                            )
                            
                            fig_cfr.update_layout(
                                height=500,
                                xaxis_title="Country",
                                yaxis_title="Case Fatality Rate (%)"
                            )
                            
                            st.plotly_chart(fig_cfr, use_container_width=True)
                            
                            # Display CFR data in table
                            st.markdown("<h4>Case Fatality Rate Data</h4>", unsafe_allow_html=True)
                            st.dataframe(cfr_df.sort_values('Case Fatality Rate (%)', ascending=False))
                        else:
                            st.warning("Could not calculate Case Fatality Rate. Insufficient data.")
                    except Exception as e:
                        st.error(f"Error calculating Case Fatality Rate: {e}")

        # Add footer with data source and refresh info
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Source:** [Our World in Data](https://github.com/owid/covid-19-data)")
        with col2:
            st.markdown(f"**Last Updated:** {df['date'].max().strftime('%Y-%m-%d')}")
            
        # Add download capability
        st.markdown("---")
        st.markdown("### Download Filtered Data")
        
        # Prepare download options
        download_options = st.radio(
            "Select download format:",
            ("CSV", "Excel"),
            horizontal=True
        )
        
        # Add download button
        if download_options == "CSV":
            csv = filtered_df.to_csv(index=False)
            download_filename = f"covid19_data_{'-'.join(selected_countries)}_{start_date}_{end_date}.csv"
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=download_filename,
                mime="text/csv"
            )
        else:
            # Generate Excel file
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name="COVID19_Data")
                
            download_filename = f"covid19_data_{'-'.join(selected_countries)}_{start_date}_{end_date}.xlsx"
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

except Exception as main_error:
    st.error("An unexpected error occurred in the application.")
    st.error(f"Error details: {main_error}")
    st.error(traceback.format_exc())
    st.info("Please refresh the page and try again. If the problem persists, check your internet connection or try again later.")