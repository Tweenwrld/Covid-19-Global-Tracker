import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import traceback
import time
import math
import io
import base64

# Page configuration
st.set_page_config(
    page_title="COVID-19 Advanced Analytics Dashboard",
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
    .insight-box {
        background-color: #f0f7ff;
        border-left: 5px solid #0e7cc0;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff8f0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0e7cc0;
    }
    .metric-title {
        font-size: 1rem;
        color: #555;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e7cc0 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🌍 COVID-19 Advanced Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Comprehensive analysis of global pandemic data with advanced statistical insights and forecasting")

# Load data
@st.cache_data(ttl=3600)
def load_data():
    """
    Load COVID-19 data from Our World in Data and perform comprehensive preprocessing.
    Returns a pandas DataFrame with COVID-19 data.
    """
    try:
        url = "data/owid-covid-data.csv"
        df = pd.read_csv(url, parse_dates=['date'])
        
        # Comprehensive data preprocessing
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(0)
        
        # Calculate rolling averages and rates of change
        for col in ['new_cases', 'new_deaths']:
            if col in df.columns:
                df[f'{col}_7day_avg'] = df.groupby('location')[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean())
                df[f'{col}_14day_avg'] = df.groupby('location')[col].transform(
                    lambda x: x.rolling(window=14, min_periods=1).mean())
                df[f'{col}_growth_rate'] = df.groupby('location')[f'{col}_7day_avg'].transform(
                    lambda x: x.pct_change(periods=7).fillna(0) * 100)
                df[f'{col}_acceleration'] = df.groupby('location')[f'{col}_growth_rate'].transform(
                    lambda x: x.diff().fillna(0))

        # Calculate per capita metrics
        if 'population' in df.columns:
            for col in ['total_cases', 'total_deaths', 'new_cases', 'new_deaths']:
                if col in df.columns:
                    df[f'{col}_per_million'] = df[col] * 1000000 / df['population']
            if 'total_tests' in df.columns and 'total_cases' in df.columns:
                df['tests_per_case'] = df['total_tests'] / df['total_cases'].replace(0, np.nan)
            if 'total_cases' in df.columns and 'total_deaths' in df.columns:
                df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases'].replace(0, np.nan)) * 100
            if 'reproduction_rate' not in df.columns and 'new_cases_7day_avg' in df.columns:
                df['approx_reproduction_rate'] = df.groupby('location')['new_cases_7day_avg'].transform(
                    lambda x: x / x.shift(7).replace(0, np.nan))
        
        # Date-related features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Wave indicators
        df['new_cases_trend'] = df.groupby('location')['new_cases_7day_avg'].transform(
            lambda x: np.sign(x.diff()))
        df['wave_change'] = df.groupby('location')['new_cases_trend'].transform(
            lambda x: x.diff().fillna(0) != 0).astype(int)
        
        # Continent mapping
        if 'continent' not in df.columns and 'iso_code' in df.columns:
            continent_map = df[df['iso_code'].str.len() == 2].drop_duplicates('location').set_index('location')['continent'].to_dict()
            df['continent'] = df['location'].map(continent_map)
        
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame(columns=['date', 'location', 'iso_code', 'new_cases', 'new_deaths', 
                                    'total_cases', 'total_deaths', 'population'])

def get_significant_date_events():
    """Return dictionary of significant COVID-19 timeline events"""
    return {
        '2020-01-23': 'Wuhan lockdown begins',
        '2020-03-11': 'WHO declares pandemic',
        '2020-12-11': 'First vaccine approved (US)',
        '2021-01-20': 'Biden administration begins',
        '2021-06-01': 'Delta variant emerges globally',
        '2021-11-26': 'Omicron variant identified',
        '2022-01-15': 'Omicron wave peak in many countries',
        '2022-05-01': 'Many countries end restrictions',
        '2023-05-05': 'WHO ends global health emergency',
        '2023-12-01': 'JN.1 variant emerges'
    }

def add_timeline_events(fig, start_date, end_date):
    """Add vertical lines for significant events to a plotly figure"""
    events = get_significant_date_events()
    filtered_events = {k: v for k, v in events.items() 
                      if pd.Timestamp(k) >= pd.Timestamp(start_date) and 
                         pd.Timestamp(k) <= pd.Timestamp(end_date)}
    for date, event in filtered_events.items():
        fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="gray")
        fig.add_annotation(
            x=date, y=1.02, yref="paper", text=event, showarrow=False,
            textangle=-90, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"
        )
    return fig

def calculate_country_insights(country_data, country_name):
    """Calculate key insights for a specific country"""
    insights = []
    if country_data.empty:
        insights.append(f"No data available for {country_name}")
        return insights
    country_data = country_data.sort_values('date')
    latest = country_data.iloc[-1]
    if 'total_cases' in country_data.columns:
        insights.append(f"Total reported cases: {int(latest['total_cases']):,}")
    if 'total_deaths' in country_data.columns:
        insights.append(f"Total reported deaths: {int(latest['total_deaths']):,}")
    if 'total_cases' in country_data.columns and 'total_deaths' in country_data.columns and latest['total_cases'] > 0:
        cfr = (latest['total_deaths'] / latest['total_cases']) * 100
        insights.append(f"Case fatality rate: {cfr:.2f}%")
    if 'new_cases_7day_avg' in country_data.columns and len(country_data) >= 28:
        recent_trend = country_data.iloc[-30:] if len(country_data) >= 30 else country_data
        last_two_weeks_avg = recent_trend['new_cases_7day_avg'].iloc[-14:].mean()
        previous_two_weeks_avg = recent_trend['new_cases_7day_avg'].iloc[-28:-14].mean() if len(recent_trend) >= 28 else None
        if previous_two_weeks_avg is not None and previous_two_weeks_avg > 0:
            percent_change = ((last_two_weeks_avg - previous_two_weeks_avg) / previous_two_weeks_avg) * 100
            trend_text = "increasing" if percent_change > 10 else ("decreasing" if percent_change < -10 else "stable")
            insights.append(f"Recent case trend: {trend_text} ({percent_change:.1f}% change over previous two weeks)")
    if 'people_vaccinated_per_hundred' in country_data.columns and not pd.isna(latest.get('people_vaccinated_per_hundred')):
        insights.append(f"Vaccination rate: {latest['people_vaccinated_per_hundred']:.1f}% with at least one dose")
    if 'total_cases' in country_data.columns and len(country_data) > 14:
        case_data = country_data['total_cases'].iloc[-14:]
        if case_data.iloc[0] > 0 and case_data.iloc[-1] > case_data.iloc[0]:
            growth_rate = (case_data.iloc[-1] / case_data.iloc[0]) ** (1/14) - 1
            if growth_rate > 0:
                doubling_time = round(np.log(2) / np.log(1 + growth_rate))
                if doubling_time < 100:
                    insights.append(f"Case doubling time: approximately {doubling_time} days")
    return insights

def detect_covid_waves(country_data, threshold=0.2):
    """Detect COVID-19 waves based on changes in the 7-day rolling average of new cases."""
    if 'new_cases_7day_avg' not in country_data.columns or len(country_data) < 30:
        return []
    data = country_data.sort_values('date')
    case_series = data['new_cases_7day_avg']
    max_value = case_series.max()
    if max_value <= 0:
        return []
    normalized = case_series / max_value
    peaks = []
    for i in range(14, len(normalized) - 14):
        window = normalized.iloc[i-14:i+15]
        if (normalized.iloc[i] == window.max() and 
            normalized.iloc[i] > threshold and
            normalized.iloc[i] > normalized.iloc[i-7] * 1.5):
            peaks.append(i)
    waves = []
    for peak_idx in peaks:
        peak_date = data.iloc[peak_idx]['date']
        peak_value = data.iloc[peak_idx]['new_cases_7day_avg']
        start_idx = peak_idx
        for j in range(peak_idx, max(0, peak_idx - 90), -1):
            if data.iloc[j]['new_cases_7day_avg'] < peak_value * 0.1:
                start_idx = j
                break
        end_idx = peak_idx
        for j in range(peak_idx, min(len(data) - 1, peak_idx + 90)):
            if data.iloc[j]['new_cases_7day_avg'] < peak_value * 0.1:
                end_idx = j
                break
        if end_idx > start_idx:
            wave_info = {
                'wave_start': data.iloc[start_idx]['date'],
                'wave_peak': peak_date,
                'wave_end': data.iloc[end_idx]['date'],
                'peak_cases': peak_value,
                'wave_duration': (data.iloc[end_idx]['date'] - data.iloc[start_idx]['date']).days
            }
            waves.append(wave_info)
    return waves

def run_statistical_analysis(country_data, metric='new_cases_7day_avg'):
    """Run comprehensive statistical analysis on country data."""
    if country_data.empty or metric not in country_data.columns:
        return None, "Insufficient data for statistical analysis."
    results = {}
    insights = []
    data = country_data[metric].dropna()
    if len(data) < 30:
        return None, "Insufficient data points (need at least 30)."
    results['mean'] = data.mean()
    results['median'] = data.median()
    results['std_dev'] = data.std()
    results['min'] = data.min()
    results['max'] = data.max()
    results['skewness'] = data.skew()
    results['kurtosis'] = stats.kurtosis(data.fillna(0))
    if len(data) >= 50:
        shapiro_test = stats.shapiro(data.sample(min(50, len(data))))
        results['normality_test'] = {
            'statistic': shapiro_test[0],
            'p_value': shapiro_test[1],
            'is_normal': shapiro_test[1] > 0.05
        }
        normality_insight = f"Distribution is {'normal' if results['normality_test']['is_normal'] else 'non-normal'}"
        insights.append(normality_insight)
    if len(data) >= 14:
        try:
            decomposition = seasonal_decompose(data, model='additive', period=7, extrapolate_trend='freq')
            results['decomposition'] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            seasonal_strength = abs(decomposition.seasonal).max()
            trend_strength = abs(decomposition.trend.dropna()).mean()
            if seasonal_strength > trend_strength * 0.3:
                insights.append(f"Strong 7-day seasonal patterns detected (strength: {seasonal_strength:.2f})")
            else:
                insights.append("No significant seasonal patterns detected")
        except:
            pass
    try:
        acf = sm.tsa.acf(data, nlags=14, fft=True)
        results['autocorrelation'] = acf
        if any(abs(acf[1:]) > 0.3):
            lag = np.argmax(abs(acf[1:])) + 1
            insights.append(f"Significant autocorrelation at lag {lag} days (value: {acf[lag]:.2f})")
    except:
        pass
    if len(data) >= 14:
        recent_growth = (data.iloc[-1] / data.iloc[-14] - 1) * 100 if data.iloc[-14] > 0 else 0
        results['recent_growth_rate'] = recent_growth
        growth_text = "increasing" if recent_growth > 10 else ("decreasing" if recent_growth < -10 else "stable")
        insights.append(f"Trend: {growth_text} ({recent_growth:.1f}% change over last 14 days)")
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    outliers = data[(data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))]
    results['outliers'] = {
        'count': len(outliers),
        'percentage': len(outliers) / len(data) * 100 if len(data) > 0 else 0
    }
    if results['outliers']['count'] > 0:
        insights.append(f"Detected {results['outliers']['count']} outliers ({results['outliers']['percentage']:.1f}%)")
    return results, insights

def generate_advanced_forecasting(df, country, metric, days=30):
    """Generate forecasts using ARIMA."""
    if df.empty or len(df[df['location'] == country]) < 30:
        return None, "Insufficient data for forecasting"
    country_data = df[df['location'] == country].sort_values('date')
    if metric not in country_data.columns:
        return None, f"Metric {metric} not available"
    forecast_data = country_data.tail(90).copy()
    last_date = forecast_data['date'].max()
    forecast_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    try:
        model = ARIMA(forecast_data[metric], order=(5,1,0)).fit()
        forecast = model.forecast(steps=days)
        conf_int = model.get_forecast(steps=days).conf_int(alpha=0.05)
        forecast_df = pd.DataFrame({
            'date': forecast_range,
            f'{metric}_forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        })
        result_df = pd.DataFrame({
            'date': forecast_data['date'],
            metric: forecast_data[metric]
        })
        full_df = pd.concat([result_df, forecast_df], ignore_index=True)
        return full_df, "Forecast generated using ARIMA"
    except Exception as e:
        return None, f"Forecasting error: {str(e)}"

def cluster_countries(df, features, n_clusters=4):
    """Cluster countries based on selected COVID-19 metrics."""
    if df.empty:
        return None, "No data available for clustering"
    latest_df = df.sort_values('date').groupby('location').tail(1).copy()
    valid_countries = latest_df.dropna(subset=features)
    if len(valid_countries) < 10:
        return None, "Not enough countries with complete data"
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(valid_countries[features])
    silhouette_scores = []
    max_clusters = min(8, len(valid_countries) // 5)
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        silhouette_scores.append((k, score))
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    valid_countries['cluster'] = kmeans.fit_predict(scaled_features)
    centroids = []
    for i in range(optimal_k):
        cluster_data = valid_countries[valid_countries['cluster'] == i]
        centroid = {
            'cluster': i,
            'size': len(cluster_data),
            'countries': cluster_data['location'].tolist()
        }
        for feature in features:
            centroid[feature] = cluster_data[feature].mean()
        centroids.append(centroid)
    return valid_countries, centroids

def analyze_covid_impact_factors(df):
    """Analyze factors affecting COVID-19 outcomes."""
    if df.empty:
        return None, "No data available"
    latest_by_country = df.sort_values('date').groupby('location').tail(1).copy()
    correlation_pairs = [
        ('population_density', 'total_cases_per_million'),
        ('median_age', 'case_fatality_rate'),
        ('aged_65_older', 'total_deaths_per_million'),
        ('gdp_per_capita', 'total_cases_per_million'),
        ('hospital_beds_per_thousand', 'case_fatality_rate'),
        ('life_expectancy', 'case_fatality_rate'),
        ('cardiovasc_death_rate', 'case_fatality_rate'),
        ('diabetes_prevalence', 'case_fatality_rate'),
        ('stringency_index', 'new_cases_per_million'),
        ('human_development_index', 'total_deaths_per_million')
    ]
    correlations = []
    for var1, var2 in correlation_pairs:
        if var1 in latest_by_country.columns and var2 in latest_by_country.columns:
            valid_data = latest_by_country.dropna(subset=[var1, var2])
            if len(valid_data) >= 20:
                corr, p_value = stats.pearsonr(valid_data[var1], valid_data[var2])
                correlations.append({
                    'variable_1': var1,
                    'variable_2': var2,
                    'correlation': corr,
                    'p_value': p_value,
                    'sample_size': len(valid_data),
                    'significance': p_value < 0.05
                })
    insights = []
    for corr in correlations:
        if corr['significance']:
            strength = abs(corr['correlation'])
            direction = "positive" if corr['correlation'] > 0 else "negative"
            strength_text = "strong" if strength > 0.7 else "moderate" if strength > 0.4 else "weak"
            insights.append(
                f"{strength_text} {direction} correlation ({corr['correlation']:.2f}) between "
                f"{corr['variable_1']} and {corr['variable_2']} (p={corr['p_value']:.4f})"
            )
    return correlations, insights

def calculate_pandemic_phases(df, major_countries):
    """Identify global pandemic phases."""
    if df.empty:
        return None
    global_data = df[df['location'] == 'World'].copy() if 'World' in df['location'].unique() else None
    if global_data is None or global_data.empty:
        if major_countries:
            global_data = df[df['location'].isin(major_countries)].groupby('date').agg({
                'new_cases': 'sum',
                'new_deaths': 'sum'
            }).reset_index()
        else:
            return None
    if len(global_data) < 60:
        return None
    global_data = global_data.sort_values('date')
    if 'new_cases_7day_avg' not in global_data.columns:
        global_data['new_cases_7day_avg'] = global_data['new_cases'].rolling(window=7, min_periods=1).mean()
    global_data['trend_direction'] = np.sign(global_data['new_cases_7day_avg'].diff(14))
    global_data['trend_change'] = global_data['trend_direction'].diff().abs() > 0
    phase_change_indices = global_data[global_data['trend_change']].index.tolist()
    phases = []
    last_phase_end = 0
    for idx in phase_change_indices:
        if idx - last_phase_end < 21:
            continue
        phase_start_date = global_data.loc[last_phase_end, 'date']
        phase_end_date = global_data.loc[idx, 'date']
        phase_data = global_data.loc[last_phase_end:idx]
        max_cases = phase_data['new_cases_7day_avg'].max()
        total_cases = phase_data['new_cases'].sum()
        phase_type = "Growth" if phase_data['new_cases_7day_avg'].iloc[-1] > phase_data['new_cases_7day_avg'].iloc[0] else "Decline"
        phases.append({
            'phase_number': len(phases) + 1,
            'start_date': phase_start_date,
            'end_date': phase_end_date,
            'duration_days': (phase_end_date - phase_start_date).days,
            'type': phase_type,
            'peak_daily_cases': max_cases,
            'total_cases': total_cases
        })
        last_phase_end = idx
    if last_phase_end < len(global_data) - 1:
        phase_start_date = global_data.loc[last_phase_end, 'date']
        phase_end_date = global_data.loc[len(global_data) - 1, 'date']
        phase_data = global_data.loc[last_phase_end:len(global_data) - 1]
        max_cases = phase_data['new_cases_7day_avg'].max()
        total_cases = phase_data['new_cases'].sum()
        phase_type = "Growth" if len(phase_data) > 1 and phase_data['new_cases_7day_avg'].iloc[-1] > phase_data['new_cases_7day_avg'].iloc[0] else "Decline"
        phases.append({
            'phase_number': len(phases) + 1,
            'start_date': phase_start_date,
            'end_date': phase_end_date,
            'duration_days': (phase_end_date - phase_start_date).days,
            'type': phase_type,
            'peak_daily_cases': max_cases,
            'total_cases': total_cases
        })
    return pd.DataFrame(phases)

def analyze_vaccination_impact(df, countries):
    """Analyze vaccination impact on outcomes."""
    if df.empty or not countries:
        return None, "No data available"
    if 'people_vaccinated_per_hundred' not in df.columns:
        return None, "Vaccination data not available"
    results = []
    for country in countries:
        country_data = df[df['location'] == country].copy().sort_values('date')
        if country_data.empty:
            continue
        vax_threshold_row = country_data[country_data['people_vaccinated_per_hundred'] >= 20].head(1)
        if vax_threshold_row.empty:
            continue
        threshold_date = vax_threshold_row.iloc[0]['date']
        before_window = country_data[(country_data['date'] < threshold_date) & 
                                   (country_data['date'] >= threshold_date - pd.Timedelta(days=90))]
        after_window = country_data[(country_data['date'] >= threshold_date) & 
                                  (country_data['date'] < threshold_date + pd.Timedelta(days=90))]
        if len(before_window) < 30 or len(after_window) < 30:
            continue
        metrics = {
            'country': country,
            'threshold_date': threshold_date,
            'vax_rate_at_threshold': vax_threshold_row.iloc[0]['people_vaccinated_per_hundred'],
            'days_before': len(before_window),
            'days_after': len(after_window)
        }
        for metric in ['new_cases_per_million', 'new_deaths_per_million']:
            if metric in before_window.columns:
                before_mean = before_window[metric].mean()
                after_mean = after_window[metric].mean()
                pct_change = ((after_mean - before_mean) / before_mean) * 100 if before_mean > 0 else np.nan
                metrics[f'{metric}_before'] = before_mean
                metrics[f'{metric}_after'] = after_mean
                metrics[f'{metric}_change_pct'] = pct_change
                try:
                    t_stat, p_value = stats.ttest_ind(before_window[metric].dropna(), after_window[metric].dropna())
                    metrics[f'{metric}_p_value'] = p_value
                    metrics[f'{metric}_significant'] = p_value < 0.05
                except:
                    metrics[f'{metric}_p_value'] = np.nan
                    metrics[f'{metric}_significant'] = False
        results.append(metrics)
    if not results:
        return None, "No countries with sufficient vaccination data"
    result_df = pd.DataFrame(results)
    insights = []
    sig_improvements = result_df[(result_df['new_cases_per_million_change_pct'] < 0) & 
                                (result_df['new_cases_per_million_significant'])]
    if len(sig_improvements) > 0:
        avg_reduction = sig_improvements['new_cases_per_million_change_pct'].mean()
        insights.append(f"Significant case reductions in {len(sig_improvements)} countries "
                       f"after 20% vaccination (avg reduction: {abs(avg_reduction):.1f}%)")
    return result_df, insights

def calculate_regional_patterns(df):
    """Calculate regional pandemic patterns."""
    if df.empty or 'continent' not in df.columns:
        return None
    continents = df['continent'].dropna().unique()
    results = []
    for continent in continents:
        continent_data = df[df['continent'] == continent].copy()
        if continent_data.empty:
            continue
        continent_agg = continent_data.groupby('date').agg({
            'new_cases': 'sum',
            'new_deaths': 'sum',
            'total_cases': 'sum',
            'total_deaths': 'sum',
            'population': 'sum'
        }).reset_index()
        if 'population' in continent_agg.columns:
            for col in ['new_cases', 'new_deaths', 'total_cases', 'total_deaths']:
                continent_agg[f'{col}_per_million'] = continent_agg[col] * 1000000 / continent_agg['population']
        continent_agg['new_cases_7day_avg'] = continent_agg['new_cases'].rolling(window=7, min_periods=1).mean()
        continent_agg['new_deaths_7day_avg'] = continent_agg['new_deaths'].rolling(window=7, min_periods=1).mean()
        latest = continent_agg.iloc[-1]
        peak_cases_row = continent_agg.loc[continent_agg['new_cases_7day_avg'].idxmax()]
        peak_deaths_row = continent_agg.loc[continent_agg['new_deaths_7day_avg'].idxmax()]
        cases_to_deaths_lag = (peak_deaths_row['date'] - peak_cases_row['date']).days
        cfr = (latest['total_deaths'] / latest['total_cases']) * 100 if latest['total_cases'] > 0 else np.nan
        results.append({
            'continent': continent,
            'population': latest['population'],
            'total_cases': latest['total_cases'],
            'total_deaths': latest['total_deaths'],
            'cases_per_million': latest.get('total_cases_per_million', np.nan),
            'deaths_per_million': latest.get('total_deaths_per_million', np.nan),
            'case_fatality_rate': cfr,
            'peak_cases_date': peak_cases_row['date'],
            'peak_cases_value': peak_cases_row['new_cases_7day_avg'],
            'peak_deaths_date': peak_deaths_row['date'],
            'peak_deaths_value': peak_deaths_row['new_deaths_7day_avg'],
            'cases_to_deaths_lag': cases_to_deaths_lag
        })
    return pd.DataFrame(results)

def detect_anomalies(country_data, metric='new_cases', window=7):
    """Detect anomalous data points."""
    if country_data.empty or metric not in country_data.columns:
        return country_data
    df = country_data.copy()
    df[f'{metric}_rolling_mean'] = df[metric].rolling(window=window, center=True).mean()
    df[f'{metric}_rolling_std'] = df[metric].rolling(window=window, center=True).std()
    df[f'{metric}_zscore'] = np.abs((df[metric] - df[f'{metric}_rolling_mean']) / df[f'{metric}_rolling_std'])
    df[f'{metric}_anomaly'] = df[f'{metric}_zscore'] > 3
    df[f'{metric}_anomaly'] = df[f'{metric}_anomaly'].fillna(False)
    return df

def analyze_weekend_effect(country_data, metric='new_cases'):
    """Analyze weekend effect on reporting."""
    if country_data.empty or metric not in country_data.columns or len(country_data) < 14:
        return None
    df = country_data.copy()
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
    day_avgs = df.groupby('day_of_week')[metric].mean().reset_index()
    weekdays = [0, 1, 2, 3, 4]
    weekends = [5, 6]
    weekday_avg = day_avgs[day_avgs['day_of_week'].isin(weekdays)][metric].mean()
    weekend_avg = day_avgs[day_avgs['day_of_week'].isin(weekends)][metric].mean()
    weekend_effect = ((weekend_avg - weekday_avg) / weekday_avg) * 100 if weekday_avg > 0 else 0
    min_day = day_avgs.loc[day_avgs[metric].idxmin()]
    max_day = day_avgs.loc[day_avgs[metric].idxmax()]
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    min_day_name = day_names[int(min_day['day_of_week'])]
    max_day_name = day_names[int(max_day['day_of_week'])]
    return {
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'weekend_effect_pct': weekend_effect,
        'min_reporting_day': min_day_name,
        'min_reporting_value': min_day[metric],
        'max_reporting_day': max_day_name,
        'max_reporting_value': max_day[metric],
        'day_averages': day_avgs
    }

def calculate_growth_rates(country_data):
    """Calculate growth rates and doubling times."""
    if country_data.empty or len(country_data) < 14:
        return None
    df = country_data.copy().sort_values('date')
    for metric in ['total_cases', 'total_deaths']:
        if metric in df.columns:
            df[f'{metric}_daily_growth'] = df[metric].pct_change() * 100
            df[f'{metric}_growth_7day_avg'] = df[f'{metric}_daily_growth'].rolling(window=7).mean()
            df[f'{metric}_doubling_time'] = np.log(2) / np.log(1 + df[f'{metric}_growth_7day_avg'] / 100)
            df[f'{metric}_doubling_time'] = df[f'{metric}_doubling_time'].replace([np.inf, -np.inf], np.nan)
    return df

def analyze_policy_impact(df, countries):
    """Analyze impact of policy stringency on outcomes."""
    if df.empty or 'stringency_index' not in df.columns:
        return None, "No policy data available"
    results = []
    for country in countries:
        country_data = df[df['location'] == country].copy().sort_values('date')
        if len(country_data) < 30 or country_data['stringency_index'].isna().all():
            continue
        high_stringency = country_data[country_data['stringency_index'] >= country_data['stringency_index'].quantile(0.75)]
        low_stringency = country_data[country_data['stringency_index'] <= country_data['stringency_index'].quantile(0.25)]
        if len(high_stringency) < 10 or len(low_stringency) < 10:
            continue
        metrics = {'country': country}
        for metric in ['new_cases_per_million', 'new_deaths_per_million']:
            if metric in country_data.columns:
                high_mean = high_stringency[metric].mean()
                low_mean = low_stringency[metric].mean()
                pct_change = ((high_mean - low_mean) / low_mean) * 100 if low_mean > 0 else np.nan
                metrics[f'{metric}_high'] = high_mean
                metrics[f'{metric}_low'] = low_mean
                metrics[f'{metric}_change_pct'] = pct_change
                try:
                    t_stat, p_value = stats.ttest_ind(high_stringency[metric].dropna(), low_stringency[metric].dropna())
                    metrics[f'{metric}_p_value'] = p_value
                    metrics[f'{metric}_significant'] = p_value < 0.05
                except:
                    metrics[f'{metric}_p_value'] = np.nan
                    metrics[f'{metric}_significant'] = False
        results.append(metrics)
    if not results:
        return None, "No countries with sufficient policy data"
    result_df = pd.DataFrame(results)
    insights = []
    sig_changes = result_df[(result_df['new_cases_per_million_change_pct'].notna()) & 
                           (result_df['new_cases_per_million_significant'])]
    if len(sig_changes) > 0:
        avg_change = sig_changes['new_cases_per_million_change_pct'].mean()
        insights.append(f"Policy stringency significantly affected cases in {len(sig_changes)} countries "
                       f"(avg change: {avg_change:.1f}%)")
    return result_df, insights

# Main application
try:
    with st.spinner('Loading COVID-19 data...'):
        df = load_data()
        if df.empty:
            st.error("No data loaded. Check internet connection or data source.")
            st.stop()
        last_update = df['date'].max().strftime('%Y-%m-%d')
        data_date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {last_update}"
        st.sidebar.info(f"Data range: {data_date_range}")
        st.sidebar.info(f"Countries/regions: {df['location'].nunique()}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Countries", f"{df['location'].nunique()}")
        with col2:
            st.info(f"Data last updated: {last_update}")
        continents = ['World', 'Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
        countries = sorted([c for c in df['location'].unique() if c not in continents])
        st.sidebar.header("📊 Data Filters")
        default_countries = ["United States", "United Kingdom", "India", "Brazil", "South Africa"]
        default_countries = [c for c in default_countries if c in countries]
        if not default_countries and countries:
            default_countries = [countries[0]]
        selected_countries = st.sidebar.multiselect(
            "Select Countries", countries, default=default_countries
        )
        with st.sidebar.expander("Advanced Filters", expanded=False):
            include_world = st.checkbox("Include global data", value=True)
            if include_world and "World" in df['location'].unique():
                if "World" not in selected_countries:
                    selected_countries = ["World"] + selected_countries
            continent_options = ['All'] + sorted(df['continent'].dropna().unique().tolist())
            selected_continent = st.selectbox("Select Continent", continent_options)
            default_end_date = df['date'].max()
            default_start_date = default_end_date - timedelta(days=365)
            start_date = st.date_input(
                "Start Date", value=default_start_date.date(),
                min_value=df['date'].min().date(), max_value=default_end_date.date()
            )
            end_date = st.date_input(
                "End Date", value=default_end_date.date(),
                min_value=start_date, max_value=default_end_date.date()
            )
            show_events = st.checkbox("Show significant events", value=True)
            anomaly_threshold = st.slider("Anomaly detection threshold", 2.0, 5.0, 3.0, 0.1)
            normalize_options = st.radio(
                "Data normalization", ["Absolute numbers", "Per million people", "Per capita (%)"]
            )
        st.sidebar.subheader("Select Metrics")
        available_metrics = {
            "Cases": {
                "New Cases": "new_cases",
                "Total Cases": "total_cases",
                "New Cases (7-day avg)": "new_cases_7day_avg",
                "Case Growth Rate (%)": "new_cases_growth_rate"
            },
            "Deaths": {
                "New Deaths": "new_deaths",
                "Total Deaths": "total_deaths",
                "New Deaths (7-day avg)": "new_deaths_7day_avg",
                "Death Growth Rate (%)": "new_deaths_growth_rate"
            }
        }
        if 'people_vaccinated_per_hundred' in df.columns:
            available_metrics["Vaccination"] = {
                "Vaccination Rate (%)": "people_vaccinated_per_hundred",
                "Fully Vaccinated (%)": "people_fully_vaccinated_per_hundred",
                "Booster Rate (%)": "total_boosters_per_hundred" if "total_boosters_per_hundred" in df.columns else None
            }
            available_metrics["Vaccination"] = {k: v for k, v in available_metrics["Vaccination"].items() if v is not None}
        hospital_metrics = {}
        for col in ['icu_patients', 'hosp_patients', 'new_tests', 'positive_rate']:
            if col in df.columns:
                hospital_metrics[col.replace('_', ' ').title()] = col
        if hospital_metrics:
            available_metrics["Healthcare"] = hospital_metrics
        selected_metric_category = st.sidebar.selectbox("Metric Category", list(available_metrics.keys()))
        selected_metric = st.sidebar.selectbox("Select Metric", list(available_metrics[selected_metric_category].keys()))
        metric_column = available_metrics[selected_metric_category][selected_metric]
        if not selected_countries:
            st.warning("Please select at least one country.")
            st.stop()
        filtered_df = df[
            (df['location'].isin(selected_countries)) &
            (df['date'] >= pd.Timestamp(start_date)) &
            (df['date'] <= pd.Timestamp(end_date))
        ]
        if selected_continent != 'All':
            filtered_df = filtered_df[filtered_df['continent'] == selected_continent]
        if filtered_df.empty:
            st.warning("No data available for selected filters.")
            st.stop()
        if normalize_options == "Per million people":
            display_suffix = "_per_million"
            if f"{metric_column}{display_suffix}" in filtered_df.columns:
                display_metric = f"{metric_column}{display_suffix}"
                metric_title = f"{selected_metric} per Million People"
            else:
                if 'population' in filtered_df.columns:
                    filtered_df[f"{metric_column}_per_million"] = filtered_df.apply(
                        lambda row: row[metric_column] * 1000000 / row['population'] 
                        if pd.notna(row['population']) and row['population'] > 0 else np.nan, axis=1
                    )
                    display_metric = f"{metric_column}_per_million"
                    metric_title = f"{selected_metric} per Million People"
                else:
                    display_metric = metric_column
                    metric_title = selected_metric
                    st.warning("Population data not available for per million calculation")
        elif normalize_options == "Per capita (%)":
            display_suffix = "_per_capita_pct"
            if 'population' in filtered_df.columns:
                filtered_df[f"{metric_column}{display_suffix}"] = filtered_df.apply(
                    lambda row: row[metric_column] * 100 / row['population'] 
                    if pd.notna(row['population']) and row['population'] > 0 else np.nan, axis=1
                )
                display_metric = f"{metric_column}{display_suffix}"
                metric_title = f"{selected_metric} (% of Population)"
            else:
                display_metric = metric_column
                metric_title = selected_metric
                st.warning("Population data not available for per capita calculation")
        else:
            display_metric = metric_column
            metric_title = selected_metric
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Trends & Insights", "🔍 Statistical Analysis", "🗺️ Geographic Analysis",
            "🧮 Comparative Analysis", "🔮 Forecasting & Modeling"
        ])
        with tab1:
            st.markdown(f"<h2 class='chart-header'>{metric_title} Analysis</h2>", unsafe_allow_html=True)
            try:
                fig = px.line(
                    filtered_df, x='date', y=display_metric, color='location',
                    title=f"{metric_title} Trends Over Time",
                    labels={'date': 'Date', display_metric: metric_title, 'location': 'Country'}
                )
                if show_events:
                    fig = add_timeline_events(fig, start_date, end_date)
                fig.update_layout(
                    height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("📊 Trend Insights", expanded=True):
                    trends_col1, trends_col2 = st.columns(2)
                    with trends_col1:
                        st.markdown("### Key Observations")
                        latest_by_country = filtered_df.sort_values('date').groupby('location').tail(1)
                        latest_by_country = latest_by_country.sort_values(display_metric, ascending=False)
                        if len(latest_by_country) > 1:
                            highest = latest_by_country.iloc[0]
                            lowest = latest_by_country.iloc[-1]
                            st.markdown(f"- **Highest {metric_title}**: {highest['location']} ({highest[display_metric]:.2f})")
                            st.markdown(f"- **Lowest {metric_title}**: {lowest['location']} ({lowest[display_metric]:.2f})")
                        if len(filtered_df) >= 14:
                            recent_by_country = filtered_df.sort_values('date').groupby('location').tail(14)
                            growth_rates = []
                            for country in selected_countries:
                                country_recent = recent_by_country[recent_by_country['location'] == country]
                                if len(country_recent) >= 7:
                                    first_value = country_recent.iloc[0][display_metric]
                                    last_value = country_recent.iloc[-1][display_metric]
                                    if first_value > 0:
                                        growth_rate = ((last_value - first_value) / first_value) * 100
                                        growth_rates.append((country, growth_rate))
                            if growth_rates:
                                fastest_growth = max(growth_rates, key=lambda x: x[1])
                                fastest_decline = min(growth_rates, key=lambda x: x[1])
                                st.markdown(f"- **Fastest Growth**: {fastest_growth[0]} ({fastest_growth[1]:.2f}% over 14 days)")
                                st.markdown(f"- **Biggest Decline**: {fastest_decline[0]} ({fastest_decline[1]:.2f}% over 14 days)")
                        if len(filtered_df) > 30:
                            st.markdown("### Detected Patterns")
                            for country in selected_countries:
                                country_data = filtered_df[filtered_df['location'] == country]
                                if len(country_data) > 30:
                                    weekend_effect = analyze_weekend_effect(country_data, metric_column)
                                    if weekend_effect and abs(weekend_effect['weekend_effect_pct']) > 10:
                                        effect_type = "lower" if weekend_effect['weekend_effect_pct'] < 0 else "higher"
                                        st.markdown(f"- **{country}**: {effect_type} weekend reporting "
                                                  f"({weekend_effect['weekend_effect_pct']:.1f}%). "
                                                  f"Lowest on {weekend_effect['min_reporting_day']}.")
                                    waves = detect_covid_waves(country_data)
                                    if waves:
                                        st.markdown(f"- **{country}**: {len(waves)} waves, highest peak on "
                                                  f"{waves[0]['wave_peak'].strftime('%Y-%m-%d')}.")
                    with trends_col2:
                        st.markdown("### Data Quality Analysis")
                        anomaly_counts = {}
                        for country in selected_countries:
                            country_data = filtered_df[filtered_df['location'] == country]
                            country_data = detect_anomalies(country_data, metric_column, window=7)
                            anomaly_count = country_data[f'{metric_column}_anomaly'].sum()
                            if anomaly_count > 0:
                                anomaly_counts[country] = anomaly_count
                        if anomaly_counts:
                            st.markdown("#### Anomalies Detected")
                            for country, count in anomaly_counts.items():
                                st.markdown(f"- **{country}**: {count} anomalous data points")
                            st.markdown("Anomalies may indicate reporting issues or significant events.")
                        else:
                            st.markdown("No significant anomalies detected.")
                        missing_data = {}
                        for country in selected_countries:
                            country_data = filtered_df[filtered_df['location'] == country]
                            date_range = pd.date_range(start=country_data['date'].min(), end=country_data['date'].max())
                            missing_dates = set(date_range) - set(country_data['date'])
                            if missing_dates:
                                missing_data[country] = len(missing_dates)
                        if missing_data:
                            st.markdown("#### Missing Data Points")
                            for country, count in missing_data.items():
                                st.markdown(f"- **{country}**: {count} missing dates")
                            st.markdown("Missing data may affect trend analysis.")
                        else:
                            st.markdown("No missing dates detected.")
            except Exception as e:
                st.error(f"Error creating trends chart: {e}")
            st.markdown("## Country-Specific Insights")
            country_tabs = st.tabs(selected_countries)
            for i, country in enumerate(selected_countries):
                with country_tabs[i]:
                    country_data = filtered_df[filtered_df['location'] == country].sort_values('date')
                    if country_data.empty:
                        st.warning("No data available for this country.")
                        continue
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    latest = country_data.iloc[-1]
                    with metrics_col1:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-title'>Total Cases</div>", unsafe_allow_html=True)
                        value = f"{int(latest['total_cases']):,}" if 'total_cases' in latest else "N/A"
                        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with metrics_col2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-title'>Total Deaths</div>", unsafe_allow_html=True)
                        value = f"{int(latest['total_deaths']):,}" if 'total_deaths' in latest else "N/A"
                        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with metrics_col3:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-title'>Case Fatality Rate</div>", unsafe_allow_html=True)
                        cfr = (latest['total_deaths'] / latest['total_cases'] * 100) if 'total_cases' in latest and latest['total_cases'] > 0 else np.nan
                        value = f"{cfr:.2f}%" if pd.notna(cfr) else "N/A"
                        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with metrics_col4:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-title'>Vaccination Rate</div>", unsafe_allow_html=True)
                        value = f"{latest['people_vaccinated_per_hundred']:.1f}%" if 'people_vaccinated_per_hundred' in latest and pd.notna(latest['people_vaccinated_per_hundred']) else "N/A"
                        st.markdown(f"<div class='metric-value'>{value}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    insights = calculate_country_insights(country_data, country)
                    st.markdown("### Key Insights")
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    if len(country_data) > 30:
                        waves = detect_covid_waves(country_data)
                        if waves:
                            st.markdown("### COVID-19 Waves")
                            for wave in waves:
                                st.markdown(f"- Wave from {wave['wave_start'].strftime('%Y-%m-%d')} to "
                                           f"{wave['wave_end'].strftime('%Y-%m-%d')}, peak on "
                                           f"{wave['wave_peak'].strftime('%Y-%m-%d')} ({wave['peak_cases']:,.0f} cases)")
        with tab2:
            st.markdown("<h2 class='chart-header'>Statistical Analysis</h2>", unsafe_allow_html=True)
            for country in selected_countries:
                st.markdown(f"### {country}")
                country_data = filtered_df[filtered_df['location'] == country]
                stats_results, stats_insights = run_statistical_analysis(country_data, display_metric)
                if stats_results:
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.markdown("#### Statistical Summary")
                        st.markdown(f"- Mean: {stats_results['mean']:.2f}")
                        st.markdown(f"- Median: {stats_results['median']:.2f}")
                        st.markdown(f"- Standard Deviation: {stats_results['std_dev']:.2f}")
                        st.markdown(f"- Skewness: {stats_results['skewness']:.2f}")
                        st.markdown(f"- Kurtosis: {stats_results['kurtosis']:.2f}")
                        if 'outliers' in stats_results:
                            st.markdown(f"- Outliers: {stats_results['outliers']['count']} "
                                       f"({stats_results['outliers']['percentage']:.1f}%)")
                    with stats_col2:
                        st.markdown("#### Insights")
                        for insight in stats_insights:
                            st.markdown(f"- {insight}")
                    if 'decomposition' in stats_results:
                        fig_decomp = make_subplots(
                            rows=3, cols=1, subplot_titles=("Trend", "Seasonal", "Residual"),
                            shared_xaxes=True, vertical_spacing=0.1
                        )
                        fig_decomp.add_trace(go.Scatter(x=country_data['date'], y=stats_results['decomposition']['trend'], 
                                                       name="Trend", line=dict(color='blue')), row=1, col=1)
                        fig_decomp.add_trace(go.Scatter(x=country_data['date'], y=stats_results['decomposition']['seasonal'], 
                                                       name="Seasonal", line=dict(color='green')), row=2, col=1)
                        fig_decomp.add_trace(go.Scatter(x=country_data['date'], y=stats_results['decomposition']['residual'], 
                                                       name="Residual", line=dict(color='red')), row=3, col=1)
                        fig_decomp.update_layout(height=600, title_text=f"Time Series Decomposition for {country}")
                        st.plotly_chart(fig_decomp, use_container_width=True)
                else:
                    st.warning(stats_insights)
        with tab3:
            st.markdown("<h2 class='chart-header'>Geographic Analysis</h2>", unsafe_allow_html=True)
            if 'iso_code' in filtered_df.columns and 'total_cases_per_million' in filtered_df.columns:
                latest_geo = filtered_df.sort_values('date').groupby('location').tail(1)
                fig_geo = px.choropleth(
                    latest_geo, locations='iso_code', color='total_cases_per_million',
                    hover_name='location', color_continuous_scale=px.colors.sequential.Plasma,
                    title='Global COVID-19 Cases per Million',
                    labels={'total_cases_per_million': 'Cases per Million'}
                )
                fig_geo.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
                st.plotly_chart(fig_geo, use_container_width=True)
                if 'total_deaths_per_million' in filtered_df.columns:
                    fig_geo_deaths = px.choropleth(
                        latest_geo, locations='iso_code', color='total_deaths_per_million',
                        hover_name='location', color_continuous_scale=px.colors.sequential.Inferno,
                        title='Global COVID-19 Deaths per Million',
                        labels={'total_deaths_per_million': 'Deaths per Million'}
                    )
                    fig_geo_deaths.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
                    st.plotly_chart(fig_geo_deaths, use_container_width=True)
            else:
                st.warning("Geographic data not available.")
            regional_data = calculate_regional_patterns(filtered_df)
            if regional_data is not None:
                st.markdown("### Regional Patterns")
                st.dataframe(regional_data[['continent', 'cases_per_million', 'deaths_per_million', 
                                          'case_fatality_rate', 'cases_to_deaths_lag']].round(2))
                fig_regional = px.bar(
                    regional_data, x='continent', y=['cases_per_million', 'deaths_per_million'],
                    barmode='group', title='Regional Comparison of Cases and Deaths per Million'
                )
                st.plotly_chart(fig_regional, use_container_width=True)
        with tab4:
            st.markdown("<h2 class='chart-header'>Comparative Analysis</h2>", unsafe_allow_html=True)
            correlations, corr_insights = analyze_covid_impact_factors(filtered_df)
            if correlations:
                st.markdown("### Impact Factors")
                corr_df = pd.DataFrame(correlations).round(3)
                st.dataframe(corr_df)
                st.markdown("#### Correlation Insights")
                for insight in corr_insights:
                    st.markdown(f"- {insight}")
                fig_corr = px.bar(
                    corr_df[corr_df['significance']], x='variable_1', y='correlation',
                    color='variable_2', title='Significant Correlations with Outcomes'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            vax_impact, vax_insights = analyze_vaccination_impact(filtered_df, selected_countries)
            if vax_impact is not None:
                st.markdown("### Vaccination Impact")
                st.dataframe(vax_impact.round(2))
                st.markdown("#### Vaccination Insights")
                for insight in vax_insights:
                    st.markdown(f"- {insight}")
                fig_vax = px.bar(
                    vax_impact, x='country', y=['new_cases_per_million_before', 'new_cases_per_million_after'],
                    barmode='group', title='Cases Before vs. After 20% Vaccination'
                )
                st.plotly_chart(fig_vax, use_container_width=True)
            policy_impact, policy_insights = analyze_policy_impact(filtered_df, selected_countries)
            if policy_impact is not None:
                st.markdown("### Policy Impact")
                st.dataframe(policy_impact.round(2))
                st.markdown("#### Policy Insights")
                for insight in policy_insights:
                    st.markdown(f"- {insight}")
                fig_policy = px.bar(
                    policy_impact, x='country', y=['new_cases_per_million_high', 'new_cases_per_million_low'],
                    barmode='group', title='Cases Under High vs. Low Policy Stringency'
                )
                st.plotly_chart(fig_policy, use_container_width=True)
        with tab5:
            st.markdown("<h2 class='chart-header'>Forecasting & Modeling</h2>", unsafe_allow_html=True)
            for country in selected_countries:
                st.markdown(f"### {country} Forecast")
                forecast_data, forecast_message = generate_advanced_forecasting(filtered_df, country, display_metric)
                if forecast_data is not None:
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_data['date'], y=forecast_data[display_metric], 
                        name="Actual", line=dict(color='blue')
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_data['date'], y=forecast_data[f'{display_metric}_forecast'], 
                        name="Forecast", line=dict(color='red', dash='dash')
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_data['date'], y=forecast_data['upper_bound'], 
                        name="Upper CI", line=dict(color='rgba(255,0,0,0.2)'), fill='tonexty'
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_data['date'], y=forecast_data['lower_bound'], 
                        name="Lower CI", line=dict(color='rgba(255,0,0,0.2)'), fill='tonexty'
                    ))
                    fig_forecast.update_layout(
                        title=f"30-Day Forecast for {country} ({metric_title})",
                        xaxis_title="Date", yaxis_title=metric_title
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    st.success(forecast_message)
                else:
                    st.warning(forecast_message)
            cluster_features = ['total_cases_per_million', 'total_deaths_per_million', 'case_fatality_rate']
            valid_features = [f for f in cluster_features if f in filtered_df.columns]
            if len(valid_features) >= 2:
                clustered_data, centroids = cluster_countries(filtered_df, valid_features)
                if clustered_data is not None:
                    st.markdown("### Country Clustering")
                    st.markdown("Countries grouped by similarity in pandemic metrics:")
                    for centroid in centroids:
                        countries_sample = ', '.join(centroid['countries'][:5])
                        if len(centroid['countries']) > 5:
                            countries_sample += f" and {len(centroid['countries'])-5} more"
                        st.markdown(f"- **Cluster {centroid['cluster']+1}** ({centroid['size']} countries): {countries_sample}")
                        for feature in valid_features:
                            st.markdown(f"  - {feature}: {centroid[feature]:.2f}")
                    fig_cluster = px.scatter(
                        clustered_data, x=valid_features[0], y=valid_features[1], 
                        color='cluster', hover_name='location',
                        title='Country Clusters Based on Pandemic Metrics'
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
        st.sidebar.header("📥 Download Data")
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="covid_filtered_data.csv">Download filtered data as CSV</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error(traceback.format_exc())