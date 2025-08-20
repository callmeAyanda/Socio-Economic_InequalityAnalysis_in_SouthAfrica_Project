# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from io import StringIO, BytesIO
import folium
from streamlit_folium import folium_static
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="SA Socio-Economic Inequality Dashboard",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #0072B2; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #009E73; border-bottom: 2px solid #009E73; padding-bottom: 0.3rem; margin-top: 1.5rem;}
    .metric-label {font-weight: bold; color: #D55E00;}
    .highlight {background-color: #000000; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .footer {text-align: center; margin-top: 2rem; color: #999; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">South African Socio-Economic Inequality Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p>Analyzing spatial inequality and service delivery across South African municipalities</p>
</div>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_municipal_data():
    """Load municipal data from Stats SA"""
    try:
        # In a real implementation, we would fetch from actual Stats SA API
        # For demonstration, we'll create synthetic data based on real patterns
        
        # Municipalities and their provinces
        municipalities = [
            'City of Johannesburg', 'City of Cape Town', 'eThekwini', 'City of Tshwane',
            'Nelson Mandela Bay', 'Buffalo City', 'Mangaung', 'Ekurhuleni',
            'uMgungundlovu', 'uThukela', 'Amajuba', 'Zululand', 'Waterberg',
            'Capricorn', 'Mopani', 'Vhembe', 'Gert Sibande', 'Nkangala', 'Chris Hani'
        ]
        
        provinces = [
            'Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Gauteng',
            'Eastern Cape', 'Eastern Cape', 'Free State', 'Gauteng',
            'KwaZulu-Natal', 'KwaZulu-Natal', 'KwaZulu-Natal', 'KwaZulu-Natal', 'Limpopo',
            'Limpopo', 'Limpopo', 'Limpopo', 'Mpumalanga', 'Mpumalanga', 'Eastern Cape'
        ]
        
        # Generate synthetic data based on real South African inequality patterns
        np.random.seed(42)  # For reproducible results
        
        data = {
            'municipality': municipalities * 5,  # 5 years of data
            'province': provinces * 5,
            'year': sorted([2018, 2019, 2020, 2021, 2022] * len(municipalities)),
            'unemployment_rate': np.random.normal(30, 10, len(municipalities) * 5).clip(15, 50),
            'access_to_water': np.random.normal(75, 15, len(municipalities) * 5).clip(40, 99),
            'access_to_electricity': np.random.normal(80, 12, len(municipalities) * 5).clip(50, 99),
            'median_income': np.random.lognormal(10.5, 0.4, len(municipalities) * 5).clip(50000, 350000),
            'education_index': np.random.beta(2, 1.5, len(municipalities) * 5) * 0.6 + 0.4,
            'gini_coefficient': np.random.beta(2, 1, len(municipalities) * 5) * 0.4 + 0.5,
            'population_density': np.random.lognormal(6, 1.5, len(municipalities) * 5).clip(50, 2000)
        }
        
        # Create some realistic patterns
        df = pd.DataFrame(data)
        
        # Adjust values based on province to reflect real SA patterns
        province_adjustments = {
            'Gauteng': {'unemployment_rate': -5, 'median_income': 1.3, 'education_index': 0.1},
            'Western Cape': {'unemployment_rate': -7, 'median_income': 1.4, 'education_index': 0.15},
            'KwaZulu-Natal': {'unemployment_rate': 3, 'median_income': 0.9, 'education_index': -0.05},
            'Eastern Cape': {'unemployment_rate': 8, 'median_income': 0.8, 'education_index': -0.08},
            'Limpopo': {'unemployment_rate': 6, 'median_income': 0.85, 'education_index': -0.07},
            'Mpumalanga': {'unemployment_rate': 2, 'median_income': 0.95, 'education_index': -0.03},
            'Free State': {'unemployment_rate': 4, 'median_income': 0.9, 'education_index': -0.04}
        }
        
        for province, adjustments in province_adjustments.items():
            mask = df['province'] == province
            df.loc[mask, 'unemployment_rate'] += adjustments['unemployment_rate']
            df.loc[mask, 'median_income'] *= adjustments['median_income']
            df.loc[mask, 'education_index'] += adjustments['education_index']
        
        # Add time trends
        for year in [2018, 2019, 2020, 2021, 2022]:
            year_mask = df['year'] == year
            if year >= 2020:  # COVID impact
                df.loc[year_mask, 'unemployment_rate'] += (year - 2019) * 2
                
            # General improvement trends
            df.loc[year_mask, 'access_to_water'] += (year - 2018) * 0.5
            df.loc[year_mask, 'access_to_electricity'] += (year - 2018) * 0.7
            df.loc[year_mask, 'education_index'] += (year - 2018) * 0.02
            
        return df
    
    except Exception as e:
        st.error(f"Error loading municipal data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_geojson_data():
    """Load GeoJSON data for South African municipalities"""
    # In a real implementation, we would load actual GeoJSON from MDB
    # For demonstration, we'll return None and use a different visualization approach
    return None

# Load data
with st.spinner("Loading South African socio-economic data..."):
    df = load_municipal_data()
    geo_data = load_geojson_data()

# Sidebar filters
st.sidebar.header("Filter Data")
selected_year = st.sidebar.selectbox("Select Year", options=sorted(df['year'].unique(), reverse=True))
selected_province = st.sidebar.selectbox("Select Province", options=["All"] + list(df['province'].unique()))

# Apply filters
filtered_df = df[df['year'] == selected_year]
if selected_province != "All":
    filtered_df = filtered_df[filtered_df['province'] == selected_province]

# Calculate inequality indices
def calculate_composite_index(row):
    """Calculate a composite inequality index based on multiple factors"""
    # Normalize each component to 0-1 scale
    unemployment_norm = 1 - (row['unemployment_rate'] / 100)  # Lower is better
    water_norm = row['access_to_water'] / 100  # Higher is better
    electricity_norm = row['access_to_electricity'] / 100  # Higher is better
    income_norm = np.log(row['median_income']) / np.log(350000)  # Log scale, higher is better
    education_norm = row['education_index']  # Already normalized
    
    # Calculate composite index (weighted average)
    weights = {
        'unemployment': 0.25,
        'water': 0.15,
        'electricity': 0.15,
        'income': 0.25,
        'education': 0.20
    }
    
    composite = (unemployment_norm * weights['unemployment'] +
                water_norm * weights['water'] +
                electricity_norm * weights['electricity'] +
                income_norm * weights['income'] +
                education_norm * weights['education'])
    
    return composite * 100  # Scale to 0-100

# Apply the composite index calculation
filtered_df['composite_index'] = filtered_df.apply(calculate_composite_index, axis=1)

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_income = filtered_df['median_income'].median()
    st.metric("Median Household Income", f"R{avg_income:,.0f}")

with col2:
    unemployment = filtered_df['unemployment_rate'].mean()
    st.metric("Average Unemployment Rate", f"{unemployment:.1f}%")

with col3:
    water_access = filtered_df['access_to_water'].mean()
    st.metric("Access to Clean Water", f"{water_access:.1f}%")

with col4:
    inequality_index = filtered_df['composite_index'].mean()
    st.metric("Composite Development Index", f"{inequality_index:.1f}/100")

# Visualizations
st.markdown('<div class="sub-header">Spatial Inequality Analysis</div>', unsafe_allow_html=True)

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Income Distribution", "Service Access", "Temporal Trends", "Inequality Index"])

with tab1:
    st.plotly_chart(px.box(filtered_df, x='province', y='median_income', 
                          title=f"Income Distribution by Province ({selected_year})",
                          labels={'median_income': 'Median Income (ZAR)', 'province': 'Province'}),
                   use_container_width=True)

with tab2:
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Access to Water', 'Access to Electricity'))
    
    fig.add_trace(go.Bar(x=filtered_df['province'], y=filtered_df['access_to_water'],
                        name='Water Access', marker_color='blue'), 1, 1)
    
    fig.add_trace(go.Bar(x=filtered_df['province'], y=filtered_df['access_to_electricity'],
                        name='Electricity Access', marker_color='orange'), 1, 2)
    
    fig.update_layout(height=400, showlegend=False, title_text="Service Access by Province")
    fig.update_yaxes(title_text="Percentage", range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Compare selected year with previous year
    prev_year = selected_year - 1
    if prev_year in df['year'].unique():
        prev_df = df[df['year'] == prev_year]
        if selected_province != "All":
            prev_df = prev_df[prev_df['province'] == selected_province]
        
        # Calculate changes
        indicators = ['unemployment_rate', 'access_to_water', 'access_to_electricity', 'median_income']
        changes = {}
        
        for indicator in indicators:
            current_val = filtered_df[indicator].mean()
            prev_val = prev_df[indicator].mean()
            changes[indicator] = current_val - prev_val
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(changes.keys()),
            y=list(changes.values()),
            marker_color=['red' if val > 0 else 'green' if indicator != 'median_income' else 'green' if val > 0 else 'red' 
                         for indicator, val in changes.items()]
        ))
        
        fig.update_layout(
            title=f"Change in Indicators from {prev_year} to {selected_year}",
            yaxis_title="Percentage Change" if 'median_income' not in indicators else "Absolute Change",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No data available for {prev_year} to compare with {selected_year}")

with tab4:
    # Scatter plot showing relationship between income and inequality
    fig = px.scatter(filtered_df, x='median_income', y='gini_coefficient',
                    color='province', size='population_density',
                    hover_name='municipality',
                    title="Income vs Inequality by Municipality",
                    labels={'median_income': 'Median Income (ZAR)', 
                           'gini_coefficient': 'Gini Coefficient',
                           'province': 'Province',
                           'population_density': 'Population Density'})
    
    st.plotly_chart(fig, use_container_width=True)

# Municipality comparison
st.markdown('<div class="sub-header">Municipality Comparison</div>', unsafe_allow_html=True)

selected_municipalities = st.multiselect(
    "Select municipalities to compare:",
    options=list(filtered_df['municipality'].unique()),
    default=list(filtered_df['municipality'].unique())[:3]
)

if selected_municipalities:
    compare_df = filtered_df[filtered_df['municipality'].isin(selected_municipalities)]
    
    # Create radar chart for comparison
    categories = ['unemployment_rate', 'access_to_water', 'access_to_electricity', 
                 'education_index', 'gini_coefficient', 'composite_index']
    
    # Normalize values for radar chart
    normalized_data = []
    for municipality in selected_municipalities:
        mun_data = compare_df[compare_df['municipality'] == municipality].iloc[0]
        values = [mun_data[cat] for cat in categories]
        
        # Normalize each value based on column min-max (except for composite index which is already 0-100)
        normalized_values = []
        for i, cat in enumerate(categories):
            if cat == 'composite_index':
                normalized_values.append(values[i] / 100)
            else:
                col_min = filtered_df[cat].min()
                col_max = filtered_df[cat].max()
                if cat == 'gini_coefficient':
                    normalized_values.append((values[i] - col_min) / (col_max - col_min))
                else:
                    normalized_values.append((values[i] - col_min) / (col_max - col_min))
        
        normalized_data.append(normalized_values)
    
    # Create radar chart
    fig = go.Figure()
    
    for i, municipality in enumerate(selected_municipalities):
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[i],
            theta=categories,
            fill='toself',
            name=municipality
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Municipality Comparison (Normalized Values)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Key insights section
st.markdown('<div class="sub-header">Key Insights</div>', unsafe_allow_html=True)

# Calculate some insights
most_unequal = filtered_df.loc[filtered_df['gini_coefficient'].idxmax()]
least_unequal = filtered_df.loc[filtered_df['gini_coefficient'].idxmin()]
highest_income = filtered_df.loc[filtered_df['median_income'].idxmax()]
lowest_income = filtered_df.loc[filtered_df['median_income'].idxmin()]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="highlight">
        <h4>Inequality Patterns</h4>
        <ul>
            <li>Highest inequality: <span class="metric-label">{most_unequal['municipality']}</span> (Gini: {most_unequal['gini_coefficient']:.3f})</li>
            <li>Lowest inequality: <span class="metric-label">{least_unequal['municipality']}</span> (Gini: {least_unequal['gini_coefficient']:.3f})</li>
            <li>Income ratio (highest:lowest): <span class="metric-label">{highest_income['median_income']/lowest_income['median_income']:.1f}x</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="highlight">
        <h4>Service Delivery</h4>
        <ul>
            <li>Water access range: <span class="metric-label">{filtered_df['access_to_water'].min():.1f}% - {filtered_df['access_to_water'].max():.1f}%</span></li>
            <li>Electricity access range: <span class="metric-label">{filtered_df['access_to_electricity'].min():.1f}% - {filtered_df['access_to_electricity'].max():.1f}%</span></li>
            <li>Unemployment range: <span class="metric-label">{filtered_df['unemployment_rate'].min():.1f}% - {filtered_df['unemployment_rate'].max():.1f}%</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Data source and methodology
with st.expander("Data Sources and Methodology"):
    st.markdown("""
    ### Data Sources
    This dashboard uses synthetic data modeled after real South African socio-economic patterns.
    In a production environment, it would connect to:
    
    - **Statistics South Africa (Stats SA)**: Census and Community Survey data
    - **Municipal Money**: National Treasury's local government financial data
    - **Municipal Demarcation Board**: Geographic boundaries
    
    ### Methodology
    - **Composite Index**: Weighted average of normalized indicators (unemployment, water access, electricity access, income, education)
    - **Gini Coefficient**: Measure of income inequality (0 = perfect equality, 1 = perfect inequality)
    - **Data Normalization**: Min-max scaling for radar chart comparisons
    
    ### Limitations
    - Synthetic data used for demonstration purposes
    - Simplified inequality model (real analysis would include more factors)
    - Geographic visualizations limited by available GeoJSON data
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>South African Socio-Economic Inequality Dashboard | Created with Streamlit</p>
    <p>Data synthesized based on typical South African inequality patterns</p>
</div>
""", unsafe_allow_html=True)
