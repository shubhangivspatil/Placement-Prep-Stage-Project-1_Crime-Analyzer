import streamlit as st
import pandas as pd
import os
import pickle
import logging
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset function
@st.cache_data
def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

# Function to preprocess the data
@st.cache_data
def preprocess_data(data):
    """Preprocess data: handle date conversion, extract year/month, and drop missing critical values."""
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    # Drop rows with missing values in critical columns
    data = data.dropna(subset=['Latitude', 'Longitude', 'Arrest'])

    return data

# Function to generate crime hotspot map
def generate_hotspot_map(data, primary_type=None, year=None, month=None):
    """Generate a Folium map with crime hotspots based on filters."""
    # Filter data based on user input
    if primary_type:
        data = data[data['Primary Type'] == primary_type]
    if year:
        data = data[data['Year'] == year]
    if month:
        data = data[data['Month'] == month]

    # Create a Folium map centered around Chicago
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)

    # Add points to the map
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(m)

    # Add a heatmap layer
    heat_data = data[['Latitude', 'Longitude']].dropna().values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)

    return m

# Function to generate temporal analysis insights
def temporal_analysis(data):
    """Generate a temporal summary of crimes over time."""
    data['Day'] = data['Date'].dt.day
    temporal_summary = data.groupby(['Year', 'Month']).size().reset_index(name='Crime Count')
    return temporal_summary

# Function to generate geospatial insights
def geospatial_analysis(data):
    """Generate a geospatial summary of crimes by district and ward."""
    geospatial_summary = data.groupby(['District', 'Ward']).size().reset_index(name='Crime Count')
    return geospatial_summary

# Function to analyze repeat crime locations
def analyze_repeat_locations(data):
    """Analyze locations with repeated criminal activities."""
    repeat_locations = data['Block'].value_counts().reset_index()
    repeat_locations.columns = ['Block', 'Crime Count']
    return repeat_locations[repeat_locations['Crime Count'] > 1]

# Function to calculate community engagement metrics
def community_engagement_metrics(data):
    """Calculate metrics for public safety improvement."""
    arrest_rate = data['Arrest'].mean() * 100
    high_crime_areas = data['Community Area'].value_counts().reset_index().head(5)
    high_crime_areas.columns = ['Community Area', 'Crime Count']

    metrics = {
        "Arrest Rate (%)": arrest_rate,
        "Top 5 High-Crime Areas": high_crime_areas
    }
    return metrics

# Main Streamlit app
def main():
    # File paths
    data_file_path = r'D:\GUVI_Projects\My_Projects\cleaned_crime.csv'
    model_file_path = r'D:\GUVI_Projects\My_Projects\random_forest_model.pkl'

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Predictive Analysis", "Temporal Analysis", "Crime Hotspot and Geospatial Analysis", "Insights"])

    if page == "Home":
        st.title("Chicago Crime Analysis Project")
        st.write(""" 
        **Objective:**
        We are observing that there has been a significant increase in crime in Chicago in recent days. Therefore, we are transferring you to Chicago as a Senior Investigation Officer under special deputation.

        Your primary objective in this role is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago. By conducting a thorough analysis of this data, you will support strategic decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety.

        **Creator:** Shubhangi Patil

        **Project:** Data Science

        **GitHub Link:** [GitHub Repository](https://github.com/shubhangivspatil)
        """)
        st.write("""---""")
        st.write("**Created by:** Shubhangi Patil")

    elif page == "Predictive Analysis":
        st.title("Predictive Analysis: Arrest Prediction")

        # Load and preprocess data
        st.subheader("Dataset Loading")
        st.write("Loading data...")
        data = load_data(data_file_path)
        data = preprocess_data(data)

        st.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

        # Display sample data
        if st.checkbox("Show Sample Data"):
            st.write(data.head())

        # Load pre-trained model
        if os.path.exists(model_file_path):
            st.write("Loading pre-trained model...")
            with open(model_file_path, 'rb') as file:
                model = pickle.load(file)

            # Allow user input for predictions
            st.subheader("Make a Prediction")
            latitude = st.number_input("Latitude", value=data['Latitude'].mean())
            longitude = st.number_input("Longitude", value=data['Longitude'].mean())
            year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
            month = st.number_input("Month", min_value=1, max_value=12, value=1)
            id_value = st.text_input("ID")
            block = st.text_input("Block")
            primary_type = st.selectbox("Primary Type", options=data['Primary Type'].unique())
            location_description = st.selectbox("Location Description", options=data['Location Description'].unique())
            description = st.selectbox("Description", options=data['Description'].unique())

            # Prepare input data
            input_df = pd.DataFrame({
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Year': [year],
                'Month': [month],
                'ID': [id_value],
                'Block': [block],
                'Primary Type': [primary_type],
                'Location Description': [location_description],
                'Description': [description]
            })

            # Align input features with model
            for feature in model.feature_names_in_:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Add missing features with default value

            # Remove extra features not used in training
            input_df = input_df[model.feature_names_in_]

            if st.button("Predict Arrest"):
                prediction = model.predict(input_df)
                result = "Arrest Likely" if prediction[0] == 1 else "No Arrest"
                st.write(f"Prediction: {result}")
        else:
            st.write("Pre-trained model not found. Please train a model in the Retrain Model page.")

    elif page == "Temporal Analysis":
        st.title("Temporal Analysis")

        # Load and preprocess data
        st.subheader("Dataset Loading")
        st.write("Loading data...")
        data = load_data(data_file_path)
        data = preprocess_data(data)

        st.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

        # Temporal analysis
        st.subheader("Crime Trends Over Time")
        temporal_summary = temporal_analysis(data)
        st.write(temporal_summary)

        # Trend of arrests over the years
        st.subheader("Trend of Arrests Over the Years")
        arrests_trend = data[data['Arrest'] == True]['Year'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        plt.plot(arrests_trend.index.astype(str), arrests_trend.values, marker='o', linestyle='-', color='red')
        plt.title("Trend of Arrests Over the Years", fontsize=16)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Arrests", fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif page == "Crime Hotspot and Geospatial Analysis":
        st.title("Crime Hotspot and Geospatial Analysis")

        # Load and preprocess data
        st.subheader("Dataset Loading")
        st.write("Loading data...")
        data = load_data(data_file_path)
        data = preprocess_data(data)

        st.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

        # Filters
        st.subheader("Filter Crime Data")
        primary_type = st.selectbox("Select Primary Type", options=[None] + list(data['Primary Type'].unique()))
        year = st.selectbox("Select Year", options=[None] + sorted(data['Year'].unique()))
        month = st.selectbox("Select Month", options=[None] + list(range(1, 13)))

        # Generate map
        st.subheader("Crime Geospatial Map")
        geo_map = generate_hotspot_map(data, primary_type, year, month)
        folium_static(geo_map)

    elif page == "Insights":
        st.title("Insights")

        st.subheader("Key Findings")
        st.write("""
        - **Temporal Trends**: Crime rates show seasonal patterns with spikes in specific months.
        - **Hotspot Identification**: High crime density is observed in downtown areas and specific wards.
        - **Arrest Effectiveness**: Analysis indicates higher arrest rates in certain districts.
        """)

        st.subheader("Repeat Crime Locations")
        data = load_data(data_file_path)
        data = preprocess_data(data)
        repeat_locations = analyze_repeat_locations(data)
        st.write("Top Locations with Repeated Criminal Activity:")
        st.write(repeat_locations)

        st.subheader("Community Engagement Metrics")
        metrics = community_engagement_metrics(data)
        st.write(f"Arrest Rate: {metrics['Arrest Rate (%)']:.2f}%")
        st.write("Top 5 High-Crime Community Areas:")
        st.write(metrics['Top 5 High-Crime Areas'])

        st.subheader("Recommendations")
        st.write("""
        - Allocate resources to identified hotspots during peak crime hours.
        - Focus on preventive measures in high-crime wards.
        - Enhance patrol frequency in districts with low arrest rates.
        - Foster stronger community relations to gather intelligence.
        - Invest in youth programs to address root causes of crime.
        """)

        st.subheader("Impact of Crime on Community Safety")
        st.write(""" 
        Crime significantly affects community safety, leading to increased fear among residents, reduced property values, and potential long-term socioeconomic impacts. Understanding crime trends is crucial for effective intervention strategies.
        """)

        st.subheader("Project Goals")
        st.write("""
        - Provide actionable insights to stakeholders.
        - Enhance data-driven decision-making for law enforcement.
        - Develop predictive models to improve crime prevention strategies.
        """)

if __name__ == "__main__":
    main()

