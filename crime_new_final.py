# import streamlit as st
# import pandas as pd
# import os
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Load dataset function
# @st.cache_data
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     return data

# # Function to preprocess the data
# def preprocess_data(data):
#     data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
#     data['Year'] = data['Date'].dt.year
#     data['Month'] = data['Date'].dt.month

#     # Drop rows with missing values in critical columns
#     data = data.dropna(subset=['Latitude', 'Longitude', 'Arrest'])

#     return data

# # Main Streamlit app
# def main():
#     # File paths
#     data_file_path = r'D:\GUVI_Projects\My_Projects\cleaned_crime.csv'
#     mappings_file_path = r'D:\GUVI_Projects\My_Projects\cleaned_mappings_crime.csv'
#     model_file_path = r'D:\GUVI_Projects\My_Projects\random_forest_model.pkl'
#     save_path = r'D:\GUVI_Projects\My_Projects'

#     # Sidebar navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to:", ["Home", "Predictive Analysis"])

#     if page == "Home":
#         st.title("Chicago Crime Analysis Project")
#         st.write(""" 
#         **Objective:**
#         We are observing that there has been a significant increase in crime in Chicago in recent days. Therefore, we are transferring you to Chicago as a Senior Investigation Officer under special deputation.

#         Your primary objective in this role is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago. By conducting a thorough analysis of this data, you will support strategic decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety. Your task is to provide actionable insights that can shape our crime prevention strategies, ensuring a safer and more secure community. This project will be instrumental in aiding law enforcement operations and enhancing the overall effectiveness of our efforts in combating crime in Chicago.
        
#         **Creator:**
#         Shubhangi Patil

#         **Project:**
#         Data Science

#         **GitHub Link:**
#         [GitHub Repository](https://github.com/shubhangivspatil)
#         """)
        
#         st.write("""---""")
#         st.write("**Created by:** Arun")

#     elif page == "Predictive Analysis":
#         st.title("Predictive Analysis: Arrest Prediction")

#         # Load and preprocess data
#         st.subheader("Dataset Loading")
#         st.write("Loading data...")
#         data = load_data(data_file_path)
#         data = preprocess_data(data)

#         st.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

#         # Display sample data
#         if st.checkbox("Show Sample Data"):
#             st.write(data.head())

#         # Load pre-trained model
#         if os.path.exists(model_file_path):
#             st.write("Loading pre-trained model...")
#             with open(model_file_path, 'rb') as file:
#                 model = pickle.load(file)

#             # Allow user input for predictions
#             st.subheader("Make a Prediction")
#             latitude = st.number_input("Latitude", value=data['Latitude'].mean())
#             longitude = st.number_input("Longitude", value=data['Longitude'].mean())
#             year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
#             month = st.number_input("Month", min_value=1, max_value=12, value=1)
#             id_value = st.text_input("ID")
#             block = st.text_input("Block")
#             primary_type = st.selectbox("Primary Type", options=data['Primary Type'].unique())
#             location_description = st.selectbox("Location Description", options=data['Location Description'].unique())

#             # Prepare input data
#             input_df = pd.DataFrame({
#                 'Latitude': [latitude],
#                 'Longitude': [longitude],
#                 'Year': [year],
#                 'Month': [month],
#                 'ID': [id_value],
#                 'Block': [block],
#                 'Primary Type': [primary_type],
#                 'Location Description': [location_description]
#             })

#             # Align input features with model
#             for feature in model.feature_names_in_:
#                 if feature not in input_df.columns:
#                     input_df[feature] = 0  # Add missing features with default value

#             # Remove extra features not used in training
#             input_df = input_df[model.feature_names_in_]

#             if st.button("Predict Arrest"):
#                 prediction = model.predict(input_df)
#                 result = "Arrest Likely" if prediction[0] == 1 else "No Arrest"
#                 st.write(f"Prediction: {result}")
#         else:
#             st.write("Pre-trained model not found. Please train a model in the Retrain Model page.")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import logging
import folium
from streamlit_folium import folium_static

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset function
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to preprocess the data
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month

    # Drop rows with missing values in critical columns
    data = data.dropna(subset=['Latitude', 'Longitude', 'Arrest'])

    return data

# Function to generate crime hotspot map
def generate_hotspot_map(data, primary_type=None, year=None, month=None):
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

    return m

# Function to generate temporal analysis insights
def temporal_analysis(data):
    data['Day'] = data['Date'].dt.day
    temporal_summary = data.groupby(['Year', 'Month']).size().reset_index(name='Crime Count')
    return temporal_summary

# Function to generate geospatial insights
def geospatial_analysis(data):
    geospatial_summary = data.groupby(['District', 'Ward']).size().reset_index(name='Crime Count')
    return geospatial_summary

# Main Streamlit app
def main():
    # File paths
    data_file_path = r'D:\GUVI_Projects\My_Projects\cleaned_crime.csv'
    model_file_path = r'D:\GUVI_Projects\My_Projects\random_forest_model.pkl'

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Predictive Analysis", "Temporal Analysis", "Geospatial Analysis", "Crime Hotspot Analysis", "Stakeholder Insights"])

    if page == "Home":
        st.title("Chicago Crime Analysis Project")
        st.write(""" 
        **Objective:**
        We are observing that there has been a significant increase in crime in Chicago in recent days. Therefore, we are transferring you to Chicago as a Senior Investigation Officer under special deputation.

        Your primary objective in this role is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago. By conducting a thorough analysis of this data, you will support strategic decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety. This project will be instrumental in aiding law enforcement operations and enhancing the overall effectiveness of our efforts in combating crime in Chicago.
        
        **Creator:**
        Shubhangi Patil

        **Project:**
        Data Science

        **GitHub Link:**
        [GitHub Repository](https://github.com/shubhangivspatil)
        """)
        
        st.write("""---""")
        st.write("**Created by:** Arun")

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

            # Prepare input data
            input_df = pd.DataFrame({
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Year': [year],
                'Month': [month],
                'ID': [id_value],
                'Block': [block],
                'Primary Type': [primary_type],
                'Location Description': [location_description]
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

    elif page == "Geospatial Analysis":
        st.title("Geospatial Analysis")

        # Load and preprocess data
        st.subheader("Dataset Loading")
        st.write("Loading data...")
        data = load_data(data_file_path)
        data = preprocess_data(data)

        st.write(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

        # Geospatial analysis
        st.subheader("District and Ward Analysis")
        geospatial_summary = geospatial_analysis(data)
        st.write(geospatial_summary)

    elif page == "Crime Hotspot Analysis":
        st.title("Crime Hotspot Analysis")

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
        st.subheader("Crime Hotspot Map")
        hotspot_map = generate_hotspot_map(data, primary_type, year, month)
        folium_static(hotspot_map)

    elif page == "Stakeholder Insights":
        st.title("Stakeholder Insights")

        st.subheader("Key Findings")
        st.write("""
        - **Temporal Trends**: Crime rates show seasonal patterns with spikes in specific months.
        - **Hotspot Identification**: High crime density is observed in downtown areas and specific wards.
        - **Arrest Effectiveness**: Analysis indicates higher arrest rates in certain districts.
        """)

        st.subheader("Recommendations")
        st.write("""
        - Allocate resources to identified hotspots during peak crime hours.
        - Focus on preventive measures in high-crime wards.
        - Enhance patrol frequency in districts with low arrest rates.
        """)

        st.subheader("Project Goals")
        st.write("""
        - Provide actionable insights to stakeholders.
        - Enhance data-driven decision-making for law enforcement.
        - Develop predictive models to improve crime prevention strategies.
        """)

if __name__ == "__main__":
    main()
