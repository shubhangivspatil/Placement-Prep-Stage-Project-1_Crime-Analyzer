import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time
from streamlit_folium import st_folium

# Optimized function to load and preprocess data
@st.cache_data
def load_data():
    data_path = 'D:\\GUVI_Projects\\My_Projects\\Cleaned_Chicago_Crime_Data.csv'
    # Load only necessary columns to reduce memory usage
    usecols = ['Date', 'Primary Type', 'Arrest', 'Latitude', 'Longitude']
    data = pd.read_csv(data_path, usecols=usecols)
    
    # Efficient datetime conversion and feature extraction
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year.astype('category')
    data['Month'] = data['Date'].dt.month.astype('category')
    
    # Optimize data types
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce', downcast='float')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce', downcast='float')
    
    return data

# Function to create a HeatMap using folium
def setup_map(data):
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
    heat_data = data[['Latitude', 'Longitude']].dropna().values.tolist()
    HeatMap(heat_data).add_to(m)
    return m

# Function for training different models with optimization
@st.cache_data
def train_model(X_train, y_train, model_type):
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all available cores
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)  # Use all available cores
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    # Load the data at the start of the main function
    data = load_data()

    # Front Page Objective and Problem Statement
    st.title("Chicago Crime Analysis Project")
    st.write(""" 
    **Objective:**
    We are observing that there has been a significant increase in crime in Chicago in recent days. Therefore, we are transferring you to Chicago as a Senior Investigation Officer under special deputation.

    Your primary objective in this role is to leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago. By conducting a thorough analysis of this data, you will support strategic decision-making, improve resource allocation, and contribute to reducing crime rates and enhancing public safety. Your task is to provide actionable insights that can shape our crime prevention strategies, ensuring a safer and more secure community. This project will be instrumental in aiding law enforcement operations and enhancing the overall effectiveness of our efforts in combating crime in Chicago.
    
    **Creator:**
    Shubhangi Patil

    **Project:**
    Data Science

    **GitHub Link:**
    [GitHub Repository](https://github.com/shubhangivspatil)
    """)

    # Sidebar Navigation
    st.sidebar.title("Chicago Crime Data Analysis")
    page = st.sidebar.radio("Select a Page", ["Crime Hotspots", "Predictive Crime Modeling", "Detailed Insights"])

    if page == "Crime Hotspots":
        st.title('Crime Hotspots in Chicago')
        selected_year = st.sidebar.selectbox('Year', options=sorted(data['Year'].unique(), reverse=True))
        filtered_data = data[data['Year'] == selected_year]

        st.subheader('Crime Hotspots Map')
        if st.button('Show Crime Hotspots Map'):
            m = setup_map(filtered_data)
            st_folium(m, width=725, height=500)

        st.subheader('Analysis by Crime Type')
        crime_type_counts = filtered_data['Primary Type'].value_counts().nlargest(10)
        st.bar_chart(crime_type_counts)

    elif page == "Predictive Crime Modeling":
        st.title('Predictive Crime Modeling')
        selected_year = st.sidebar.selectbox('Year for Prediction', options=sorted(data['Year'].unique(), reverse=True))
        filtered_data = data[data['Year'] == selected_year]

        st.subheader('Choose a Model for Crime Prediction')
        model_type = st.selectbox('Model Type', ['Random Forest', 'XGBoost'])

        # Convert Month to numerical values for modeling
        filtered_data['Month'] = filtered_data['Month'].cat.codes  # Convert categories to numerical codes

        X = filtered_data[['Month']]
        y = filtered_data['Arrest'].astype(int)

        unique_values = y.unique()
        if len(unique_values) > 2:
            st.error(f"Unexpected target values: {unique_values}")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        start_time = time.time()
        model = train_model(X_train, y_train, model_type)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        end_time = time.time()

        st.write(f'Model Accuracy ({model_type}): {accuracy:.2f}')
        st.write(f'Training Time: {end_time - start_time:.2f} seconds')

    elif page == "Detailed Insights":
        st.title('Detailed Crime Reports and Insights')
        st.subheader('Recommendations')
        st.text("""
        - Enhance Surveillance in high-risk areas.
        - Align law enforcement staffing with crime trends for better resource allocation.
        - Foster stronger community policing efforts.
        - Use data-driven approaches to optimize patrol routes.
        - Increase public awareness and engagement.
        """)

        st.subheader('Conclusion')
        st.write("""
        Implementing these recommendations can lead to significant reductions in crime rates and enhance public safety, ensuring a more secure environment for the community.
        """)

    if st.sidebar.button('Save Processed Data'):
        filtered_data.to_csv('Processed_Crime_Data.csv', index=False)
        st.success('Data saved successfully!')

if __name__ == '__main__':
    main()
