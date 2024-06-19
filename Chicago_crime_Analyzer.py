import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to load and preprocess data
@st.cache_data
def load_data():
    data_path = 'D:\\GUVI_Projects\\My_Projects\\Cleaned_Chicago_Crime_Data.csv'
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
    return data

# Function to create a HeatMap using folium
def setup_map(data):
    # Initialize the map centered around Chicago
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
    # Generate heat data from the DataFrame
    heat_data = data[['Latitude', 'Longitude']].dropna().values.tolist()
    # Create and add heat map layer to the map
    HeatMap(heat_data).add_to(m)
    return m

def main():
    data = load_data()
    
    st.sidebar.header('User Input Features')
    selected_year = st.sidebar.selectbox('Year', options=sorted(data['Year'].unique(), reverse=True))
    filtered_data = data[data['Year'] == selected_year]

    st.title('Chicago Crime Data Analysis Dashboard')

    # Crime Hotspots Map and Analysis by Crime Type
    st.header('Crime Hotspots and Analysis by Crime Type')

    # Use columns to display map and crime type analysis side by side
    col1, col2 = st.columns([2, 3])

    # Crime Hotspots Map
    with col1:
        st.subheader('Crime Hotspots Map')
        if st.button('Show Crime Hotspots Map'):
            from streamlit_folium import st_folium  # Import here to avoid runtime errors
            m = setup_map(filtered_data)
            st_folium(m, width=725, height=500)  # Specify height to ensure it displays correctly

    # Analysis by Crime Type
    with col2:
        st.subheader('Analysis by Crime Type')
        crime_type_counts = filtered_data['Primary Type'].value_counts().nlargest(10)
        st.bar_chart(crime_type_counts)

    # Predictive Crime Modeling
    st.header('Predictive Crime Modeling')
    X = filtered_data[['Month']]
    y = filtered_data['Arrest'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.write('Model Accuracy:', model.score(X_test, y_test))

    # Detailed Crime Reports and Insights
    st.header('Detailed Crime Reports and Insights')
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
