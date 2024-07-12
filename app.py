import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the RandomForest model and the scaler
rf_model = load('rf_model.joblib')
scaler = load('scaler.joblib')

rf_model_tod = load('tod_rf_model.joblib')
scaler_tod = load('tod_scaler.joblib')

# Function to load data with caching
@st.cache_data  # ensure the data is loaded only once
def load_data():
    return pd.read_csv('data2.csv')
def load_district():
    # Load the CSV data
    return pd.read_csv('districtneighborhoods.csv')
def load_tod():
    return pd.read_csv('tod.csv')
def load_main_dataset():
    df = pd.read_csv('main_dataset.csv')
    df['received_datetime'] = pd.to_datetime(df['received_datetime'])  # Ensure this column is datetime
    df['onscene_datetime'] = pd.to_datetime(df['onscene_datetime'])  # Convert this column as well

    return df

data = load_data()
district_data = load_district()
tod_data = load_tod()
main_data = load_main_dataset()



def police_allocation(num_police_vehicles, hour_of_day, district_for_allocation, model, scaler):
    # Grabbing requested district:
    data_subset = main_data.loc[main_data['police_district'].isin([district_for_allocation])].copy()
    # Grabbing relevant hour range:
    data_subset['hour'] = main_data['received_datetime'].dt.hour
    if 0 <= hour_of_day < 8:
        filtered_data_subset = data_subset[(data_subset['hour'] >= 0) & (data_subset['hour'] < 8)]
    elif 8 <= hour_of_day < 16:
        filtered_data_subset = data_subset[(data_subset['hour'] >= 8) & (data_subset['hour'] < 16)]
    else:
        filtered_data_subset = data_subset[(data_subset['hour'] >= 16) & (data_subset['hour'] <= 23 )]

    # Calculating the frequency of calls for service for each neighborhood in the district:
    neighborhood_frequency = filtered_data_subset['analysis_neighborhood'].value_counts()
    # Summing the frequencies for ratio calculation:
    sum_frequencies = neighborhood_frequency.sum()
    # Creating and populating a dictionary with the ratios:
    neighborhood_ratios = {}
    for neighborhood, count in neighborhood_frequency.items():
        ratio = count / sum_frequencies
        neighborhood_ratios[neighborhood] = ratio

    # Creating a response time columns to use for composite score calculation:
    filtered_data_subset['onscene_datetime'] = pd.to_datetime(filtered_data_subset['onscene_datetime'], errors='coerce')
    filtered_data_subset['response_time_minutes'] = (filtered_data_subset['onscene_datetime'] - filtered_data_subset['received_datetime']).dt.total_seconds() / 60
    # Group by district and neighborhood, then aggregate
    neighborhood_summary = filtered_data_subset.groupby(['police_district', 'analysis_neighborhood']).agg({
    'response_time_minutes': 'mean',  # Average response time in minutes
    'cad_number': 'count'  # Count of incidents as incident count
    }).reset_index()

    # Rename columns for clarity
    neighborhood_summary.rename(columns={
    'response_time_minutes': 'average_response_time',
    'cad_number': 'incident_count'
    }, inplace=True)

    # Normalization for clarity:
    scaler = StandardScaler()
    neighborhood_summary['normalized_incidents'] = scaler.fit_transform(neighborhood_summary[['incident_count']])
    neighborhood_summary['normalized_response'] = scaler.fit_transform(neighborhood_summary[['average_response_time']])

    # Example of weighting: Assign a higher weight to response time if it's considered more critical
    weight_for_incidents = 0.5
    weight_for_response_time = 0.5

    # Normalize both incident count and response time
    max_incidents = neighborhood_summary['incident_count'].max()
    max_response_time = neighborhood_summary['average_response_time'].max()

    neighborhood_summary['normalized_incidents'] = neighborhood_summary['incident_count'] / max_incidents
    neighborhood_summary['normalized_response_time'] = neighborhood_summary['average_response_time'] / max_response_time

    # Calculate a composite score
    neighborhood_summary['composite_score'] = (weight_for_incidents * neighborhood_summary['normalized_incidents']) \
                                            + (weight_for_response_time * (1 - neighborhood_summary['normalized_response_time']))
    highest_composite_score = neighborhood_summary['composite_score'].max()
    # Using composite score to balance the police allocation ratio so that response time can be more stabilized through the neighborhoods:
    neighborhood_summary['police_allocation_ratio'] = (neighborhood_summary['incident_count']/sum_frequencies) + ((highest_composite_score - neighborhood_summary['composite_score'])/10)
    total = neighborhood_summary['police_allocation_ratio'].sum() - 1

    # Assigning vehicles:
    max_index = neighborhood_summary['police_allocation_ratio'].idxmax()
    neighborhood_summary.loc[max_index, 'police_allocation_ratio'] -= total
    neighborhood_summary['vehicle_assignment'] = np.floor(neighborhood_summary['police_allocation_ratio']*num_police_vehicles)
    return neighborhood_summary[['analysis_neighborhood', 'incident_count', 'average_response_time', 'vehicle_assignment']]












# Calculate historical averages
@st.cache_data
def get_historical_averages(data,timeframe):
    return data.groupby([timeframe, 'cluster'])['incident_count'].mean().reset_index()

historical_averages = get_historical_averages(data,'month')
historical_averages_tod = get_historical_averages(tod_data,'time_of_day')
st.title("Incident Prediction and Police Allocation")

# Input fields for month and cluster which could influence incident rates
month_value = st.slider("Select the month:", min_value=1, max_value=12, step=1)
#cluster_value = st.slider("Select the cluster (based on neighborhood characteristics):", min_value=0, max_value=4, step=1)
tod_value = st.slider("Select the hour of the day:", min_value=0, max_value=23, step=1)



num_police_vehicles = st.slider("Select number of police vehicles available:", min_value = 1, max_value = 50, step = 2)



districts = district_data['police_district'].unique()
district = st.selectbox("Select a district:", districts)

if district:
    neighborhoods = district_data[district_data['police_district'] == district]['analysis_neighborhood'].unique()
    neighborhood = st.selectbox("Select a neighborhood:", neighborhoods)



if st.button('Predict Incident Counts for Month'):
    # Filter the historical averages based on the selected month and cluster
    if district and 'neighborhood' in locals():
        cluster_value = data[data['analysis_neighborhood'] == neighborhood]['cluster'].iloc[0]
        st.write(f"Selected Cluster: {cluster_value}")

    filtered_data = historical_averages[
        (historical_averages['month'] == f'2023-{month_value:02}') & 
        (historical_averages['cluster'] == cluster_value)
    ]
    

    if not filtered_data.empty:
        # Extract the first (and should be only) incident count value
        historical_incident = filtered_data['incident_count'].iloc[0]

        # Normalize the historical incident counts
        normalized_incidents = scaler.transform([[historical_incident]])[0][0]

        # Prepare the data for prediction
        input_data = pd.DataFrame({
            'month_encoded': [month_value],
            'cluster': [cluster_value],
            'normalized_incidents': [normalized_incidents]
        })

        # Predict using the RandomForest model
        prediction = rf_model.predict(input_data)[0]
        st.write(f"Predicted number of incidents: {prediction}")
    else:
        st.error("No historical data available for the selected month and cluster.")


if st.button('Predict Incident Counts for Hour'):
    # Filter the historical averages based on the selected time of day and cluster
    if district and 'neighborhood' in locals():
        cluster_value = tod_data[tod_data['analysis_neighborhood'] == neighborhood]['cluster'].iloc[0]
        st.write(f"Selected Cluster: {cluster_value}")
    filtered_data_tod = historical_averages_tod[
        (historical_averages_tod['time_of_day'] == tod_value) & 
        (historical_averages_tod['cluster'] == cluster_value)
    ]

    if not filtered_data_tod.empty:
        # Extract the first (and should be only) incident count value
        historical_incident_tod = filtered_data_tod['incident_count'].iloc[0]

        # Normalize the historical incident counts
        normalized_incidents_tod = scaler_tod.transform([[historical_incident_tod]])[0][0]

        # Prepare the data for prediction
        input_data_tod = pd.DataFrame({
            'time_of_day': [tod_value],
            'cluster': [cluster_value],
            'normalized_incidents': [normalized_incidents_tod]
        })

        # Predict using the RandomForest model
        prediction_tod = rf_model_tod.predict(input_data_tod)[0]
        st.write(f"Predicted number of incidents: {prediction_tod}")
    else:
        st.error("No historical data available for the selected time of day and cluster.")


if st.button('Allocate Police Force'):
   allocation = police_allocation(num_police_vehicles, tod_value, district, rf_model_tod, scaler_tod)
   st.write(allocation) 
