#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 2024

@author: lefteris
"""

import streamlit as st
import requests
import folium
import streamlit.components.v1 as components
from app.map_integration import MapIntegration
from app.database import Database
from app.utils import haversine_distance as calculate_distance, SEVERITY_LEVELS
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Set Flask API URL (update with your Flask app's URL or localhost)
BASE_URL = "http://127.0.0.1:5000"  # Make sure Flask is running locally or remotely


st.set_page_config(layout="wide")

# Function to calculate the nearest base for each user
def find_nearest_base(user_df, base_df):
    nearest_bases = []
    
    # Iterate through each user
    for i, user_row in user_df.iterrows():
        user_lat = user_row['coordinates_lat']
        user_lon = user_row['coordinates_long']
        
        min_distance = float('inf')
        nearest_base = None
        
        # Iterate through each base to find the nearest
        for j, base_row in base_df.iterrows():
            base_lat = base_row['coordinates_lat']
            base_lon = base_row['coordinates_long']
            
            # Calculate distance between user and base
            distance = calculate_distance(user_lat, user_lon, base_lat, base_lon)
            
            # Check if this is the closest base so far
            if distance < min_distance:
                min_distance = distance
                nearest_base = base_row['base_type']
                base_id = base_row['id']
        
        # Append the nearest base and distance to the result list
        nearest_bases.append({'user_id': user_row['id'], 'nearest_base': nearest_base,'base_id' : base_id,
                              'distance_km': min_distance})
    
    # Return as a DataFrame
    return pd.DataFrame(nearest_bases)

def calculate_response_time(distance, base_type):
    """
    Calculate estimated response time based on distance and base type.

    Average speeds vary by facility type based on vehicle characteristics:
    - Fire trucks: slower, heavy vehicles
    - EMS units: average speed
    - Police vehicles: faster, lighter vehicles

    Args:
        distance: Distance in kilometers
        base_type: Type of emergency facility

    Returns:
        Estimated response time in minutes
    """
    # Average speed in km/min for different facility types
    # Based on typical urban emergency response speeds (~50-70 km/h)
    speeds = {
        'Fire Station': 0.8,        # ~48 km/h (slower, heavy vehicles)
        'EMS Station': 1.0,          # ~60 km/h (ambulances)
        'Police Station': 1.2,       # ~72 km/h (faster, lighter vehicles)
        'Base of Operations': 0.9,   # Generic facility (legacy)
        'Staging Area': 0.9,         # Generic facility (legacy)
        'Incident Command Post': 1.0 # Generic facility (legacy)
    }

    speed = speeds.get(base_type, 1.0)  # Default to 60 km/h
    travel_time = distance / speed
    prep_time = 3  # Minutes to prepare and deploy

    return round(travel_time + prep_time, 2)

def calc_base(formatted_data, formatted_data_op):
    
    # Convert the list of dictionaries into Pandas DataFrames
    user_df = pd.DataFrame(formatted_data)
    base_df = pd.DataFrame(formatted_data_op)
    
    # Find the nearest bases for each user
    proximity_df = find_nearest_base(user_df, base_df)
    proximity_df['distance_km_rounded'] = proximity_df['distance_km'].round(0)
    incident_count_by_km = proximity_df.groupby(['distance_km_rounded']).size().reset_index(name='count')
    merged_df = pd.merge(user_df, proximity_df, left_on='id', right_on="user_id")
    merged_df['response_time'] = merged_df.apply(lambda row: calculate_response_time(row['distance_km'], row['nearest_base']), axis=1)

    return merged_df, incident_count_by_km, user_df, base_df, proximity_df
    


SEVERITY_OPTIONS = SEVERITY_LEVELS  # Use shared constant


def run_optimization():
    """Call the optimize endpoint and store results in session state."""
    try:
        response = requests.post(f"{BASE_URL}/optimize", timeout=30)
        if response.status_code == 200:
            result = response.json()
            # Handle new response format with metadata
            if 'assignments' in result:
                st.session_state['optimization_results'] = result['assignments']
                st.session_state['optimization_metadata'] = {
                    'solver_status': result.get('solver_status', 'Unknown'),
                    'is_optimal': result.get('is_optimal', False),
                    'total_capacity': result.get('total_capacity', 0),
                    'num_emergencies': result.get('num_emergencies', 0),
                    'violations': result.get('violations', []),
                    'triage_applied': result.get('triage_applied', False),
                    'queued': result.get('queued', []),
                    'assigned_count': result.get('assigned_count', 0),
                    'queued_count': result.get('queued_count', 0)
                }
            else:
                # Backwards compatibility with old format
                st.session_state['optimization_results'] = result
                st.session_state['optimization_metadata'] = {}
            st.session_state['optimization_run'] = True
            return True
        elif response.status_code == 400:
            error_data = response.json()
            if 'error' in error_data:
                st.error(f"Optimization failed: {error_data.get('message', error_data['error'])}")
            else:
                st.error(f"Optimization failed. Server returned: {response.status_code}")
            return False
        else:
            st.error(f"Optimization failed. Server returned: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Flask backend.")
        return False
    except requests.exceptions.Timeout:
        st.error("Optimization timed out.")
        return False


def create_filters(merged_df, base_df):

    # Sidebar Filters
    st.sidebar.header('Filters')

    # Get base IDs and types from actual base data
    all_base_ids = np.sort(base_df['id'].unique())
    all_base_ids = [str(x) for x in all_base_ids]
    all_base_types = np.sort(base_df['base_type'].unique())

    min_distance = merged_df.distance_km.min()
    max_distance = merged_df.distance_km.max()
    min_response_time = merged_df.response_time.min()
    max_response_time = merged_df.response_time.max()

    selected_bases = st.sidebar.multiselect('Filter by Base ID', ['ALL'] + all_base_ids, default='ALL')
    selected_base_types = st.sidebar.multiselect('Filter by Base Type', ['ALL'] + list(all_base_types), default='ALL')

    selected_distance = st.sidebar.slider('Filter Distance From Base', min_value=np.floor(min_distance), max_value=np.ceil(max_distance),
                                        value=(np.floor(min_distance), np.ceil(max_distance)))
    selected_response_time = st.sidebar.slider('Filter Response Time From Base', min_value=np.floor(min_response_time), max_value=np.ceil(max_response_time),
                                        value=(np.floor(min_response_time), np.ceil(max_response_time)))

    # Optimization section
    st.sidebar.header('Optimization')
    if st.sidebar.button('Run Optimization'):
        with st.spinner('Running optimization...'):
            run_optimization()

    # Filter by optimized assignment
    selected_optimized_base = None
    if st.session_state.get('optimization_run', False):
        opt_results = st.session_state.get('optimization_results', [])
        if opt_results:
            opt_base_ids = sorted(set(str(r['base_id']) for r in opt_results))
            selected_optimized_base = st.sidebar.selectbox(
                'Filter by Assigned Base (Optimized)',
                ['ALL'] + opt_base_ids
            )

    # Filter bases by selected IDs and types
    bs_id_filt = all_base_ids if ('ALL' in selected_bases or len(selected_bases) == 0) else selected_bases
    bs_type_filt = all_base_types if ('ALL' in selected_base_types or len(selected_base_types) == 0) else selected_base_types
    bs_id_filt = [int(x) for x in bs_id_filt]

    # Filter base_df to get the bases to display
    filtered_base_df = base_df[(base_df['id'].isin(bs_id_filt)) &
                               (base_df['base_type'].isin(bs_type_filt))].copy()
    filtered_base_ids = filtered_base_df['id'].tolist()

    # Filter emergencies: show those whose nearest base is in the filtered set
    filtered_df = merged_df[(merged_df['base_id'].isin(filtered_base_ids)) &
                         (merged_df['distance_km'] >= selected_distance[0]) &
                         (merged_df['distance_km'] <= selected_distance[1]) &
                         (merged_df['response_time'] >= selected_response_time[0]) &
                         (merged_df['response_time'] <= selected_response_time[1])].copy()

    # Apply optimized base filter if selected
    if selected_optimized_base and selected_optimized_base != 'ALL':
        opt_results = st.session_state.get('optimization_results', [])
        assigned_em_ids = [r['em_id'] for r in opt_results if str(r['base_id']) == selected_optimized_base]
        filtered_df = filtered_df[filtered_df['user_id'].isin(assigned_em_ids)]
        # Also filter bases to only show the selected optimized base
        filtered_base_ids = [int(selected_optimized_base)]

    return filtered_df, filtered_base_ids

def main():
    st.title("Crisis Management Dashboard")
    # Create a sidebar with navigation options
    page = st.sidebar.radio("Select a page", ["Submit Form", "Insert Base","Map", "Data",
                                              "Analytics"])

    # Display content based on the selected page
    if page == "Submit Form":
        submit_form()
    elif page == "Insert Base":
        insert_base()
    elif page == "Map":
        admin_map()
    elif page == "Data":
        show_data()
    elif page == "Analytics":
        all_plots()

def submit_form():
    # Form to input user data
    with st.form("submit_form"):
        name = st.text_input("Name")
        phone_number = st.text_input("Phone Number")
        message = st.text_input("Message")
        coordinates_lat = st.text_input("Coordinates Latitude")
        coordinates_long = st.text_input("Coordinates Longitude")
        severity = st.selectbox(
            "Severity Level",
            options=[1, 2, 3],
            format_func=lambda x: f"{x} - {SEVERITY_OPTIONS[x]}",
            index=1,  # Default to Urgent
            help="1=Critical (life-threatening), 2=Urgent (serious), 3=Normal (non-urgent)"
        )
        date_input = st.date_input("Select a date")
        time_input = st.time_input("Select a time")
        dt = datetime.combine(date_input, time_input)
        dt_str = dt.isoformat()  # ISO 8601 format (e.g., '2023-09-22T14:30:00')
        # Submit button for form
        submit_button = st.form_submit_button(label="Submit Data")

    # Process form submission
    if submit_button:
        # Create a dictionary with the form data
        data = {
            "name": name,
            "phone_number": phone_number,
            "message": message,
            "coordinates_lat": coordinates_lat,
            "coordinates_long": coordinates_long,
            "severity": severity,
            "dt": dt_str
        }
        
        # Send POST request to the Flask API with the form data
        try:
            response = requests.post(f"{BASE_URL}/submit", json=data, timeout=5)

            # Check the response from Flask
            if response.status_code == 200:
                response_data = response.json()
                st.success("Data submitted successfully!")
                st.json(response_data)  # Display the returned data from Flask
            else:
                st.error(f"Failed to submit data. Server returned: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask backend. Make sure it's running on http://127.0.0.1:5000")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def insert_base():
    # Create form fields for base data input
    st.header("Insert Base Information")
    
    # Form to input base data
    with st.form("insert_base_form"):
        name = st.text_input("Base Name")
        base_type = st.text_input("Base Type (e.g., medical, fire station)")
        coordinates_lat = st.text_input("Coordinates Latitude")
        coordinates_long = st.text_input("Coordinates Longitude")
        capacity = st.text_input("Capacity (e.g., number of people or resources)")
        
        # Submit button for form
        submit_button = st.form_submit_button(label="Insert Base Data")
    
    # Process form submission
    if submit_button:
        # Create a dictionary with the form data
        data = {
            "name": name,
            "base_type": base_type,
            "coordinates_lat": coordinates_lat,
            "coordinates_long": coordinates_long,
            "capacity": capacity
        }
        
        # Send POST request to the Flask API with the form data
        try:
            response = requests.post(f"{BASE_URL}/insertbase", json=data, timeout=5)

            # Check the response from Flask
            if response.status_code == 200:
                response_data = response.json()
                st.success("Base data submitted successfully!")
                st.json(response_data)  # Display the returned data from Flask
            else:
                st.error(f"Failed to submit base data. Server returned: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Flask backend. Make sure it's running on http://127.0.0.1:5000")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")



# Define the function to display the admin map
def admin_map(coord_init=[39.553464, 21.759884]):  # lat, lon
    try:
        # Set the URL to your Flask route (make sure the Flask app is running)
        response = requests.post(f"{BASE_URL}/admin/map", timeout=5)
        # Check if the request was successful
        if response.status_code == 200:
            # Get the iframe HTML from the response
            data = response.json()

            # Extract user data and base data
            formatted_data = data.get('user_data', [])
            formatted_data_op = data.get('base_data', [])

            if not formatted_data or not formatted_data_op:
                st.warning("No data available. Please add emergency calls and bases first.")
                return

            merged_df, _, _, base_df, _ = calc_base(formatted_data, formatted_data_op)
            filtered_df, filtered_base_ids = create_filters(merged_df, base_df)

            usr_ids_filt = filtered_df.user_id.unique()

            formatted_data_op = [record for record in formatted_data_op if record["id"] in filtered_base_ids]
            formatted_data = [record for record in formatted_data if record["id"] in usr_ids_filt]
            # Render the Folium map in the Streamlit app
            st.title("Administrator Map")

            em_map = MapIntegration(coord_init, formatted_data_op)
            em_map.plot_base_data()
            em_map.plot_user_data(formatted_data)
            folium_map = em_map.map  # Generate the map

            iframe = folium_map.get_root()._repr_html_()
            # Use Streamlit's components.html to display the Folium map
            components.html(f"""
            <div style="width: 100%; height: 100%;">
                {iframe}
            </div>
            <style>
                iframe {{
                    width: 100% !important;
                    height: 100% !important;
                }}
            </style>
            """, height=850)

            # Display optimization results table if available
            if st.session_state.get('optimization_run', False):
                opt_results = st.session_state.get('optimization_results', [])
                opt_meta = st.session_state.get('optimization_metadata', {})

                st.subheader("Optimization Results - Base Assignments")

                # Show solver status and metadata
                if opt_meta:
                    solver_status = opt_meta.get('solver_status', 'Unknown')
                    is_optimal = opt_meta.get('is_optimal', False)
                    total_cap = opt_meta.get('total_capacity', 0)
                    num_em = opt_meta.get('num_emergencies', 0)
                    violations = opt_meta.get('violations', [])
                    triage_applied = opt_meta.get('triage_applied', False)

                    status_color = "green" if is_optimal else "orange"
                    st.markdown(f"**Solver Status:** :{status_color}[{solver_status}] | "
                                f"**Total Capacity:** {total_cap} | **Emergencies:** {num_em}")

                    # Show triage warning if applied
                    if triage_applied:
                        assigned_count = opt_meta.get('assigned_count', 0)
                        queued_count = opt_meta.get('queued_count', 0)
                        st.warning(f"**TRIAGE APPLIED:** Capacity ({total_cap}) < Emergencies ({num_em}). "
                                   f"**{assigned_count}** assigned, **{queued_count}** queued by priority.")

                    if violations:
                        st.error(f"Capacity violations detected! {len(violations)} base(s) exceeded capacity:")
                        for v in violations:
                            st.write(f"  - Base {v['base_id']}: assigned {v['assigned']} units, capacity is {v['capacity']}")

                if opt_results:
                    opt_df = pd.DataFrame(opt_results)
                    opt_df = opt_df.rename(columns={
                        'base_id': 'Assigned Base',
                        'em_id': 'Emergency ID',
                        'supply': 'Units',
                        'distance': 'Assigned Dist (km)',
                        'nearest_distance': 'Nearest Dist (km)',
                        'nearest_base_id': 'Nearest Base'
                    })
                    opt_df['Assigned Dist (km)'] = opt_df['Assigned Dist (km)'].round(2)
                    opt_df['Nearest Dist (km)'] = opt_df['Nearest Dist (km)'].round(2)
                    # Flag cases where assigned base is not the nearest
                    opt_df['Not Nearest?'] = opt_df['Assigned Base'] != opt_df['Nearest Base']
                    st.dataframe(opt_df, use_container_width=True)

                    # Show capacity summary per base
                    st.subheader("Base Capacity Usage")
                    base_summary = opt_df.groupby('Assigned Base').agg({
                        'Units': 'sum',
                        'Emergency ID': 'count'
                    }).reset_index()
                    base_summary.columns = ['Base ID', 'Total Units Assigned', 'Emergencies Assigned']
                    st.dataframe(base_summary, use_container_width=True)

                    # Show summary
                    not_nearest_count = opt_df['Not Nearest?'].sum()
                    if not_nearest_count > 0:
                        st.info(f"{not_nearest_count} emergencies assigned to non-nearest base (due to capacity constraints)")

                # Show queued emergencies if triage was applied
                if opt_meta.get('triage_applied', False):
                    queued = opt_meta.get('queued', [])
                    if queued:
                        st.subheader("Queued Emergencies (Awaiting Resources)")
                        st.markdown("*These emergencies are waiting for resources to become available, sorted by priority.*")
                        queued_df = pd.DataFrame(queued)
                        # Format severity
                        if 'severity' in queued_df.columns:
                            queued_df['Severity'] = queued_df['severity'].map(SEVERITY_OPTIONS)
                        display_cols = ['id', 'name', 'phone_number', 'Severity', 'message']
                        display_cols = [c for c in display_cols if c in queued_df.columns]
                        queued_df = queued_df.rename(columns={'id': 'Emergency ID', 'name': 'Name',
                                                              'phone_number': 'Phone', 'message': 'Message'})
                        st.dataframe(queued_df[['Emergency ID', 'Name', 'Phone', 'Severity', 'Message']],
                                     use_container_width=True)
        else:
            st.error(f"Failed to load map from Flask app. Server returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Flask backend. Make sure it's running on http://127.0.0.1:5000")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



# Function to fetch data for the dropdowns
def fetch_data():
    """Fetch data from Flask API with error handling."""
    try:
        user_data_response = requests.post(f'{BASE_URL}/display_data', timeout=5)

        if user_data_response.status_code == 200:
            data = user_data_response.json()
            user_data = data.get('user_data', [])
            base_data = data.get('base_data', [])
            return user_data, base_data
        else:
            st.error(f'Failed to fetch data from the server. Server returned: {user_data_response.status_code}')
            return [], []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Flask backend. Make sure it's running on http://127.0.0.1:5000")
        return [], []
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return [], []
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return [], []

# Function to send a POST request to delete an entry
def delete_entry(data_id, table_name):
    """Delete entry with error handling."""
    delete_url = f'{BASE_URL}/delete'
    data = {
        'id': data_id,
        'table': table_name
    }
    try:
        response = requests.post(delete_url, data=data, timeout=5)

        if response.status_code == 200:
            st.sidebar.success(f"Successfully deleted entry with id {data_id} from {table_name}.")
        else:
            st.sidebar.error(f"Failed to delete the entry. Server returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("Cannot connect to Flask backend. Make sure it's running on http://127.0.0.1:5000")
    except requests.exceptions.Timeout:
        st.sidebar.error("Request timed out. Please try again.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")


def show_data():
    
    # Extract user data and base data
    formatted_data, formatted_data_op = fetch_data()
          
    # Render the Folium map in the Streamlit app
    st.title("Administrator Data")
    
    # Convert the list of dictionaries into Pandas DataFrames
    user_df = pd.DataFrame(formatted_data)
    base_df = pd.DataFrame(formatted_data_op)
    
    # Display the tables in Streamlit
    st.title("User Data Table")
    st.dataframe(user_df)  # or st.table(user_df) for a static table
    
    st.title("Operation Base Data Table")
    st.dataframe(base_df)  # or st.table(base_df) for a static table
    
    
    
    # Create selection UI for the table
    table_choice = st.sidebar.selectbox(
        "Choose the table to delete from:",
        ("Emergency Bases", "Emergency Data")
    )
    
    # Depending on the choice, show the appropriate entries for deletion
    if formatted_data_op and table_choice == "Emergency Bases":
        # Show Emergency Bases data
        base_ids = [entry['id'] for entry in formatted_data_op]
        selected_id = st.sidebar.selectbox("Select the ID to delete:", base_ids)
    elif formatted_data and table_choice == "Emergency Data":
        # Show Emergency Data
        user_ids = [entry['id'] for entry in formatted_data]
        selected_id = st.sidebar.selectbox("Select the ID to delete:", user_ids)
    else:
        st.sidebar.warning("No data available to delete.")
        selected_id = None

    # Add a delete button to submit the delete request
    if st.sidebar.button("Delete Entry") and selected_id is not None:
        delete_entry(selected_id, table_choice)




def all_plots(coord_init=[39.553464, 21.759884]):
    formatted_data, formatted_data_op = fetch_data()
    st.title("Calls")
    
    
    
    merged_df, incident_count_by_km, user_df, base_df, proximity_df = calc_base(formatted_data, formatted_data_op)
    
    
    
    
    
    # 1. Heatmap of Incident Density
    fig1 = px.scatter_mapbox(user_df, lat='coordinates_lat', lon='coordinates_long',
                             mapbox_style="carto-positron",
                             zoom=10, center={"lat": coord_init[0], "lon": coord_init[1]})
    
    
    
    # 4. Incident Count by Emergency Base Proximity
    # Example proximity data (mock example)
    fig4 = px.bar(incident_count_by_km, x='distance_km_rounded',y="count",
                  title="Incident Count by Proximity to Emergency Bases")
    fig4.update_xaxes(title="Proximity to Base (in km rounded)")
    fig4.update_yaxes(title="Incident Count")
    
    
    
    
        

    # 3. Incident Type Over Time
    merged_df['dt'] = pd.to_datetime(merged_df['dt'])
    merged_df['date'] = merged_df['dt'].dt.date
    incident_count_by_date = merged_df.groupby(['date', 'nearest_base']).size().reset_index(name='count')
    fig3 = px.line(incident_count_by_date, x='date', y='count', color='nearest_base',
                   title="Incident Types Over Time")


    # 2. Response Time Distribution
    fig2 = px.histogram(merged_df, x='response_time', nbins=20, title="Distribution of Response Times")
    fig2.update_xaxes(title="Response Time (minutes)")
    fig2.update_yaxes(title="Count")

    # 5. Severity Distribution
    if 'severity' in user_df.columns:
        severity_counts = user_df['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        severity_counts['severity_label'] = severity_counts['severity'].map(SEVERITY_OPTIONS)
        severity_counts = severity_counts.sort_values('severity')

        fig5 = px.pie(severity_counts, values='count', names='severity_label',
                      title="Emergency Severity Distribution",
                      color='severity_label',
                      color_discrete_map={
                          'Critical': '#e74c3c',
                          'Urgent': '#f39c12',
                          'Normal': '#27ae60'
                      })
        fig5.update_traces(textposition='inside', textinfo='percent+label')
    else:
        fig5 = None

    col1, col2 = st.columns(2)

    with col1:
        st.header("Heatmap of Incident Density")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.header("Response Time Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.header("Incident Type Over Time")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.header("Incident Count by Emergency Base Proximity")
        st.plotly_chart(fig4, use_container_width=True)

    # Third row for severity
    if fig5:
        col5, col6 = st.columns(2)
        with col5:
            st.header("Severity Distribution")
            st.plotly_chart(fig5, use_container_width=True)
        with col6:
            # Severity summary stats
            st.header("Severity Summary")
            if 'severity' in user_df.columns:
                for sev, label in SEVERITY_OPTIONS.items():
                    count = len(user_df[user_df['severity'] == sev])
                    pct = (count / len(user_df) * 100) if len(user_df) > 0 else 0
                    color = {'Critical': 'red', 'Urgent': 'orange', 'Normal': 'green'}.get(label, 'gray')
                    st.markdown(f"**:{color}[{label}]**: {count} ({pct:.1f}%)")

    



if __name__ == "__main__":
    main()   