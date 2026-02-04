#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 2023

@author: lefteris

@env: hermis

@subject: emergency app

"""

import os
from flask import Flask, request, jsonify
from app.database import Database
from app.map_integration import MapIntegration
from app.routing_algorithm import OptimalEm
import pandas as pd


# Configuration from environment variables
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
PORT = int(os.environ.get('FLASK_PORT', 5000))
DB_PATH = os.environ.get('DB_PATH', 'app/emergency_data.db')

app = Flask(__name__)

db = Database(DB_PATH)  # Initialize the database

# Create the table (you might do this in a separate initialization step)
db.create_table()


def shutdown_server():
    # Get the WSGI server (e.g., gunicorn, uWSGI) from the environment
    # and initiate the shutdown process.
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with a WSGI server')
    func()


# Define a route for the homepage
@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/submit', methods=['POST'])
def submit_data():
    if request.is_json:
        json_data = request.get_json()
        name = json_data.get('name')
        phone_number = json_data.get('phone_number')
        message = json_data.get('message')
        coordinates_lat = json_data.get('coordinates_lat')
        coordinates_long = json_data.get('coordinates_long')
        severity = json_data.get('severity', 2)
        dt = json_data.get('dt')
    else:
        name = request.form.get('name')
        phone_number = request.form.get('phone_number')
        message = request.form.get('message')
        coordinates_lat = request.form.get('coordinates_lat')
        coordinates_long = request.form.get('coordinates_long')
        severity = request.form.get('severity', 2)
        dt = request.form.get('dt')

    # Validate required fields
    if not all([name, phone_number, message, coordinates_lat, coordinates_long, dt]):
        return jsonify({"error": "Missing required fields"}), 400

    # Validate coordinates
    try:
        lat = float(coordinates_lat)
        lon = float(coordinates_long)
        if not (-90 <= lat <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400
        if not (-180 <= lon <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Coordinates must be valid numbers"}), 400

    # Validate severity (1=Critical, 2=Urgent, 3=Normal)
    try:
        severity = int(severity)
        if severity not in [1, 2, 3]:
            return jsonify({"error": "Severity must be 1 (Critical), 2 (Urgent), or 3 (Normal)"}), 400
    except (ValueError, TypeError):
        severity = 2  # Default to Urgent

    # Validate datetime format
    try:
        from datetime import datetime
        datetime.fromisoformat(dt)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 (e.g., 2024-09-22T14:30:00)"}), 400

    # Insert the user data into the database
    try:
        db.insert_user_data(name, phone_number, message, lat, lon, dt, severity)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    user_data = {
        "result": "Data received and stored successfully!",
        "name": name,
        "phone_number": phone_number,
        "message": message,
        "coordinates_lat": lat,
        "coordinates_long": lon,
        "severity": severity,
        "dt": dt,
    }

    return jsonify(user_data)

@app.route('/insertbase', methods=['POST'])
def add_base():
    if request.is_json:
        json_data = request.get_json()
        name = json_data.get('name')
        base_type = json_data.get('base_type')
        coordinates_lat = json_data.get('coordinates_lat')
        coordinates_long = json_data.get('coordinates_long')
        capacity = json_data.get('capacity')
    else:
        name = request.form.get('name')
        base_type = request.form.get('base_type')
        coordinates_lat = request.form.get('coordinates_lat')
        coordinates_long = request.form.get('coordinates_long')
        capacity = request.form.get('capacity')

    # Validate required fields
    if not all([name, base_type, coordinates_lat, coordinates_long, capacity]):
        return jsonify({"error": "Missing required fields"}), 400

    # Validate coordinates
    try:
        lat = float(coordinates_lat)
        lon = float(coordinates_long)
        if not (-90 <= lat <= 90):
            return jsonify({"error": "Latitude must be between -90 and 90"}), 400
        if not (-180 <= lon <= 180):
            return jsonify({"error": "Longitude must be between -180 and 180"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Coordinates must be valid numbers"}), 400

    # Validate capacity
    try:
        cap = int(capacity)
        if cap <= 0:
            return jsonify({"error": "Capacity must be a positive integer"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Capacity must be a valid integer"}), 400

    # Insert the base data into the database
    try:
        db.insert_operation_base(name, base_type, lat, lon, cap)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    base_data = {
        "result": "Data received and stored successfully!",
        "name": name,
        "base_type": base_type,
        "coordinates_lat": lat,
        "coordinates_long": lon,
        "capacity": cap
    }

    return jsonify(base_data)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    db.close()  # Close the database connection
    shutdown_server()
    return 'Server shutting down...'


@app.route('/admin/map', methods=['POST'])
def admin_map():
    user_data = db.fetch_user_data()
    base_data = db.fetch_operation_base()
    # Create a list of dictionaries with appropriate keys
    # user_data columns: id, name, phone_number, message, coordinates_lat, coordinates_long, severity, dt
    formatted_data = [
        {
            "id": item[0],
            "name": item[1],
            "phone_number": item[2],
            "message": item[3],
            "coordinates_lat": item[4],
            "coordinates_long": item[5],
            "severity": item[6] if len(item) > 7 else 2,
            "dt": item[7] if len(item) > 7 else item[6]
        }
        for item in user_data
    ]


    formatted_data_op = [
        {
            "id": item[0],
            "name": item[1],
            "base_type": item[2],
            "coordinates_lat": item[3],
            "coordinates_long": item[4],
            "capacity": item[5]
        }
        for item in base_data
    ]

    # Combine both data sets into one response
    response_data = {
        'user_data': formatted_data,
        'base_data': formatted_data_op
    }

    # Return the combined data as a single JSON response
    return jsonify(response_data)

    

@app.route('/display_data',  methods=['POST'])
def display_data():
    user_data = db.fetch_user_data()
    base_data = db.fetch_operation_base()
    # Create a list of dictionaries with appropriate keys
    # user_data columns: id, name, phone_number, message, coordinates_lat, coordinates_long, severity, dt
    formatted_data = [
        {
            "id": item[0],
            "name": item[1],
            "phone_number": item[2],
            "message": item[3],
            "coordinates_lat": item[4],
            "coordinates_long": item[5],
            "severity": item[6] if len(item) > 7 else 2,
            "dt": item[7] if len(item) > 7 else item[6]
        }
        for item in user_data
    ]


    formatted_data_op = [
        {
            "id": item[0],
            "name": item[1],
            "base_type": item[2],
            "coordinates_lat": item[3],
            "coordinates_long": item[4],
            "capacity": item[5]
        }
        for item in base_data
    ]


    response_data = {
        'user_data': formatted_data,
        'base_data': formatted_data_op
    }

    # Return the combined data as a single JSON response
    return jsonify(response_data)



@app.route('/delete', methods=['POST'])
def delete_data():
    data_id = request.form.get('id')
    dtable = request.form.get('table')

    # Validate required fields
    if not data_id or not dtable:
        return jsonify({"error": "Missing required fields (id and table)"}), 400

    # Validate ID is an integer
    try:
        data_id = int(data_id)
    except (ValueError, TypeError):
        return jsonify({"error": "ID must be a valid integer"}), 400

    # Validate table name
    if dtable not in ['Emergency Bases', 'Emergency Data']:
        return jsonify({"error": "Invalid table name. Must be 'Emergency Bases' or 'Emergency Data'"}), 400

    # Delete data
    try:
        if dtable == 'Emergency Bases':
            db.delete_operation_base(data_id)
        elif dtable == 'Emergency Data':
            db.delete_user_data(data_id)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500

    return jsonify({"result": f"Data deleted successfully", "id": data_id, "table": dtable})


@app.route('/optimize', methods=['POST'])
def optimize():

    results_bases = db.fetch_operation_base()
    columns_base = [desc[0] for desc in db.cursor.description]
    bases_df = pd.DataFrame(results_bases, columns=columns_base)

    results_em = db.fetch_user_data()
    columns_em = [desc[0] for desc in db.cursor.description]
    em_df = pd.DataFrame(results_em, columns=columns_em)

    if len(bases_df) == 0:
        return jsonify({
            'error': 'No bases available',
            'message': 'Please add at least one operation base before running optimization.'
        }), 400

    if len(em_df) == 0:
        return jsonify({
            'error': 'No emergencies',
            'message': 'No emergency calls to optimize.'
        }), 400

    opt = OptimalEm(bases_df, em_df)

    # Use triage-enabled optimization
    result = opt.get_report_with_triage(write_log=True)

    # Validate results - check capacity constraints
    report = result['assignments']
    violations = []

    if len(report) > 0:
        base_totals = report.groupby('base_id')['supply'].sum().to_dict()
        capacity_map = dict(zip(bases_df['id'], bases_df['capacity']))

        for base_id, total in base_totals.items():
            cap = capacity_map.get(base_id, 0)
            if total > cap:
                violations.append({
                    'base_id': int(base_id),
                    'assigned': int(total),
                    'capacity': int(cap)
                })

    # Build response
    response = {
        'assignments': report.to_dict('records'),
        'solver_status': result['solver_status'],
        'is_optimal': result['is_optimal'],
        'total_capacity': result['total_capacity'],
        'num_emergencies': result['num_emergencies'],
        'violations': violations,
        'triage_applied': result['triage_applied']
    }

    # Add queue info if triage was applied
    if result['triage_applied']:
        queued_df = result['queued']
        response['queued'] = queued_df.to_dict('records')
        response['assigned_count'] = result.get('assigned_count', len(report))
        response['queued_count'] = result.get('queued_count', len(queued_df))

    return jsonify(response)


if __name__ == '__main__':
    print(f"Starting Flask server on {HOST}:{PORT} (debug={DEBUG_MODE})")
    print(f"Database: {DB_PATH}")
    app.run(debug=DEBUG_MODE, host=HOST, port=PORT)




