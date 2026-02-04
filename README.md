# Emergency Response Resource Allocation System

A proof-of-concept application demonstrating optimization techniques for emergency resource allocation using linear programming, geospatial analysis, and full-stack web development.

## Project Overview

This system models the assignment of emergency response resources (e.g., fire stations, EMS units, police stations) to emergency calls with the goal of minimizing total response distance. It demonstrates:

- **Operations Research**: Linear programming optimization using PuLP
- **Geospatial Analysis**: Haversine distance calculations for location-based routing
- **Full-Stack Development**: Flask REST API backend + Streamlit dashboard frontend
- **Data Visualization**: Interactive maps (Folium), charts (Plotly), and analytics

### Use Case: Strategic Planning Tool

This application is designed as a **strategic planning and analysis tool** rather than real-time dispatch. Example scenarios:

- **Facility Location Planning**: "Given a set of historical emergency calls, where should we position new response bases?"
- **Resource Allocation Analysis**: "How should we optimally allocate limited resources across multiple simultaneous incidents?"
- **Performance Benchmarking**: "Compare actual response patterns against optimal allocation"

### Important Note

This is a **simplified proof-of-concept** focusing on optimization formulation and technical implementation. Real-world emergency dispatch systems involve additional complexity including:

- Real-time dynamic dispatch with continuous call arrivals
- Unit availability and status tracking (en-route, on-scene, available)
- Multi-objective optimization (minimize max response time, not just total distance)
- Integration with CAD (Computer-Aided Dispatch) systems

## Features

### Core Functionality

1. **Emergency Call Management**
   - Submit emergency calls with location coordinates and severity level
   - Severity levels: Critical (1), Urgent (2), Normal (3)
   - Store call data with timestamps
   - Visualize call locations on interactive maps

2. **Operation Base Management**
   - Define response facility locations and capacities
   - Support multiple facility types
   - Track available resources per base

3. **Optimization Engine**
   - Linear programming formulation using PuLP
   - Minimize total response distance
   - Respect capacity constraints at bases
   - Ensure all calls receive response
   - **Triage System**: When capacity is insufficient, prioritizes by severity (Critical first) then by time (oldest first), queuing lower-priority emergencies

4. **Analytics Dashboard**
   - Interactive map with proximity-based filtering
   - Response time distribution analysis
   - Incident density heatmaps
   - Temporal pattern analysis

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  ┌──────────┬───────────┬──────────┬───────────────────┐   │
│  │ Submit   │ Insert    │ Map      │ Data     Analytics │   │
│  │ Form     │ Base      │ View     │ Tables   Dashboard │   │
│  └──────────┴───────────┴──────────┴───────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/REST API
┌─────────────────────▼───────────────────────────────────────┐
│                    Flask API Backend                         │
│  ┌──────────┬───────────┬──────────┬───────────────────┐   │
│  │ /submit  │/insertbase│/admin/map│/optimize  /delete │   │
│  └──────────┴───────────┴──────────┴───────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────────┬──────────────────┐
        │             │                 │                  │
┌───────▼──────┐ ┌───▼──────────┐ ┌────▼──────────┐ ┌────▼────────┐
│  Database    │ │Routing       │ │Map            │ │ Geospatial  │
│  Layer       │ │Algorithm     │ │Integration    │ │ Utils       │
│  (SQLite)    │ │ (PuLP)       │ │ (Folium)      │ │ (Haversine) │
└──────────────┘ └──────────────┘ └───────────────┘ └─────────────┘
```

### Components

- **`main.py`**: Flask REST API backend with endpoints for CRUD operations and optimization
- **`streamlit_frontend.py`**: Multi-page dashboard for visualization and interaction
- **`app/database.py`**: SQLite database wrapper for emergency calls and operation bases
- **`app/routing_algorithm.py`**: Linear programming optimization using PuLP
- **`app/map_integration.py`**: Folium-based interactive map generation
- **`tests/create_dummy.py`**: Test data generator for development and demos

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

```bash
pip install flask
pip install streamlit
pip install pandas
pip install numpy
pip install folium
pip install plotly
pip install pulp
pip install requests
```

### Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd crisis_management_prj
```

2. **Install dependencies**

```bash
pip install -r requirements.txt  # If provided
# OR install manually as listed above
```

3. **Initialize the database**

The database is created automatically on first run. Default location: `app/emergency_data.db`

4. **Configure environment (optional)**

```bash
cp .env.example .env
# Edit .env to customize settings
```

| Variable    | Default               | Description             |
| ----------- | --------------------- | ----------------------- |
| FLASK_DEBUG | false                 | Enable Flask debug mode |
| FLASK_HOST  | 127.0.0.1             | API host address        |
| FLASK_PORT  | 5000                  | API port                |
| DB_PATH     | app/emergency_data.db | Database file location  |

## Usage

### Running the Application

**Step 1: Start the Flask Backend**

```bash
python main.py
```

The API will run on `http://127.0.0.1:5000`.

> **Note:** You'll see a warning about the development server - this is normal for local development. For production, use a WSGI server like Gunicorn: `gunicorn -w 4 main:app`

**Step 2: Start the Streamlit Frontend** (in a separate terminal)

```bash
streamlit run streamlit_frontend.py
```

The dashboard will open in your browser (typically `http://localhost:8501`).

### Generating Test Data

To populate the system with dummy data for testing:

```bash
# Ensure Flask backend is running first
python tests/create_dummy.py
```

This creates:

- 100 random emergency calls
- 10 operation bases with varying capacities
- All centered around coordinates (39.553464, 21.759884)

### Dashboard Navigation

The Streamlit interface has 5 pages:

1. **Submit Form**: Add new emergency calls manually
   - Enter name, phone, message, coordinates, severity level, timestamp

2. **Insert Base**: Add new operation bases
   - Enter base name, type, coordinates, capacity

3. **Map**: Interactive visualization
   - View all calls and bases on map
   - Filter by base ID, base type, distance, response time
   - Color-coded markers (red = bases, white = calls)

4. **Data**: View and manage data
   - Tables showing all calls and bases
   - Delete functionality

5. **Analytics**: Charts and insights
   - Incident density heatmap
   - Response time distribution
   - Incidents over time by base type
   - Proximity analysis

## API Documentation

### Base URL: `http://127.0.0.1:5000`

### Endpoints

#### `POST /submit`

Submit a new emergency call.

**Request Body (JSON):**

```json
{
  "name": "John Doe",
  "phone_number": "555-1234",
  "message": "Medical emergency",
  "coordinates_lat": 39.5534,
  "coordinates_long": 21.7598,
  "severity": 1,
  "dt": "2026-01-15T14:30:00"
}
```

| Field    | Required | Description                                 |
| -------- | -------- | ------------------------------------------- |
| severity | No       | 1=Critical, 2=Urgent, 3=Normal (default: 2) |

**Response:**

```json
{
  "result": "Data received and stored successfully!",
  "name": "John Doe",
  "phone_number": "555-1234",
  "message": "Medical emergency",
  "coordinates_lat": 39.5534,
  "coordinates_long": 21.7598,
  "severity": 1,
  "dt": "2026-01-15T14:30:00"
}
```

#### `POST /insertbase`

Add a new operation base.

**Request Body (JSON):**

```json
{
  "name": "Station 5",
  "base_type": "Fire Station",
  "coordinates_lat": 39.5600,
  "coordinates_long": 21.7700,
  "capacity": 5
}
```

#### `POST /admin/map`

Fetch all data for map visualization.

**Response:**

```json
{
  "user_data": [
    {
      "id": 1,
      "name": "John Doe",
      "phone_number": "555-1234",
      "message": "Medical emergency",
      "coordinates_lat": 39.5534,
      "coordinates_long": 21.7598,
      "dt": "2026-01-15T14:30:00"
    }
  ],
  "base_data": [
    {
      "id": 1,
      "name": "Station 5",
      "base_type": "Fire Station",
      "coordinates_lat": 39.5600,
      "coordinates_long": 21.7700
    }
  ]
}
```

#### `POST /display_data`

Fetch all data including capacity information.

#### `POST /optimize`

Run optimization algorithm to assign bases to calls. Automatically applies triage when capacity is insufficient.

**Response (JSON):**

```json
{
  "assignments": [...],
  "queued": [...],
  "triage_applied": true,
  "solver_status": "Optimal",
  "total_capacity": 50,
  "num_emergencies": 100,
  "assigned_count": 50,
  "queued_count": 50
}
```

| Field          | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| assignments    | List of base-to-emergency assignments with distances        |
| queued         | Emergencies that couldn't be assigned (when triage applied) |
| triage_applied | Whether capacity was insufficient and triage was used       |

#### `POST /delete`

Delete an entry by ID.

**Request Body (form data):**

```
id: 1
table: "Emergency Data" or "Emergency Bases"
```

#### `POST /shutdown`

Gracefully shutdown the server and close database connections.

## Optimization Model

### Linear Programming Formulation

**Decision Variables:**

- `x[i][j]` = units of help sent from base `i` to call `j`

**Objective Function:**
Minimize total distance:

```
Minimize: Σᵢ Σⱼ (distance[i,j] × x[i,j])
```

**Constraints:**

1. **Supply Constraint**: Each base cannot exceed its capacity

```
Σⱼ x[i,j] ≤ capacity[i]  for all bases i
```

2. **Demand Constraint**: Each call must receive at least 1 unit of help

```
Σᵢ x[i,j] ≥ 1  for all calls j
```

3. **Non-negativity**: All assignments must be non-negative integers

```
x[i,j] ≥ 0 and integer
```

### Distance Calculation

Uses the **Haversine formula** for great-circle distance between two points on Earth:

```python
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # Earth radius in km
    # Convert to radians and apply Haversine formula
    # Returns distance in kilometers
```

## Database Schema

### `user_data` Table (Emergency Calls)

| Column           | Type     | Description                    |
| ---------------- | -------- | ------------------------------ |
| id               | INTEGER  | Primary key (auto-increment)   |
| name             | TEXT     | Caller name                    |
| phone_number     | TEXT     | Contact number                 |
| message          | TEXT     | Emergency description          |
| coordinates_lat  | REAL     | Latitude                       |
| coordinates_long | REAL     | Longitude                      |
| severity         | INTEGER  | 1=Critical, 2=Urgent, 3=Normal |
| dt               | DATETIME | Timestamp                      |

### `op_pos` Table (Operation Bases)

| Column           | Type     | Description                  |
| ---------------- | -------- | ---------------------------- |
| id               | INTEGER  | Primary key (auto-increment) |
| name             | TEXT     | Base name                    |
| base_type        | TEXT     | Type of facility             |
| coordinates_lat  | REAL     | Latitude                     |
| coordinates_long | REAL     | Longitude                    |
| capacity         | INTEGER  | Available resource units     |
| dt               | DATETIME | Timestamp                    |

## Known Limitations

### Domain Limitations

1. **Batch Processing**: Optimizes all calls simultaneously rather than handling dynamic arrivals.
2. **Simplified Objective**: Minimizes total distance rather than maximum response time (which is more critical in emergencies).
3. **No Unit Status**: Doesn't track whether resources are available, en-route, or on-scene.
4. **Static Capacity**: Doesn't model resources becoming available after completing calls.

## Future Enhancements

### High Priority

- [ ] Add input validation for all API endpoints
- [ ] Implement proper database connection pooling

### Medium Priority

- [ ] Add unit status tracking (available, dispatched, on-scene, returning)
- [ ] Implement dynamic dispatch simulation (calls arrive over time)
- [ ] Add multi-objective optimization (minimize max response time + total distance)
- [ ] Export optimization reports (CSV, PDF)
- [ ] Add authentication for admin operations

### Low Priority

- [ ] Real-time updates using WebSockets
- [ ] Historical analysis and reporting
- [ ] Route visualization on map (polylines from base to call)
- [ ] Performance optimization (vectorize distance calculations)
- [ ] Docker container deployment
- [ ] Production WSGI server configuration (Gunicorn/uWSGI)

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: Streamlit
- **Database**: SQLite
- **Optimization**: PuLP (CBC solver)
- **Mapping**: Folium
- **Visualization**: Plotly
- **Geospatial**: Custom Haversine implementation
- **Data Processing**: Pandas, NumPy

## Development Notes

### Base Types

- Fire Station
- EMS Station
- Police Station

### Response Time Calculation

Current formula:

```python
response_time = (distance × type_weight) + 10 minutes
```

Weights (km/min):

- Fire Station: 0.8
- EMS Station: 1.0
- Police Station: 1.2

*Note: This is a simplified model. Real response time depends on vehicle speed, traffic, terrain, and preparation time.*

### Severity Levels

| Level | Label    | Priority |
| ----- | -------- | -------- |
| 1     | Critical | Highest  |
| 2     | Urgent   | Medium   |
| 3     | Normal   | Lowest   |

Used by the triage system when total base capacity is less than the number of emergencies.

## Contributing

This is a portfolio project for demonstration purposes. However, if you find bugs or have suggestions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

### GNU General Public License v3.0

## Author

Eleftherios Ntovoris

---

**Disclaimer**: This is a proof-of-concept system for educational and portfolio purposes. It is not intended for use in actual emergency response operations.
