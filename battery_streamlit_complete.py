import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import time
import json
import io
from datetime import datetime, timedelta
import math

# Page configuration
st.set_page_config(
    page_title="Battery Cell Simulation System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .cell-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .cell-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #3b82f6;
    }
    
    .task-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .status-running {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #065f46;
        padding: 0.5rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-idle {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        color: #374151;
        padding: 0.5rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'cells_data' not in st.session_state:
        st.session_state.cells_data = {}
    if 'task_dict' not in st.session_state:
        st.session_state.task_dict = {}
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = []
    if 'is_simulation_running' not in st.session_state:
        st.session_state.is_simulation_running = False
    if 'simulation_time' not in st.session_state:
        st.session_state.simulation_time = 0
    if 'cell_counter' not in st.session_state:
        st.session_state.cell_counter = 0
    if 'task_counter' not in st.session_state:
        st.session_state.task_counter = 0

init_session_state()

# Cell configuration based on your original code
CELL_TYPES = {
    'lfp': {
        'name': 'LFP (Lithium Iron Phosphate)',
        'voltage': 3.2,
        'min_voltage': 2.8,
        'max_voltage': 3.6,
        'color': '#10b981',
        'capacity_range': (90, 100)
    },
    'li-ion': {
        'name': 'Li-ion (Lithium Ion)',
        'voltage': 3.6,
        'min_voltage': 3.2,
        'max_voltage': 4.0,
        'color': '#3b82f6',
        'capacity_range': (85, 100)
    },
    'nmc': {
        'name': 'NMC (Nickel Manganese Cobalt)',
        'voltage': 3.7,
        'min_voltage': 3.0,
        'max_voltage': 4.2,
        'color': '#8b5cf6',
        'capacity_range': (88, 100)
    },
    'lto': {
        'name': 'LTO (Lithium Titanate)',
        'voltage': 2.4,
        'min_voltage': 1.5,
        'max_voltage': 2.7,
        'color': '#f59e0b',
        'capacity_range': (85, 95)
    }
}

def create_cell(cell_type):
    """Create a new cell following your original structure"""
    st.session_state.cell_counter += 1
    cell_key = f"cell_{st.session_state.cell_counter}_{cell_type}"
    
    config = CELL_TYPES[cell_type]
    voltage = config['voltage']
    min_voltage = config['min_voltage']
    max_voltage = config['max_voltage']
    current = 0.0
    temp = round(random.uniform(25, 40), 1)
    capacity = round(voltage * current, 2) if current > 0 else round(random.uniform(*config['capacity_range']), 1)
    
    cell_data = {
        "type": cell_type,
        "voltage": voltage,
        "current": current,
        "temp": temp,
        "capacity": capacity,
        "min_voltage": min_voltage,
        "max_voltage": max_voltage,
        "status": "idle",
        "active_task": None,
        "task_start_time": None,
        "color": config['color']
    }
    
    st.session_state.cells_data[cell_key] = cell_data
    return cell_key

def create_task(task_type, parameters):
    """Create a task following your original structure"""
    st.session_state.task_counter += 1
    task_key = f"task_{st.session_state.task_counter}"
    
    task_data = {"task_type": task_type, **parameters}
    st.session_state.task_dict[task_key] = task_data
    return task_key

def simulate_cell_behavior(cell_key, cell_data, task_data, elapsed_time):
    """Simulate realistic battery behavior"""
    if not task_data:
        return cell_data
    
    task_type = task_data["task_type"]
    duration = task_data.get("time_seconds", 300)
    
    # Calculate progress (0 to 1)
    progress = min(elapsed_time / duration, 1.0)
    
    if task_type == "CC_CV":
        # Charging simulation with CC and CV phases
        current_val = task_data.get("current", 1.0)
        cv_voltage = task_data.get("cv_voltage", cell_data["max_voltage"])
        
        if progress < 0.7:  # CC phase (first 70% of time)
            # Constant current, rising voltage
            cell_data["current"] = current_val
            voltage_rise = (cv_voltage - cell_data["voltage"]) * 0.01
            cell_data["voltage"] = min(cell_data["voltage"] + voltage_rise, cv_voltage)
            cell_data["temp"] = min(cell_data["temp"] + 0.05, 45)
            cell_data["capacity"] = min(cell_data["capacity"] + 0.1, 100)
        else:  # CV phase (last 30% of time)
            # Constant voltage, tapering current
            cell_data["voltage"] = cv_voltage
            cell_data["current"] = current_val * (1 - (progress - 0.7) / 0.3)
            cell_data["temp"] = max(cell_data["temp"] - 0.02, 25)
        
        cell_data["status"] = "charging"
        
    elif task_type == "CC_CD":
        # Discharging simulation
        current_val = task_data.get("current", 1.0) if "current" in task_data else 1.0
        
        # Extract current from cc_cp field if needed
        if "cc_cp" in task_data:
            cc_cp = task_data["cc_cp"]
            if 'A' in str(cc_cp):
                current_val = float(str(cc_cp).replace('A', ''))
            elif 'W' in str(cc_cp):
                power = float(str(cc_cp).replace('W', ''))
                current_val = power / cell_data["voltage"]
        
        cell_data["current"] = -current_val  # Negative for discharge
        
        # Voltage drops during discharge
        voltage_drop = (cell_data["voltage"] - cell_data["min_voltage"]) * 0.008
        cell_data["voltage"] = max(cell_data["voltage"] - voltage_drop, cell_data["min_voltage"])
        
        # Temperature rise and capacity decrease
        cell_data["temp"] = min(cell_data["temp"] + 0.03, 40)
        cell_data["capacity"] = max(cell_data["capacity"] - 0.08, 0)
        cell_data["status"] = "discharging"
        
    elif task_type == "IDLE":
        # Rest simulation - parameters stabilize
        cell_data["current"] = 0.0
        cell_data["voltage"] += random.uniform(-0.005, 0.005)  # Small voltage fluctuation
        cell_data["temp"] = 25 + (cell_data["temp"] - 25) * 0.98  # Cooling to ambient
        cell_data["status"] = "idle"
    
    return cell_data

def run_simulation_step():
    """Run one simulation step for all cells"""
    if not st.session_state.is_simulation_running:
        return
    
    st.session_state.simulation_time += 1
    current_time = st.session_state.simulation_time
    
    # Record data point
    data_point = {"time": current_time}
    
    for cell_key, cell_data in st.session_state.cells_data.items():
        # Check if cell has an active task
        if cell_data.get("active_task"):
            task_key = cell_data["active_task"]
            task_data = st.session_state.task_dict.get(task_key)
            
            if task_data:
                # Calculate elapsed time for this task
                start_time = cell_data.get("task_start_time", current_time)
                elapsed_time = current_time - start_time
                
                # Simulate cell behavior
                cell_data = simulate_cell_behavior(cell_key, cell_data, task_data, elapsed_time)
                
                # Check if task is complete
                duration = task_data.get("time_seconds", 300)
                if elapsed_time >= duration:
                    cell_data["active_task"] = None
                    cell_data["task_start_time"] = None
                    cell_data["status"] = "idle"
                    cell_data["current"] = 0.0
        
        # Record cell data
        data_point[f"{cell_key}_voltage"] = cell_data["voltage"]
        data_point[f"{cell_key}_current"] = cell_data["current"]
        data_point[f"{cell_key}_temp"] = cell_data["temp"]
        data_point[f"{cell_key}_capacity"] = cell_data["capacity"]
        
        # Update session state
        st.session_state.cells_data[cell_key] = cell_data
    
    # Add data point to simulation history
    st.session_state.simulation_data.append(data_point)

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>🔋 Advanced Battery Cell Simulation System</h1>
    <p>Professional Battery Testing & Analysis Platform</p>
    <p>Real-time Simulation • Multi-Cell Management • Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Simulation Control Center")
    
    # Time and status display
    minutes = st.session_state.simulation_time // 60
    seconds = st.session_state.simulation_time % 60
    st.metric("🕐 Simulation Time", f"{minutes:02d}:{seconds:02d}")
    
    status_color = "🟢" if st.session_state.is_simulation_running else "🔴"
    st.metric("📊 Status", f"{status_color} {'Running' if st.session_state.is_simulation_running else 'Stopped'}")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start" if not st.session_state.is_simulation_running else "⏸️ Pause", 
                    type="primary", use_container_width=True):
            st.session_state.is_simulation_running = not st.session_state.is_simulation_running
    
    with col2:
        if st.button("⏹️ Reset", use_container_width=True):
            st.session_state.is_simulation_running = False
            st.session_state.simulation_time = 0
            st.session_state.simulation_data = []
            # Reset all cells
            for cell_key in st.session_state.cells_data:
                st.session_state.cells_data[cell_key]["active_task"] = None
                st.session_state.cells_data[cell_key]["task_start_time"] = None
                st.session_state.cells_data[cell_key]["status"] = "idle"
                st.session_state.cells_data[cell_key]["current"] = 0.0
            st.success("Simulation reset!")
    
    st.divider()
    
    # Quick Stats
    st.subheader("📈 Quick Stats")
    total_cells = len(st.session_state.cells_data)
    active_tasks = sum(1 for cell in st.session_state.cells_data.values() if cell.get("active_task"))
    data_points = len(st.session_state.simulation_data)
    
    st.metric("Total Cells", total_cells)
    st.metric("Active Tasks", active_tasks)
    st.metric("Data Points", data_points)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔋 Cell Configuration", 
    "⚡ Task Management", 
    "📊 Real-time Monitoring", 
    "📈 Data Analysis", 
    "💾 Export & Settings"
])

with tab1:
    st.header("🔋 Cell Configuration")
    
    # Add cells section
    st.subheader("Add New Cells")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multiple cell creation
        st.write("**Quick Cell Creation:**")
        cell_type_quick = st.selectbox(
            "Select Cell Type",
            options=list(CELL_TYPES.keys()),
            format_func=lambda x: CELL_TYPES[x]['name'],
            key="quick_cell_type"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            num_cells = st.number_input("Number of cells to add", min_value=1, max_value=10, value=1)
        with col_b:
            if st.button("➕ Add Cells", type="primary"):
                for i in range(num_cells):
                    cell_key = create_cell(cell_type_quick)
                st.success(f"Added {num_cells} {CELL_TYPES[cell_type_quick]['name']} cell(s)")
                st.rerun()
    
    with col2:
        st.write("**Individual Cell Types:**")
        for cell_type, config in CELL_TYPES.items():
            if st.button(f"Add {config['name']}", key=f"add_{cell_type}", use_container_width=True):
                cell_key = create_cell(cell_type)
                st.success(f"Added {config['name']}")
                st.rerun()
    
    st.divider()
    
    # Display existing cells
    if st.session_state.cells_data:
        st.subheader("🔧 Manage Existing Cells")
        
        # Create grid layout for cells
        cells_per_row = 3
        cell_keys = list(st.session_state.cells_data.keys())
        
        for i in range(0, len(cell_keys), cells_per_row):
            cols = st.columns(cells_per_row)
            
            for j, col in enumerate(cols):
                if i + j < len(cell_keys):
                    cell_key = cell_keys[i + j]
                    cell_data = st.session_state.cells_data[cell_key]
                    
                    with col:
                        # Cell card
                        cell_type = cell_data['type']
                        config = CELL_TYPES[cell_type]
                        
                        st.markdown(f"""
                        <div class="cell-card">
                            <h4 style="color: {config['color']}; margin-bottom: 1rem;">
                                🔋 {cell_key}
                            </h4>
                            <div style="background: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <p><strong>Type:</strong> {config['name']}</p>
                                <p><strong>Status:</strong> <span style="color: {config['color']};">{cell_data['status'].upper()}</span></p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Cell metrics in grid
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("⚡ Voltage", f"{cell_data['voltage']:.2f}V")
                            st.metric("🌡️ Temperature", f"{cell_data['temp']:.1f}°C")
                        
                        with metric_col2:
                            st.metric("🔌 Current", f"{cell_data['current']:.2f}A")
                            st.metric("🔋 Capacity", f"{cell_data['capacity']:.1f}%")
                        
                        # Control buttons
                        button_col1, button_col2 = st.columns(2)
                        with button_col1:
                            if st.button("🎲 Randomize", key=f"rand_{cell_key}"):
                                # Randomize cell parameters
                                cell_data['voltage'] = config['voltage'] + random.uniform(-0.2, 0.2)
                                cell_data['temp'] = round(random.uniform(25, 40), 1)
                                cell_data['capacity'] = round(random.uniform(*config['capacity_range']), 1)
                                st.success("Parameters randomized!")
                                st.rerun()
                        
                        with button_col2:
                            if st.button("🗑️ Remove", key=f"remove_{cell_key}"):
                                del st.session_state.cells_data[cell_key]
                                st.success("Cell removed!")
                                st.rerun()
    else:
        st.info("No cells configured. Add cells using the options above.")

with tab2:
    st.header("⚡ Task Management")
    
    if not st.session_state.cells_data:
        st.warning("Please add cells first before creating tasks.")
    else:
        # Task creation form
        st.subheader("Create New Task")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_cell = st.selectbox(
                "Select Cell for Task",
                options=list(st.session_state.cells_data.keys()),
                format_func=lambda x: f"{x} ({CELL_TYPES[st.session_state.cells_data[x]['type']]['name']})"
            )
            
            task_type = st.selectbox(
                "Task Type",
                options=["CC_CV", "CC_CD", "IDLE"],
                help="CC_CV: Charging, CC_CD: Discharging, IDLE: Rest period"
            )
        
        with col2:
            duration = st.number_input("Duration (seconds)", min_value=10, max_value=3600, value=300)
            
            if task_type in ["CC_CV", "CC_CD"]:
                current_input = st.text_input(
                    "Current/Power",
                    value="5A",
                    help="Enter current (e.g., '5A') or power (e.g., '10W')"
                )
                
                if task_type == "CC_CV":
                    cv_voltage = st.number_input(
                        "CV Voltage (V)",
                        min_value=3.0,
                        max_value=4.5,
                        value=4.0,
                        step=0.1
                    )
        
        # Create task button
        if st.button("🚀 Create Task", type="primary"):
            task_params = {"time_seconds": duration}
            
            if task_type == "CC_CV":
                task_params.update({
                    "cc_cp": current_input,
                    "cv_voltage": cv_voltage,
                    "current": float(current_input.replace('A', '').replace('W', '')) if current_input.replace('A', '').replace('W', '').replace('.', '').isdigit() else 5.0
                })
            elif task_type == "CC_CD":
                task_params.update({
                    "cc_cp": current_input,
                    "current": float(current_input.replace('A', '').replace('W', '')) if current_input.replace('A', '').replace('W', '').replace('.', '').isdigit() else 5.0
                })
            
            task_key = create_task(task_type, task_params)
            
            # Assign task to cell
            if not st.session_state.cells_data[selected_cell].get("active_task"):
                st.session_state.cells_data[selected_cell]["active_task"] = task_key
                st.session_state.cells_data[selected_cell]["task_start_time"] = st.session_state.simulation_time
                st.success(f"Task {task_key} assigned to {selected_cell}")
            else:
                st.warning(f"Cell {selected_cell} already has an active task!")
        
        st.divider()
        
        # Display active tasks
        st.subheader("🎯 Active Tasks Overview")
        
        active_tasks_data = []
        for cell_key, cell_data in st.session_state.cells_data.items():
            if cell_data.get("active_task"):
                task_key = cell_data["active_task"]
                task_data = st.session_state.task_dict.get(task_key, {})
                
                # Calculate progress
                start_time = cell_data.get("task_start_time", 0)
                elapsed = st.session_state.simulation_time - start_time
                duration = task_data.get("time_seconds", 300)
                progress = min(elapsed / duration * 100, 100)
                
                active_tasks_data.append({
                    "Cell": cell_key,
                    "Task": task_key,
                    "Type": task_data.get("task_type", "Unknown"),
                    "Progress": f"{progress:.1f}%",
                    "Elapsed": f"{elapsed}s",
                    "Duration": f"{duration}s",
                    "Status": cell_data["status"]
                })
        
        if active_tasks_data:
            df_tasks = pd.DataFrame(active_tasks_data)
            st.dataframe(df_tasks, use_container_width=True)
            
            # Task control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏹️ Stop All Tasks"):
                    for cell_key in st.session_state.cells_data:
                        st.session_state.cells_data[cell_key]["active_task"] = None
                        st.session_state.cells_data[cell_key]["task_start_time"] = None
                        st.session_state.cells_data[cell_key]["status"] = "idle"
                        st.session_state.cells_data[cell_key]["current"] = 0.0
                    st.success("All tasks stopped!")
                    st.rerun()
        else:
            st.info("No active tasks. Create tasks above to get started.")
        
        # All tasks history
        if st.session_state.task_dict:
            st.subheader("📋 Task History")
            
            tasks_history = []
            for task_key, task_data in st.session_state.task_dict.items():
                tasks_history.append({
                    "Task ID": task_key,
                    "Type": task_data.get("task_type", "Unknown"),
                    "Duration": f"{task_data.get('time_seconds', 0)}s",
                    "Parameters": str({k: v for k, v in task_data.items() if k != "task_type"})
                })
            
            df_history = pd.DataFrame(tasks_history)
            st.dataframe(df_history, use_container_width=True)

with tab3:
    st.header("📊 Real-time Monitoring")
    
    if st.session_state.simulation_data and st.session_state.cells_data:
        # Create comprehensive real-time charts
        recent_data = st.session_state.simulation_data[-100:]  # Last 100 data points
        df = pd.DataFrame(recent_data)
        
        if not df.empty:
            # Multi-parameter dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('🔋 Voltage Trends', '⚡ Current Trends', '🌡️ Temperature Trends', '📊 Capacity Trends'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Plot data for each cell
            for cell_key, cell_data in st.session_state.cells_data.items():
                color = cell_data['color']
                cell_type = CELL_TYPES[cell_data['type']]['name']
                
                # Voltage plot
                if f"{cell_key}_voltage" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], 
                            y=df[f"{cell_key}_voltage"],
                            name=f"{cell_key}",
                            line=dict(color=color, width=3),
                            hovertemplate=f"<b>{cell_key}</b><br>Time: %{{x}}s<br>Voltage: %{{y:.2f}}V<extra></extra>"
                        ),
                        row=1, col=1
                    )
                
                # Current plot
                if f"{cell_key}_current" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], 
                            y=df[f"{cell_key}_current"],
                            name=f"{cell_key}",
                            line=dict(color=color, width=3),
                            showlegend=False,
                            hovertemplate=f"<b>{cell_key}</b><br>Time: %{{x}}s<br>Current: %{{y:.2f}}A<extra></extra>"
                        ),
                        row=1, col=2
                    )
                
                # Temperature plot
                if f"{cell_key}_temp" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], 
                            y=df[f"{cell_key}_temp"],
                            name=f"{cell_key}",
                            line=dict(color=color, width=3),
                            showlegend=False,
                            hovertemplate=f"<b>{cell_key}</b><br>Time: %{{x}}s<br>Temperature: %{{y:.1f}}°C<extra></extra>"
                        ),
                        row=2, col=1
                    )
                
                # Capacity plot
                if f"{cell_key}_capacity" in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['time'], 
                            y=df[f"{cell_key}_capacity"],
                            name=f"{cell_key}",
                            line=dict(color=color, width=3),
                            showlegend=False,
                            hovertemplate=f"<b>{cell_key}</b><br>Time: %{{x}}s<br>Capacity: %{{y:.1f}}%<extra></extra>"
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                height=700,
                title_text="🔋 Real-time Battery Parameters Dashboard",
                title_x=0.5,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=2)
            
            fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
            fig.update_yaxes(title_text="Current (A)", row=1, col=2)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig.update_yaxes(title_text="Capacity (%)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Real-time metrics cards
            st.subheader("📊 Current Status Overview")
            
            # Create metrics grid
            cols = st.columns(len(st.session_state.cells_data))
            for i, (cell_key, cell_data) in enumerate(st.session_state.cells_data.items()):
                with cols[i]:
                    cell_type = CELL_TYPES[cell_data['type']]['name']
                    status_emoji = "🔄" if cell_data['status'] == 'charging' else "⬇️" if cell_data['status'] == 'discharging' else "⏸️"
                    
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4 style="color: {cell_data['color']}; margin-bottom: 1rem;">
                            {status_emoji} {cell_key}
                        </h4>
                        <p><strong>Type:</strong> {cell_type}</p>
                        <p><strong>Status:</strong> {cell_data['status'].upper()}</p>
                        <hr>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; text-align: left;">
                            <div><strong>⚡ Voltage:</strong><br>{cell_data['voltage']:.2f}V</div>
                            <div><strong>🔌 Current:</strong><br>{cell_data['current']:.2f}A</div>
                            <div><strong>🌡️ Temp:</strong><br>{cell_data['temp']:.1f}°C</div>
                            <div><strong>🔋 Capacity:</strong><br>{cell_data['capacity']:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Auto-refresh for real-time updates
        if st.session_state.is_simulation_running:
            run_simulation_step()
            time.sleep(0.5)  # 0.5 second delay for smoother animation
            st.rerun()
    
    else:
        st.info("🚀 Start the simulation to see real-time monitoring data!")
        if st.button("▶️ Quick Start Demo", type="primary"):
            # Create demo cells and tasks
            demo_cell1 = create_cell('lfp')
            demo_cell2 = create_cell('li-ion')
            
            # Create demo tasks
            task1 = create_task('CC_CV', {
                'time_seconds': 300,
                'cc_cp': '3A',
                'cv_voltage': 3.6,
                'current': 3.0
            })
            
            task2 = create_task('CC_CD', {
                'time_seconds': 200,
                'cc_cp': '2A',
                'current': 2.0
            })
            
            # Assign tasks
            st.session_state.cells_data[demo_cell1]['active_task'] = task1
            st.session_state.cells_data[demo_cell1]['task_start_time'] = 0
            
            st.session_state.cells_data[demo_cell2]['active_task'] = task2
            st.session_state.cells_data[demo_cell2]['task_start_time'] = 0
            
            st.session_state.is_simulation_running = True
            st.success("Demo started! Check the real-time monitoring.")
            st.rerun()

with tab4:
    st.header("📈 Advanced Data Analysis")
    
    if st.session_state.simulation_data and len(st.session_state.simulation_data) > 1:
        df_full = pd.DataFrame(st.session_state.simulation_data)
        
        # Analysis overview
        st.subheader("🎯 Simulation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Runtime", f"{st.session_state.simulation_time}s")
        with col2:
            st.metric("📈 Data Points", len(st.session_state.simulation_data))
        with col3:
            total_energy = 0
            for cell_key, cell_data in st.session_state.cells_data.items():
                total_energy += abs(cell_data['voltage'] * cell_data['current'])
            st.metric("⚡ Total Power", f"{total_energy:.2f}W")
        with col4:
            avg_temp = np.mean([cell_data['temp'] for cell_data in st.session_state.cells_data.values()])
            st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C")
        
        st.divider()
        
        # Comparative analysis
        st.subheader("🔬 Comparative Cell Analysis")
        
        # Cell comparison charts
        comparison_metrics = st.selectbox(
            "Select metrics to compare",
            ["All Parameters", "Voltage", "Current", "Temperature", "Capacity"]
        )
        
        if comparison_metrics == "All Parameters":
            # Multi-metric comparison
            fig_comparison = make_subplots(
                rows=2, cols=2,
             
