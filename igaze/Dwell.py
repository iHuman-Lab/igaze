import pandas as pd

# Define your file path
file_path = r"C:\Users\elahe\OneDrive\Desktop\Eyetracker\593_RZ_5037271_Aug-05-15-35-12.csv"

# Load your dataset
df = pd.read_csv(file_path)

# Function to calculate dwell time and dwell count for a specific AOI
def calculate_dwell_time_and_dwell_count(df, x_min, x_max, y_min, y_max):
    
    # Filter the data to include only gaze points within the AOI
    aoi_gaze_points = df[
        (df['gaze_position_x'] >= x_min) & 
        (df['gaze_position_x'] <= x_max) & 
        (df['gaze_position_y'] >= y_min) & 
        (df['gaze_position_y'] <= y_max)
    ]

    # Calculate the total dwell time by summing durations
    total_dwell_time_in_aoi = aoi_gaze_points['duration'].sum()

    # Calculate the dwell count (number of individual gaze points within AOI)
    dwell_count = len(aoi_gaze_points)

    # Return both dwell time and dwell count
    return total_dwell_time_in_aoi, dwell_count

# Define your AOIs with x and y boundaries
aois = [
    {"name": "AOI_1", "x_min": 70, "x_max": 80, "y_min": 100, "y_max": 110},
    {"name": "AOI_2", "x_min": 50, "x_max": 60, "y_min": 90, "y_max": 100},
    {"name": "AOI_3", "x_min": 30, "x_max": 40, "y_min": 70, "y_max": 80},
]

# Initialize total dwell times
total_dwell_times = {}

# Calculate and store dwell time and dwell count for each AOI
for aoi in aois:
    dwell_time, dwell_count = calculate_dwell_time_and_dwell_count(df, aoi["x_min"], aoi["x_max"], aoi["y_min"], aoi["y_max"])
    total_dwell_times[aoi["name"]] = dwell_time
    print(f"AOI: {aoi['name']} - Dwell time: {dwell_time} ms, Dwell count: {dwell_count}")

# Calculate the percentage of time spent in AOI_1
total_time_spent = sum(total_dwell_times.values())
time_spent_AOI1 = total_dwell_times["AOI_1"] / total_time_spent * 100

print(f"Time spent in AOI_1: {time_spent_AOI1:.2f}%")

