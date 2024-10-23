

import pandas as pd

# Load the data
file_path = (r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Atari\593_RZ_5037271_Aug-05-15-35-12.csv")
data = pd.read_csv(file_path)

# Define thresholds
spatial_threshold = 1.0  # in pixels (Euclidean distance between gaze points)
temporal_threshold = 100  # in milliseconds (minimum duration for a fixation)

# Extract fixations based on thresholds
def extract_fixation(data, temporal_threshold=100, spatial_threshold=1.0):
    fixations = []  # List to store fixations

    # Loop through the data starting from the second element
    for i in range(1, len(data)):
        # Calculate the duration and distance between successive gaze points
        duration = data['duration'][i]
        distance = ((data['gaze_position_x'][i] - data['gaze_position_x'][i-1]) ** 2 + 
                    (data['gaze_position_y'][i] - data['gaze_position_y'][i-1]) ** 2) ** 0.5
        
        # Check if both conditions are met for a fixation
        if duration >= temporal_threshold and distance <= spatial_threshold:
            fixations.append({
                "gaze_position_x": data['gaze_position_x'][i],
                "gaze_position_y": data['gaze_position_y'][i],
                "duration": duration
            })

    return fixations  # Return the list of fixations

# Call the function to extract fixations
fixations = extract_fixation(data, temporal_threshold=100, spatial_threshold=1.0)

# Count the number of fixations
number_of_fixations = len(fixations)

# Calculate the total time (in seconds)
total_time_ms = data['duration'].sum()  # Total time in milliseconds
total_time_seconds = total_time_ms / 1000  # Convert to seconds

# Calculate fixation rate (fixations per second), handling total time being zero
fixation_rate = number_of_fixations / total_time_seconds if total_time_seconds > 0 else 0

# Display the results
print(f"Number of fixations: {number_of_fixations}")
print(f"Total time (seconds): {total_time_seconds:.2f}")
print(f"Fixation rate (fixations per second): {fixation_rate:.2f}")
