# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:12:05 2024

@author: elahe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Load the CSV file
file_path = (r"C:\Users\elahe\igaze\pcadf.csv")
data = pd.read_csv(file_path)

# Convert the string to list
data['gaze_start'] = data['gaze_start'].apply(eval)  
data['gaze_end'] = data['gaze_end'].apply(eval)      

# Extract x1, y1 and x2, y2 
data['x1'] = data['gaze_start'].apply(lambda point: point[0])  
data['y1'] = data['gaze_start'].apply(lambda point: point[1])  
data['x2'] = data['gaze_end'].apply(lambda point: point[0])    
data['y2'] = data['gaze_end'].apply(lambda point: point[1])    

#saccade amplitude
data['saccade_amplitude'] = np.sqrt((data['x2'] - data['x1'])**2 + (data['y2'] - data['y1'])**2)
saccade_amplitude = data['saccade_amplitude']
 #Show data
print(data[['x1', 'y1', 'x2', 'y2', 'saccade_amplitude']])

#saccade frequency
saccade_count = (data['saccade_amplitude'] > 0.1).sum()
frame_id = data['frame_id']

# Filter the data for saccade amplitude less than 0.1
filtered_data = data[data['saccade_amplitude'] < 0.1]


