import os
import csv

# Directory where audio files are stored
audio_directory = '/Users/bhanutejavaravenkata/desktop/wavee'

# Storing file paths and corresponding labels
data = []

for filename in os.listdir(audio_directory):
    if filename.endswith('.wav'):  
        file_path = os.path.join(audio_directory, filename)
        

        language_label = filename[:2].lower()  

        data.append({'File Path': file_path, 'Label': language_label})

csv_file_path = os.path.expanduser('/Users/bhanutejavaravenkata/Desktop/audio_samp.csv')  # Save to the desktop
fieldnames = ['File Path', 'Label']

with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(data)
