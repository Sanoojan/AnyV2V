import cv2
import os

# Define the video path and output folder
video_path = "demo/News_reading.mp4"  # Replace with your video file path
output_folder = "demo/News_reading"   # Folder to save frames
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

frame_count = 0

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when video ends
    
    # Construct the filename for the frame
    frame_filename = os.path.join(output_folder, f"{frame_count:05d}.png")
    
    # Save the frame as a PNG file
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

# Release the video capture object
cap.release()
print(f"Frames saved to '{output_folder}'. Total frames: {frame_count}")