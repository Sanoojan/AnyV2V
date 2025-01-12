# from PIL import Image
# import os

# # Define the folder containing the images and the output GIF path
# input_folder = "demo/A guy reading the news"  # Replace with the path to your folder
# output_gif = "demo/news_reading.gif"               # Name of the output GIF
# fps = 8                                # Frames per second
# frame_duration = int(1000 / fps)        # Duration per frame in milliseconds

# # Get a sorted list of image files in the folder
# image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])

# # Select the first 50 images
# selected_images = image_files[:32]

# # Open the images
# frames = [Image.open(img) for img in selected_images]

# # Save as a GIF
# frames[0].save(
#     output_gif,
#     save_all=True,
#     append_images=frames[1:],
#     duration=frame_duration,
#     loop=0  # 0 means infinite loop
# )

# print(f"GIF saved as '{output_gif}' with {fps} FPS")



import cv2
import os

# Define the folder containing the frames and the output MP4 path
input_folder = "demo/A guy reading the news"  # Replace with the path to your folder
output_video = "demo/news_reading.mp4"             # Name of the output video
fps = 8                                # Frames per second

# Get a sorted list of image files in the folder
image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg'))])

# Select the first 50 images
selected_images = image_files[:32]

# Read the first image to get video dimensions
first_frame = cv2.imread(selected_images[0])
height, width, channels = first_frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each frame to the video
for image_file in selected_images:
    frame = cv2.imread(image_file)
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print(f"MP4 video saved as '{output_video}' with {fps} FPS")