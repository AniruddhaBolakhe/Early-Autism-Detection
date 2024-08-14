import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips

def split_video_into_segments(input_dir, output_dir, segment_duration=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other video formats if needed
            video_path = os.path.join(input_dir, filename)
            video = VideoFileClip(video_path)
            video_duration = int(video.duration)
            
            # Split the video into segments
            for start_time in range(0, video_duration, segment_duration):
                end_time = min(start_time + segment_duration, video_duration)
                segment = video.subclip(start_time, end_time)
                
                # Create a filename for the segment
                segment_filename = f"{os.path.splitext(filename)[0]}_part{start_time // segment_duration + 1}.mp4"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Write the segment to the output directory
                segment.write_videofile(segment_path, codec="libx264", audio_codec="aac")
                
            video.close()

# Define the input and output directories
input_directory = "C:/Users/User/Desktop/sensory"
output_directory = "C:/Users/User/Desktop/output_sensory"

# Call the function to split videos into 10-second segments
split_video_into_segments(input_directory, output_directory)
