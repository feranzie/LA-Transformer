import cv2
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Capture a specific frame from a video.')
parser.add_argument('fr', type=int, help='Frame number to capture')

args = parser.parse_args()

frame_number = args.fr

# Variables

#video_file = 'C:/Users/DELL/Downloads/video_10.mp4'
video_file = '/notebooks/players/2output_video_with_bboxes.mp4'
#'/notebooks/LA-Transformer/3idtesttrack.mp4'

#video_file = 'C:/Users/DELL/Downloads/20240725T203000.mkv'
# frame_number = 1663
output_image = f'frames/output_frame{frame_number}.jpg'  # Frame you want to capture

# Open the video file
cap = cv2.VideoCapture(video_file)

# Set the frame position
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)



# Read the frame
ret, frame = cap.read()

if ret:
    # Annotate the frame number at the top-left corner
    cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the frame as an image
    cv2.imwrite(output_image, frame)
    print(f"Frame saved as {output_image}")
else:
    print("Failed to capture the frame.")

# Release the video capture object
cap.release()

