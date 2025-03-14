import numpy as np
import cv2
import argparse
import os

# Global variables for drawing
drawing = False
current_x, current_y = -1, -1

def mouse(event, x, y, flags, param):
    global drawing, current_x, current_y, selection

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_x, current_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(frame_with_overlay, (current_x, current_y), (x, y), (0, 0, 255), 2)
            cv2.line(selection_mask, (current_x, current_y), (x, y), 255, 2)  # Update selection mask
            current_x, current_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Set up argument parser
parser = argparse.ArgumentParser(description="Play a video stored in a .npy file and overlay a mask (optional).")
parser.add_argument("filename", type=str, help="specify number of video")
args = parser.parse_args()

# Load the video data
try:
    video = np.load(os.path.join(os.getcwd(), "labeler/", f"{args.filename}", "video.npy"))
    mask = np.load(os.path.join(os.getcwd(), "labeler/", f"{args.filename}", "label.npy"))
    labeled_frames = np.load(os.path.join(os.getcwd(), "labeler/", f"{args.filename}", "existing_labels.npy"))
except Exception as e:
    print(f"Error loading file: {e}")
    exit(0)

# Validate video shape
if video.ndim != 3 or mask.ndim != 3 or video.shape != mask.shape:
    print("Invalid video or mask dimensions.")
    exit(0)

# Scale video and mask to 0-255 if necessary
if video.dtype != np.uint8:
    video = (video * 255 / np.max(video)).astype(np.uint8)

# Create OpenCV window
window_name = "Video with Mask"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(window_name, mouse)

# Initialize playback settings
current_frame = 0
selection = np.full(video.shape[:2], False)  # Selected pixels for current frame

# Create a temporary mask for drawing
selection_mask = np.zeros(video.shape[:2], dtype=np.uint8)

# Video playback loop
while current_frame < video.shape[2]:
    if current_frame in labeled_frames:
        current_frame += 1
        continue

    # Get current frame and overlay
    frame = video[:, :, current_frame]
    frame_with_overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to RGB

    # Overlay selection
    frame_with_overlay[selection_mask == 255] = [0, 0, 255]  # Red for selected pixels

    cv2.imshow(window_name, frame_with_overlay)

    # Handle keyboard input
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear current selection
        selection_mask[:] = 0
    elif key == ord('d'):  # Save and move to the next frame
        selection[:, :] = selection_mask > 0
        mask[:,:,current_frame] = selection_mask
        np.save(os.path.join(os.getcwd(), "labeler/", f"{args.filename}", "label.npy"), mask)
        print(f"Saved selection for frame {current_frame}")
        current_frame += 1
        selection_mask[:] = 0  # Reset mask for the next frame

cv2.destroyAllWindows()