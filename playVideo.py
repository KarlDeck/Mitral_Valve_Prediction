import numpy as np
import cv2
import argparse
import os


def save_video_with_mask(video_filename, mask_filename=None, output_filename="output_video.mp4", fps=30):
    """
    Save a video with an optional mask overlay.
    
    Parameters:
        video_filename (str): Path to the .npy file containing the video.
        mask_filename (str): Path to the .npy file containing the mask (optional).
        output_filename (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    # Load the video
    try:
        video = np.load(os.path.join(os.getcwd(), "video", f"{video_filename}.npy"))
        if mask_filename:
            mask = np.load(os.path.join(os.getcwd(), "video", f"{mask_filename}.npy"))
        else:
            mask = None
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Validate video shape
    if video.ndim != 3:
        print("The video file must contain a valid 3D video array (expected shape: x, y, timeframes).")
        return

    # Validate mask if provided
    if mask is not None:
        if mask.ndim != 3:
            print("The mask file must contain a valid 3D video array (expected shape: x, y, timeframes).")
            return
        if video.shape != mask.shape:
            print("The video and mask must have the same dimensions.")
            return
        
        # Scale mask to 0-255 for OpenCV
        if mask.dtype != np.uint8:
            mask = (mask * 255 / np.max(mask)).astype(np.uint8)

        # Convert grayscale mask to RGB
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3, mask.shape[2]), dtype=np.uint8)
        for i in range(mask.shape[2]):
            mask_rgb[:, :, :, i] = np.stack([mask[:, :, i]] * 3, axis=-1)

    # Convert grayscale video to RGB
    rgb_video = np.zeros((video.shape[0], video.shape[1], 3, video.shape[2]), dtype=np.uint8)
    for i in range(video.shape[2]):
        rgb_video[:, :, :, i] = np.stack([video[:, :, i]] * 3, axis=-1)

    # Scale video to 0-255 if necessary
    if rgb_video.dtype != np.uint8:
        rgb_video = (rgb_video * 255 / np.max(rgb_video)).astype(np.uint8)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    frame_size = (rgb_video.shape[1], rgb_video.shape[0])  # (width, height)
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # Write frames to the output video
    for i in range(video.shape[2]):
        frame = rgb_video[:, :, :, i]
        if mask is not None:
            # Overlay mask with 50% transparency
            mask_frame = mask_rgb[:, :, :, i]
            mask_frame[:, :, 0] = 0  # Set red channel to 0
            mask_frame[:, :, 2] = 0  # Set blue channel to 0
            frame = cv2.addWeighted(frame, 0.5, mask_frame, 0.5, 0)

        # Write the frame
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved to {output_filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Play a video stored in a .npy file and overlay a mask (optional).")
    parser.add_argument("filename", type=str, help="Path to the .npy file containing the video.")
    parser.add_argument("mask_filename", type=str, nargs='?', default=None, help="Path to the .npy file containing the mask video (optional).")
    args = parser.parse_args()

    # Load the video data from the .npy file
    try:
        video = np.load(os.path.join(os.getcwd(), "video", f"{args.filename}.npy"))
        
        # Load mask only if provided
        if args.mask_filename:
            mask = np.load(os.path.join(os.getcwd(), "video", f"{args.mask_filename}.npy"))
        else:
            mask = None
            
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Validate video shape
    if video.ndim != 3:
        print("The video file must contain a valid 3D video array (expected shape: x, y, timeframes).")
        return

    # If mask is provided, validate its shape
    if mask is not None:
        if mask.ndim != 3:
            print("The mask file must contain a valid 3D video array (expected shape: x, y, timeframes).")
            return
        if video.shape != mask.shape:
            print("The video and mask must have the same dimensions.")
            return
        
        # Scale to 0-255 for OpenCV display if necessary
        if mask.dtype != np.uint8:
            mask = (mask * 255 / np.max(mask)).astype(np.uint8)

        # Create an empty array to store the RGB video with shape (112, 112, 3, 500)
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3, mask.shape[2]), dtype=np.uint8)

        # Convert grayscale to RGB by copying grayscale values across all 3 channels
        for i in range(mask.shape[2]):
            mask_rgb[:, :, :, i] = np.stack([mask[:, :, i]] * 3, axis=-1)

    # Create an empty array to store the RGB video with shape (112, 112, 3, 500)
    rgb_video = np.zeros((video.shape[0], video.shape[1], 3, video.shape[2]), dtype=np.uint8)

    # Convert grayscale to RGB by copying grayscale values across all 3 channels
    for i in range(video.shape[2]):
        rgb_video[:, :, :, i] = np.stack([video[:, :, i]] * 3, axis=-1)


    # Scale video to 0-255 if necessary
    if rgb_video.dtype != np.uint8:
        rgb_video = (rgb_video * 255 / np.max(rgb_video)).astype(np.uint8)

    # Create an OpenCV window with the exact dimensions of the array
    window_name = "Video with Mask"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, rgb_video.shape[1], rgb_video.shape[0])  # (width, height)

    # Video playback settings
    current_frame = 0
    paused = False

    # Play the video
    while current_frame < video.shape[2]:
        # Get current frame
        frame = rgb_video[:, :, :, current_frame]

        if mask is not None:
            # Overlay the mask with 50% transparency if mask is provided
            mask_frame = mask_rgb[:, :, :, current_frame]
            mask_frame[:, :, 0] = 0  # Set red channel to 0
            mask_frame[:, :, 2] = 0  # Set blue channel to 0
            frame = cv2.addWeighted(frame, 0.5, mask_frame, 0.5, 0)

        cv2.imshow(window_name, frame)

        # Wait for keypress and check for pause, resume, or navigation
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == ord('p'):  # Press 'p' to pause/resume
            paused = not paused
            quit = False
            while paused:  # Wait until 'p' is pressed again to resume
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p'):  # Press 'p' to resume
                    paused = False
                    break
                elif key == ord('d'):  # Press 'd' to move to the next frame
                    current_frame += 1
                    if current_frame >= video.shape[2]:  # Make sure it doesn't go out of bounds
                        current_frame = video.shape[2] - 1
                    frame = rgb_video[:, :, :, current_frame]
                    if mask is not None:
                        mask_frame = mask_rgb[:, :, :, current_frame]
                        mask_frame[:, :, 0] = 0  # Set red channel to 0
                        mask_frame[:, :, 2] = 0  # Set blue channel to 0
                        frame = cv2.addWeighted(frame, 0.5, mask_frame, 0.5, 0)
                    cv2.imshow(window_name, frame)
                elif key == ord('a'):  # Press 'a' to move to the previous frame
                    current_frame -= 1
                    if current_frame < 0:  # Make sure it doesn't go out of bounds
                        current_frame = 0
                    frame = rgb_video[:, :, :, current_frame]
                    if mask is not None:
                        mask_frame = mask_rgb[:, :, :, current_frame]
                        mask_frame[:, :, 0] = 0  # Set red channel to 0
                        mask_frame[:, :, 2] = 0  # Set blue channel to 0
                        frame = cv2.addWeighted(frame, 0.5, mask_frame, 0.5, 0)
                    cv2.imshow(window_name, frame)
                elif key == ord('r'):  # Press 'r' to restart the video
                    current_frame = 0
                    paused = False
                    break
                if key == ord('q'):  # Press 'q' to quit
                    quit = True
                    break
            if quit:
                break
        elif key == ord('r'):  # Press 'r' to restart the video
            current_frame = 0  # Restart the video from the first frame
        elif not paused:
            current_frame += 1  # Increment the frame if not paused

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()