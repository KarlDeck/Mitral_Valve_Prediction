from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import gzip
import numpy as np
import os
import pandas as pd

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def preprocess_train_data(data):
    video_frames = []
    mask_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=np.bool_)
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            names.append(name)
    return names, video_frames, mask_frames

def custom_preprocess_train_data(data):
    video_frames = []
    mask_frames = []
    roi_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=np.bool_)
        roi = np.zeros((height, width, n_frames), dtype=np.bool_)
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            roi[:, :, frame] = item['box'][:, :]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
            roi_frame = roi[:, :, frame]
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)
            roi_frame = np.expand_dims(roi_frame, axis=2).astype(np.int32)
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            roi_frames.append(roi_frame)
            names.append(name)
    return names, video_frames, mask_frames, roi_frames

def preprocess_test_data(data):
    video_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        video_frames += list(video)
        names += [item['name'] for _ in video]
    return names, video_frames

def get_sequences(arr):
    first_indices, last_indices, lengths = [], [], []
    n, i = len(arr), 0
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths

def export_video(video, name):
    np.save(str(os.getcwd()) + '/video/' + name + '.npy', video)

def export_video_to_labeler(index, video, mask, labels):
    np.save(str(os.getcwd()) + '/labeler/' + str(index) + "/video.npy", video)
    np.save(str(os.getcwd()) + '/labeler/' + str(index) + "/label.npy", mask)
    np.save(str(os.getcwd()) + '/labeler/' + str(index) + "/existing_labels.npy", labels)


import cv2
import numpy as np

def resize_image(image, target_width, target_height, original_width, original_height):
    # Scale factor
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Calculate padding
    pad_width = target_width - new_width
    pad_height = target_height - new_height
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # Pad the image
    return cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)  # Black padding
    )

def resize_mask_image(image, target_width, target_height, original_width, original_height):
    if image is None:
        raise ValueError("Input image is None.")
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original dimensions must be positive integers.")
    
    scale = min(target_width / original_width, target_height / original_height)
    if scale <= 0:
        raise ValueError("Scale factor must be greater than 0.")
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    if new_width <= 0 or new_height <= 0:
        raise ValueError("New dimensions must be positive integers.")
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        image = image.astype(np.uint8)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    pad_width = target_width - new_width
    pad_height = target_height - new_height
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # Pad the mask
    return cv2.copyMakeBorder(
        resized_image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0,)  # Black padding
    )

def undo_resize_image(resized_image, original_width, original_height, target_width, target_height):
    """
    Undo resizing and padding for an image.

    Parameters:
    - resized_image: The resized and padded image (NumPy array).
    - original_width: Original width of the image.
    - original_height: Original height of the image.
    - target_width: Target width of the resized image (including padding).
    - target_height: Target height of the resized image (including padding).

    Returns:
    - Original image restored to original dimensions.
    """
    # Scale factor used in the original resizing
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Calculate padding amounts
    pad_width = target_width - new_width
    pad_height = target_height - new_height
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Remove padding
    cropped_image = resized_image[top:target_height-bottom, left:target_width-right]

    # Resize back to original dimensions
    original_image = cv2.resize(cropped_image, (original_width, original_height))

    return original_image


def undo_resize_mask_image(resized_mask, original_width, original_height, target_width, target_height):
    """
    Undo resizing and padding for a mask.

    Parameters:
    - resized_mask: The resized and padded mask (NumPy array).
    - original_width: Original width of the mask.
    - original_height: Original height of the mask.
    - target_width: Target width of the resized mask (including padding).
    - target_height: Target height of the resized mask (including padding).

    Returns:
    - Original mask restored to original dimensions.
    """
    # Scale factor used in the original resizing
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Calculate padding amounts
    pad_width = target_width - new_width
    pad_height = target_height - new_height
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Remove padding
    cropped_mask = resized_mask[top:target_height-bottom, left:target_width-right]

    cropped_mask_uint8 = cropped_mask.astype('uint8')

    original_mask = cv2.resize(cropped_mask_uint8, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    return original_mask.astype(bool)



def resize_videos_with_padding_scale(videos, masks=None, target_size=None, scale=False):
    if target_size == None:
        target_size = (256,256)
    target_width, target_height = target_size
    resized_videos = list()
    resized_masks = list()

    if masks:
        assert len(videos) == len(masks)

    for i in range(len(videos)):
        video = videos[i]
        if masks:
            mask = masks[i]
            assert videos[i].shape[:2] == masks[i].shape[:2]
        # Original dimensions
        original_height, original_width = video.shape[:2]
        
        resized_video = np.zeros((target_width, target_height, video.shape[2]))
        if masks:
            resized_mask = np.zeros((target_width, target_height, mask.shape[2]))

        # go through frames
        for i in range(video.shape[2]):
            # Video
            image = video[:, :, i]
            resized_image = resize_image(image, target_width, target_height, original_width, original_height)
            resized_video[:, :, i] = resized_image

            # Mask
            if masks:
                mask_image = mask[:, :, i]
                resized_mask_image = resize_mask_image(mask_image, target_width, target_height, original_width, original_height)
                resized_mask[:, :, i] = resized_mask_image

        if scale:
            resized_video = resized_video / 255.0

        resized_videos.append(resized_video)
        if masks:
            resized_masks.append(resized_mask)
    
    return resized_videos, resized_masks



def undo_resize_videos_with_padding(
    resized_videos, original_sizes, resized_masks=None, scale=False
):
    original_videos = []
    original_masks = []

    for idx, resized_video in enumerate(resized_videos):
        original_height, original_width = original_sizes[idx]
        video = np.zeros((original_height, original_width, resized_video.shape[2]))

        # Reverse scaling if applied
        if scale:
            resized_video = resized_video * 255.0

        for i in range(resized_video.shape[2]):
            frame = resized_video[:, :, i]
            video[:, :, i] = undo_resize_image(frame, original_width, original_height, resized_video.shape[1], resized_video.shape[0])

        original_videos.append(video)

        # Handle masks if provided
        if resized_masks:
            resized_mask = resized_masks[idx]
            mask = np.zeros((original_height, original_width, resized_mask.shape[2]))

            for i in range(resized_mask.shape[2]):
                mask_frame = resized_mask[:, :, i]
                mask[:, :, i] = undo_resize_mask_image(mask_frame, original_width, original_height, resized_mask.shape[1], resized_mask.shape[0])

            original_masks.append(mask)

    return original_videos, original_masks






def calculate_masked_image(image, mask):
    image_rgb = np.repeat(image, 3, axis=-1)
    
    # Convert the mask from (n, n, 1) to (n, n, 3) where mask is green (0, 255, 0) and other areas are (0, 0, 0)
    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[mask.squeeze() == 1] = [0, 1, 0]  # Green color for mask
    
    # Apply 50% transparency to the mask
    alpha = 0.5
    final_image = image_rgb * (1 - alpha) + mask_rgb * alpha
    
    # Clip the result to ensure valid values (0-1 for float images)
    final_image = np.clip(final_image, 0, 1)
    
    return final_image