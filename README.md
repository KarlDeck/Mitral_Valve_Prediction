# Mitral Valve (MV) prediction on Echocardiography data using U-Nets

The mitral valve (MV) serves as a crucial component of the heart, directing circulation between the left atrium and left ventricle. It consists of two sections, the anterior and posterior leaflets, both connected to a structural ring called the mitral annulus. Under normal conditions, the left atrium contracts during diastole, allowing blood to pass through the open MV into the expanding left ventricle.

Extracting the MV from echocardiographic footage is a fundamental step in developing artificial intelligence models that can aid healthcare professionals in various tasks such as disease identification, treatment planning, and real-time surgical guidance. This project focuses on the segmentation of the MV in echocardiographic sequences.

## Objective
The dataset comprises 65 training videos, each containing three manually annotated frames marking the MV, along with a bounding box for localization. The test dataset includes 20 videos, where segmentation must be performed on every frame.

### Training Videos:
![TrainVideo](https://github.com/user-attachments/assets/11f9d5fc-f0c7-40a6-8104-8d100eb35c5b)

## Data Processing
After resizing the Videos the Training data is split into the expert-labeled data (later used for fine-tuning) and the amateur-labeled data. For the expert data a dataloader is used to augment the dataset because the size of the expert dataset is to small.
![image](https://github.com/user-attachments/assets/c5650d95-3c22-4136-92d8-f94c970a143f)

## Loss

### Intersection over Union (IoU) Loss:

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

with the predicted segmentation mask \( A \) and the ground truth mask \( B \)

The **IoU loss** is given by:

$$
\text{IoU Loss} = 1 - \text{IoU}
$$

## Training

For training a custom U-Net is used. The Model is first traind using the entire dataset in its original form and later finetuned using the augmented expert data
### 1:
```python
history = valve_model.fit(resized_array_videos, resized_array_masks, epochs=20, steps_per_epoch=80, validation_split=0.1)
```
### 2:
```python
## Activate fine tuning
for layer in valve_model.layers[:2]:
    layer.trainable = False
optimizer.learning_rate.assign(1e-5)

## Fine tune Model on augmented expert data
history = valve_model.fit(expert_train_generator, epochs=10, steps_per_epoch=80)
```

## Results

The results of the fully trained model used on the test set can be seen below. The predicted masks are overlayed onto the original (non-labeled) test videos.
### Example 1:
![output_video_8-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/bf9520a5-ff04-433c-9dcc-223357b331e2)
### Example 2:
![output_video_3-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/f4cc9678-d103-4f6b-8e21-5ea788d0dc2d)

## Acknowledgments
This Project was done together with [Niklas Britz (@Nibr1609)](https://github.com/nibr1609)
