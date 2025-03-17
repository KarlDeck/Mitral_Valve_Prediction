# Mitral Valve (MV) prediction on Echocardiography data using U-Nets

The mitral valve (MV) serves as a crucial component of the heart, directing circulation between the left atrium and left ventricle. It consists of two sections, the anterior and posterior leaflets, both connected to a structural ring called the mitral annulus. Under normal conditions, the left atrium contracts during diastole, allowing blood to pass through the open MV into the expanding left ventricle.

Extracting the MV from echocardiographic footage is a fundamental step in developing artificial intelligence models that can aid healthcare professionals in various tasks such as disease identification, treatment planning, and real-time surgical guidance. This project focuses on the segmentation of the MV in echocardiographic sequences.

## Objective
The dataset comprises 65 training videos, each containing three manually annotated frames marking the MV, along with a bounding box for localization. The test dataset includes 20 videos, where segmentation must be performed on every frame.

### Training Videos:
![TrainVideo](https://github.com/user-attachments/assets/11f9d5fc-f0c7-40a6-8104-8d100eb35c5b)

## Data Processing
After resizing the Videos the Training data is Augmented and loaded into a Dataloader
