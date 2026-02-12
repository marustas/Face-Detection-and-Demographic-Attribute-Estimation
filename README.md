## Face Detection Module

### Overview

The face detection module is responsible for identifying and localizing human faces in images. This module serves as the first stage in the overall pipeline, enabling subsequent attribute prediction tasks such as age and gender classification.

The goal of this component is to evaluate and compare multiple face detection approaches in terms of detection accuracy, robustness, and computational efficiency.

The module supports multiple detectors and provides a unified interface for running experiments and benchmarking performance.

---

### Supported Face Detection Models

The following face detection models are implemented and evaluated:

#### 1. OpenCV Haar Cascade (Baseline)

This is a classical computer vision approach based on Haar-like features and a cascade classifier.

Characteristics:

- Very fast inference
- Low computational requirements
- No GPU required
- Lower detection accuracy compared to modern CNN-based methods
- Used as a baseline for comparison

Implementation:

```
src/models/face_detection/opencv.py
```

---

#### 2. MTCNN (Multi-task Cascaded Convolutional Networks)

This is a deep learning-based face detector consisting of a cascade of convolutional neural networks.

Characteristics:

- Higher accuracy than classical methods
- Robust to moderate pose variations
- Supports GPU acceleration
- Good balance between speed and performance

Implementation:

```
src/models/face_detection/mtcnn.py
```

Dependency:

```
facenet-pytorch
```

---

#### 3. RetinaFace

RetinaFace is a state-of-the-art single-stage face detector based on deep convolutional networks.

Characteristics:

- High detection accuracy
- Robust to occlusion, pose variation, and difficult lighting conditions
- Detects small and partially visible faces effectively
- Higher computational cost compared to other detectors

Implementation:

```
src/models/face_detection/retina.py
```

Dependency:

```
retina-face
```

---

### Dataset: WIDER FACE

Face detection performance is evaluated using the WIDER FACE dataset, which is a widely used benchmark for face detection.

Dataset properties:

- 32,000 images
- 393,000 annotated faces
- Real-world scenes with varying difficulty
- Includes small, occluded, and crowded faces

The dataset is automatically downloaded using torchvision and stored locally:

```
data/widerface/
```

Dataset loader implementation:

```
src/datasets/face_dataloader.py
```

---

### Evaluation Metrics

The following metrics are used to evaluate detector performance:

#### Precision

Measures the proportion of correctly detected faces among all detected faces.

```
Precision = TP / (TP + FP)
```

- OpenCV: 0.6629
- MTCNN: 0.9191
- RetinaFace: 0.9911

#### Recall

Measures the proportion of correctly detected faces among all ground truth faces.

```
Recall = TP / (TP + FN)
```

- OpenCV: 0.065
- MTCNN: 0.3326
- RetinaFace: 0.3187

#### Intersection over Union (IoU)

Measures overlap between predicted and ground truth bounding boxes.

```
IoU = Area of Overlap / Area of Union
```

#### Inference Time

Average time required to process one image.

- OpenCV: 0.0236
- MTCNN: 0.1359
- RetinaFace: 0.4555

### Running Detection Experiments

Detection experiments can be run using:

```
 python3 -m src.experiments.face_detector_experiments
```
