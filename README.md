# 🖼️ CIFAR-10 Image Classification using MLP and CNN
### Deep Learning applied to Image Classification with PyTorch

## 📌 Project Overview

This project was developed as part of a university project.  
The goal was to explore and compare two neural network architectures for image classification using the **CIFAR-10 dataset**:

- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**

The main objective was to improve classification performance through architecture adjustments, activation function experiments, dropout, and hyperparameter optimization.

The models were implemented using **PyTorch**, with performance evaluated through accuracy, balanced accuracy, precision, recall, and F1-score.

---

## 👥 Team Members

- **Almir Alves**
- **Erbert Gadelha**
- **Felipe Barreto**
- **João Pedro**
- **Vinicius Barbosa**

---

## 📦 Dataset

The project uses the **CIFAR-10 dataset**, which contains 60,000 color images distributed across 10 classes.

Each image has dimensions of 32 × 32 pixels and 3 RGB channels.

- Training set: 50.000 images
- Test set: 10.000 images
- Number of classes: 10
- Input size: 3.072 features (32 × 32 × 3)

The classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset: [CIFAR-10 – University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

---

# 🧠 MLP Approach (19/05/2025 – 02/06/2025)

## ⚙️ Initial Hyperparameters

The initial MLP implementation used the following configuration:

| Hyperparameter | Value |
|---|---:|
| Input size | 3.072 |
| Number of classes | 10 |
| Learning rate | 0.0001 |
| Number of epochs | 100 |
| Batch size | 16 |

## 🏗️ Network Architecture

The initial architecture consisted of fully connected layers:

- **Input layer:** 3.072 → 64 neurons
- **Hidden layer 1:** 64 → 128 neurons
- **Hidden layer 2:** 128 → 64 neurons
- **Output layer:** 10 classes

The following activation and loss functions were used:

- **ELU:** selected to mitigate the dead neuron problem by allowing smoother negative activations.
- **CrossEntropyLoss:** measures the difference between predicted class probabilities and the actual labels.

## 🔧 Architecture Improvements

An additional hidden layer was introduced to increase the model's learning capacity.

The updated architecture was:

```python
self.fc_input = nn.Linear(64 * 8 * 8, 128)
self.fc_hidden1 = nn.Linear(128, 256)
self.fc_hidden2 = nn.Linear(256, 128)
self.fc_hidden3 = nn.Linear(128, 64)
self.fc_output = nn.Linear(64, 10)
```

### Dropout

Dropout was subsequently implemented after the internal layers to reduce overfitting.

The selected dropout rates were:

- Dropout 1: 0.2
- Dropout 2: 0.4
- Dropout 3: 0.6

This configuration was tested to improve generalization during training.

The updated architecture was:

```python
self.fc_input = nn.Linear(64 * 8 * 8, 128)
self.dropout1 = nn.Dropout(p=0.2)
self.fc_hidden1 = nn.Linear(128, 256)
self.dropout2 = nn.Dropout(p=0.4)
self.fc_hidden2 = nn.Linear(256, 128)
self.dropout3 = nn.Dropout(p=0.6)
self.fc_hidden3 = nn.Linear(128, 64)
self.fc_output = nn.Linear(64, 10)
```

## 📊 Final MLP Evaluation

The final MLP evaluation produced the following metrics:

| Metric | Result |
|---|---:|
| Accuracy | 0.5314 |
| Balanced Accuracy | 0.5314 |
| Precision | 0.535149982317983 |
| Recall | 0.5314 |
| F1-score | 0.5312550135782537 |

These results were calculated using the test dataset.

---

# 🧠 CNN Approach (04/06/2025 - 16/06/2025)

## ⚙️ Initial Hyperparameters

The initial CNN configuration was:

| Hyperparameter | Value |
|---|---:|
| Input size | 3.072 |
| Number of classes | 10 |
| Learning rate | 0.001 |
| Number of epochs | 10 |
| Batch size | 32 |

## 🏗️ Initial Network Architecture

The initial CNN consisted of two convolutional layers followed by fully connected layers.

### 🔹 Convolutional Layers

**1. First convolutional layer**

```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

- Receives 3 RGB channels.
- Applies 32 filters of size 3 × 3.
- Extracts initial image features.
- After max pooling, produces feature maps of size 32 × 16 × 16.

**2. Second convolutional layer**

```python
self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
```

- Receives 32 feature maps.
- Applies 64 filters of size 3 × 3.
- Extracts more complex visual patterns.
- After max pooling, produces feature maps of size 64 × 8 × 8.

**3. Max Pooling**

```python
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

Reduces the spatial dimensions by half by retaining the maximum value within each 2 × 2 region.

### 🔹 Fully Connected Layers

```python
self.fc_hidden1 = nn.Linear(64 * 8 * 8, 120)
self.fc_hidden2 = nn.Linear(120, 84)
self.fc_output = nn.Linear(84, 10)
```

The fully connected layers process the extracted features and generate predictions for the ten CIFAR-10 classes.

### Activation and Loss Functions

- **ReLU:** introduces non-linearity into the network.
- **CrossEntropyLoss:** measures the difference between predicted probabilities and actual class labels.

## 📊 Initial CNN Evaluation

Using the initial configuration, the CNN achieved an average final loss of 0.1874 after approximately 2 minutes and 40 seconds of execution.

| Metric | Result |
|---|---:|
| Accuracy | 0.6991 |
| Balanced Accuracy | 0.6991 |
| Precision | 0.7065 |
| Recall | 0.6991 |
| F1-score | 0.6998 |

The initial CNN already achieved significantly better results than the MLP.

---

## 🔬 CNN Optimization and Experiments

Several experiments were conducted to improve the CNN's performance.

### 1️⃣ Activation Function Comparison

Two alternative activation functions were evaluated: LeakyReLU and ELU.

**LeakyReLU**

LeakyReLU addresses the dead neuron problem by allowing a small, non-zero gradient for negative inputs.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7276 |
| Balanced Accuracy | 0.7276 |
| Precision | 0.7281 |
| Recall | 0.7276 |
| F1-score | 0.7265 |

Average loss: 0.1335.

**ELU**

ELU uses an exponential function for negative inputs, providing smoother activations.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7070 |
| Balanced Accuracy | 0.7070 |
| Precision | 0.7103 |
| Recall | 0.7070 |
| F1-score | 0.7076 |

Average loss: 0.1476.

### 2️⃣ Additional Hidden Layer

An additional hidden layer was also tested:

```python
self.fc_input = nn.Linear(64 * 8 * 8, 128)
self.fc_hidden1 = nn.Linear(128, 256)
self.fc_hidden2 = nn.Linear(256, 128)
self.fc_hidden3 = nn.Linear(128, 64)
self.fc_output = nn.Linear(64, 10)
```

The results were:

| Activation | Average Loss | Accuracy |
|---|---:|---:|
| LeakyReLU | 0.1650 | 0.7231 |
| ELU | 0.1829 | 0.7095 |

Based on these experiments, the original fully connected structure was retained, together with LeakyReLU, as it produced better evaluation metrics.

### 3️⃣ Dropout

Dropout was introduced after the internal fully connected layers to reduce overfitting.

The selected configuration was:

```python
self.fc_hidden1 = nn.Linear(64 * 8 * 8, 120)
self.dropout1 = nn.Dropout(p=0.25)

self.fc_hidden2 = nn.Linear(120, 84)
self.dropout2 = nn.Dropout(p=0.15)

self.fc_output = nn.Linear(84, 10)
```

Results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7442 |
| Balanced Accuracy | 0.7442 |
| Precision | 0.7415 |
| Recall | 0.7442 |
| F1-score | 0.7418 |

### 4️⃣ Increasing the Number of Epochs

Increasing the number of epochs allows the model to train for longer and potentially learn more complex representations.

To help prevent overfitting, additional techniques were considered:

- **Data Augmentation:** artificially increases the diversity of training samples.
- **Learning Rate Scheduler:** automatically adjusts the learning rate throughout training.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 0.8022 |
| Balanced Accuracy | 0.8022 |
| Precision | 0.8036 |
| Recall | 0.8022 |
| F1-score | 0.8008 |

Average training loss: 0.6822.

### 5️⃣ Batch Normalization and Global Average Pooling

Two additional techniques were implemented:

- **Batch Normalization:** normalizes intermediate activations to help stabilize and accelerate training.
- **Global Average Pooling:** reduces the number of parameters by averaging feature maps before classification.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 0.8063 |
| Balanced Accuracy | 0.8063 |
| Precision | 0.8074 |
| Recall | 0.8063 |
| F1-score | 0.8056 |

Average loss: 0.5415.

---

# 🔍 Hyperparameter Optimization – Grid Search

For the final experiments, the CNN architecture was further developed by adding convolutional and fully connected layers.

A Grid Search was performed to compare different optimizers and learning rates.

## ⚙️ Search Space

| Hyperparameter | Option 1 | Option 2 |
|---|---|---|
| Optimizer | Adam | SGD |
| Learning rate | 0.001 | 0.01 |

The four possible combinations were evaluated.

## 📊 Grid Search Results

| Optimizer | Learning Rate | Accuracy | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| Adam | 0.001 | **0.8252** | 0.8252 | 0.8300 | 0.8252 | 0.8234 |
| Adam | 0.01 | 0.8034 | 0.8034 | 0.8244 | 0.8034 | 0.8033 |
| SGD | 0.001 | 0.7597 | 0.7597 | 0.7627 | 0.7597 | 0.7563 |
| SGD | 0.01 | 0.8138 | 0.8138 | 0.8155 | 0.8138 | 0.8111 |

## 🏆 Best Hyperparameters Found

The best configuration found during the Grid Search was:

- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Accuracy:** 82.52%

This combination achieved the highest accuracy among the four tested configurations.

### 📈 Grid Search Model Comparison

<div align="center">
  <img width="550" height="800 alt="image" src="https://github.com/user-attachments/assets/3d8ae5bf-c0c6-4739-a19d-d498edb16caf" />
</div>

---

# 📊 Model Evaluation and Visualizations

In addition to the evaluation metrics, visualizations were generated to better understand the classification performance of both models.

## 🔹 Confusion Matrices

Confusion matrices provide a detailed view of the predictions made by each model, showing the relationship between actual and predicted classes.

### MLP Confusion Matrix

<div align="center">
  <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/1b6a50d7-f832-4905-aaf6-89e3f344919f" />
</div>

### CNN Confusion Matrix

<div align="center">
  <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/5f18e9aa-295e-474f-9d22-8bfe4ee0ac38" />

</div>

## 🔹 Correct Predictions per Image

These graphs show whether each individual test image was correctly classified.

Each point represents one test image:

- **1:** Correct prediction.
- **0:** Incorrect prediction.

This visualization makes it possible to compare the distribution of correct and incorrect predictions across the test dataset.

### MLP – Correct Predictions per Image

<div align="center">
  <img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/78422c7d-d7f1-4d56-b230-5d65a8d0f2b4" />
</div>

### CNN – Correct Predictions per Image

<div align="center">
  <img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/f5544e4b-040c-4568-b107-6035f4d69be3" />
</div>

---

# ⚖️ MLP vs. CNN

The experiments demonstrate the advantages of convolutional architectures for image classification.

The final MLP achieved an accuracy of **53.14%**, while the best CNN configuration obtained an accuracy of **82.52%**.

| Model | Accuracy | F1-score |
|---|---:|---:|
| MLP | 53.14% | 53.13% |
| CNN | **82.52%** | **82.34%** |

The CNN performed better because convolutional layers are designed to extract spatial features from images, preserving local relationships between pixels, while an MLP processes flattened images, losing their explicit spatial structure.

---

## 🧰 Libraries Used

- `torch` – Neural network implementation and model training
- `torchvision` – CIFAR-10 dataset loading and image transformations
- `numpy` – Numerical operations
- `pandas` – Data manipulation
- `matplotlib` – Data visualization and model evaluation graphs
- `seaborn` – Statistical data visualization
- `scikit-learn` – Evaluation metrics and confusion matrices
---

## 🔗 Final Notes

This project highlights the importance of neural network architecture selection, training strategies, and hyperparameter optimization in deep learning.

By comparing an MLP and a CNN, it was observed how convolutional layers can significantly improve image classification performance by learning spatial representations directly from images.

📌 I invite you to explore the full repository and dive deeper into the implementation.
