# Tree Leaf Recognition Using Image Processing

## 1. Project Overview

This project aims to develop a Python-based system that accurately recognizes and classifies tree species using leaf images. Additionally, the system detects whether a leaf is healthy or diseased, enhancing its potential applications in ecology, forestry, and biodiversity conservation.

This is the 11 species of leaves the model can recognize : 
Mango,
Arjun, 
Alstonia Scholaris, 
Gauva,
Jamun, 
Jatropha,
Pongamia Pinnata, 
Basil, 
Pomegranate, 
Lemon, 
Chinar,

To train my model, I used this dataset, which contains images of healthy and diseased leaves of these species : https://drive.google.com/file/d/1UyOY0bVCd_K3uIrHd-9mTMubu8ZoGyB5/view?usp=drive_link 

### Key Objectives:
- **Tree Species Identification**: Classify leaves into corresponding species.
- **Disease Detection**: Differentiate between healthy and diseased leaves.
- **Accessibility**: Provide a tool for both experts and non-experts in tree identification.

### Use Cases:
- **Ecological Research**: Fast species identification for biodiversity studies.
- **Forest Management**: Monitor environmental health by detecting diseased leaves.
- **Educational Tool**: Assist students and researchers in learning tree species identification.

---

## 2. Source Code

### Project Structure:

### Project Structure

- **/project-root**  
  - **/model**: File containig the pre-trained model
      - **modele_feuille.h5**: Model  pre-trained
  - **/code**: contains the different code
      - **training_code.py**: it's the code used for trained the model
      - **application_code.py**: it's the main code using the pre-trained model to make the tree leaf recognition
  - **requirements.txt**: Python dependencies  
  - **README.md**: Project documentation  




---

## 3. Performance Metrics

### For the training of the model (training_code)

#### Achievements:
- **Tree Species Classification and Disease Detection Accuracy**: 82% on the test dataset.
- **Speed**: Total time of the training: ~ 4 hours (24 minutes per Epochs).
- **RAM used**: ~ 5GB for the training
- **Memory Efficiency**: Model size: 11 Mo.

#### Visualizations:
- **Accuracy Over Epochs**  
  ![Accuracy Graph](data/accuracy_graph.png)
- **Loss Over Epochs**  
  ![Loss Graph](data/loss_graph.png)  


### For the use of the model (application_code)

#### Achievements:
- **Speed**: Total time of the execution of the code for 1 image: ~ 15 seconds.
- **RAM used**: ~ 5GB 
- **Memory Efficiency**: Model size: 11 Mo.


---

## 4. Installation and Usages

### Step 1: Required Installations

#### 1. Download the Necessary Files
- Clone or download the files from the GitHub repository.
- **Dataset Download (for training your own model):** [Dataset](https://drive.google.com/file/d/1UyOY0bVCd_K3uIrHd-9mTMubu8ZoGyB5/view?usp=drive_link)
- Ensure you have a compatible Python environment (e.g., Spyder or Anaconda).

#### 2. Library Installation

##### To Use `application_code` (Tree Leaf Recognition Code):
- **TensorFlow:**  
  Run `pip install tensorflow` in the terminal or `conda install tensorflow` if using Anaconda.
- **NumPy** (if not pre-installed):  
  Run `pip install numpy` in the terminal or `conda install numpy` if using Anaconda.

##### To Train Your Own Model with `training_code`:
- **TensorFlow:** Same as above.
- **Psutil:**  
  Run `pip install psutil` in the terminal or `conda install psutil` if using Anaconda.
- **Time:** Already pre-installed with Python.
- **Pandas:** Usually pre-installed with Python.

---

### Step 2: Code Modifications

Both `application_code` and `training_code` require access to specific file paths on your computer.  
You must update the paths in the code (indicated by comments: **# !!!!!! Change the path accordingly !!!!!!**) with the correct file locations on your system.

---

### Step 3: Usage Instructions

Once you’ve correctly updated the paths and installed all necessary libraries, you can run the program:  

1. For `application_code`, ensure the path of the test image is correct.  
2. The program will identify the species among 11 possible options and indicate whether the leaf is healthy or diseased.  
   - **D**: Diseased  
   - **H**: Healthy

Launch the program, and it will display the predicted species and health status of the leaf.


---

## 5. References and Documentation

This project uses the following technologies and libraries:
- [TensorFlow](https://www.tensorflow.org/) with MobileNetV2 architecture

Additional reading:
- [Image Classification with Deep Learning](https://www.tensorflow.org/tutorials/images/classification)

Original Dataset (modifications have been made) : 
-[Plant_leaves_dataset](https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification/data)

---

## 6. Issues and Contributions

### Known Issues:
- **Low-Quality Images**: Model accuracy decreases with blurry or noisy images.
- **Dataset Imbalance**: Limited samples for certain species or diseased leaves.
- **Lack of image variety**: The photos are all taken in the same environnement.
- **Speed of the code**: Both training and application can be limiting with their time of execution (non-industrial usage).

### How to Contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/new-feature

---

## 7. Future Work

### Potential Improvements: 
- **Expand Dataset**: Include more species and additional leaf conditions to improve model generalization.
- **Optimize Model**: Reduce the model's size and memory usage for faster performance on lower-spec devices, including mobile deployment.
- **Real-time Detection**: Implement a real-time detection system using a camera feed to identify species and health status on the go.
- **User Interface Development**: Create a user-friendly GUI or mobile app for easier interaction with the model, making it more accessible to non-technical users.
- **Explainability**: Implement visualization techniques to explain the model's decisions, helping users understand why a particular species or disease status was predicted.

These improvements aim to broaden the system's usability, enhance its performance, and expand its applicability in diverse fields.




