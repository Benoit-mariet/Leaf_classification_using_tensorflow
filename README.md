# Tree Leaf Recognition Using Image Processing

## 1. Project Overview

This project aims to develop a Python-based system that accurately recognizes and classifies tree species using leaf images. Additionally, the system detects whether a leaf is healthy or diseased, enhancing its potential applications in ecology, forestry, and biodiversity conservation.

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

### Achievements:
- **Tree Species Classification Accuracy**: XX% on the test dataset.
- **Disease Detection Accuracy**: XX%.
- **Speed**: Average inference time per image: XX ms.
- **Memory Efficiency**: Model size: XX MB.

### Visualizations:
- **Accuracy Over Epochs**  
  ![Accuracy Graph](data/accuracy_graph.jpg)  


---

## 4. References and Documentation

This project uses the following technologies and libraries:
- [TensorFlow](https://www.tensorflow.org/) with MobileNetV2 architecture

Additional reading:
- [Image Classification with Deep Learning](https://www.tensorflow.org/tutorials/images/classification)

---

## 5. Issues and Contributions

### Known Issues:
- **Low-Quality Images**: Model accuracy decreases with blurry or noisy images.
- **Dataset Imbalance**: Limited samples for certain species or diseased leaves.

### How to Contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/new-feature





