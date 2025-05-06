# NATURAL LANGUAGE PROCESSING CS6320- PROJECT 


![image](https://github.com/user-attachments/assets/95a831fc-37a4-40a6-b3d0-7bec0a0360e5)
### This is the demo of how the project should look.

# 📷 Image Caption Generator

This project focuses on generating descriptive captions for images using deep learning techniques. By combining Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks for sequence generation, the model generates natural language descriptions for images from the **Flickr8K dataset**.

## 🚀 Project Goal

The primary goal of this project is to automatically generate meaningful and contextually relevant captions for images. The model should be able to understand the visual content and produce natural language sentences that describe the image accurately.

##  Dataset

- **Flickr8K Dataset**: Consists of 8,000 images, each paired with five different captions provided by humans.
- The dataset helps train and evaluate the model's ability to generalize and generate descriptive sentences.

##  Project Architecture

### 1️⃣ Feature Extraction (CNN)

- Pretrained **VGG16** model is used.
- The fully connected layer outputs are extracted as feature vectors (4096 dimensions).
- These features represent the image content for the caption generator.

### 2️⃣ Caption Generation (LSTM)

- A tokenizer is used to convert text into sequences.
- The model consists of embedding layers, LSTM layers, and dense layers.
- The image features and text sequences are combined to predict the next word in the caption.

### 3️⃣ Training

- The model is trained using pairs of image features and captions.
- The objective is to minimize categorical cross-entropy loss between predicted words and actual words.


##  How to Run the Project

### Prerequisites

- Python 3.x
- TensorFlow / Keras
- Numpy
- NLTK
- tqdm
- PIL

Install required libraries:


## Steps

1. Extract Features

 Use VGG16 to preprocess images and save features.

2. Prepare Data

Clean the captions and tokenize them.

Prepare input-output sequences for the model.

3. Build and Train the Model

Combine extracted features and captions.

Train using LSTM architecture.

4. Generate Captions

Use the trained model to predict captions on new images.


## Difficulties Faced
- Handling large datasets and preprocessing
- Training time (4 hours) and requirement of powerful GPU’s.  
- Having less GPU power, we have to settle with less number of epochs.




##  Interesting Aspects

- I used personal photos of myself and friends to evaluate the results. 
- It was our first time using CNN to extract the image features. 




##  Lessons Learned
- Preprocessing of the data was extensive. 
- Fine-tuning models for better results.
- Integrating different AI domains (CV + NLP)


##  Future Work
Use Transformer-based models like Vision Transformers or ImageBERT for better captions.

Increase dataset size to avoid overfitting.

Deploy the model as a web or mobile app.

Implement an attention mechanism to improve caption relevance.

## Acknowledgements
Thanks to the creators of Flickr8K Dataset and the open-source deep learning frameworks.
