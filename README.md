# Fake-News-Detector
This project is a Deep Learning-based Fake News Detection System developed using TensorFlow, Keras, and a Gated Recurrent Unit (GRU) network.  The model analyzes textual news content and predicts whether the news article is Real or Fake. 
Overview

This project is a Deep Learning-based Fake News Detection System developed using TensorFlow, Keras, and a Gated Recurrent Unit (GRU) network.

The model analyzes textual news content and predicts whether the news article is Real or Fake. The system is trained on a large dataset of real and fake news articles and demonstrates a complete Natural Language Processing (NLP) workflow.

Features
Fake News Classification
Real News Classification
GRU-based Deep Learning Model
Text Tokenization
Sequence Padding
Word Embeddings
Model Persistence
User Input Prediction
Confidence Score Display
Image-Based Result Visualization
Dataset

Dataset Used:

Fake and Real News Dataset

Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset Files:

Fake.csv
True.csv

Columns:

title
text
subject
date

Labels:

0 → Fake News
1 → Real News
Project Workflow

Dataset
↓
Data Cleaning
↓
Tokenization
↓
Sequence Conversion
↓
Padding
↓
Embedding Layer
↓
GRU Layer
↓
Dense Layers
↓
Prediction

Model Architecture

Embedding Layer

Vocabulary Size: 10,000
Embedding Dimension: 128

GRU Layer

Hidden Units: 64

Dense Layer

32 Neurons
ReLU Activation

Output Layer

1 Neuron
Sigmoid Activation
Technologies Used
Python
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Scikit-Learn
Pickle
Installation

Install required packages:

pip install tensorflow pandas numpy matplotlib scikit-learn

Training

Run the training script:

python train.py

Outputs:

fake_news_gru.keras
tokenizer.pkl
Prediction

Run the prediction script:

python predict.py

Enter a news article when prompted.

Example:

Enter News Article:

Scientists discover a dragon living on Mars.

Output:

Prediction: Fake News

Confidence: 95.32%

Saved Files
fake_news_gru.keras

Stores:

Model Architecture
Trained Weights
Optimizer State
tokenizer.pkl

Stores:

Vocabulary Mapping
Word Index Information
Tokenization Configuration
Learning Objectives

This project helps understand:

Natural Language Processing
Text Preprocessing
Tokenization
Word Embeddings
Recurrent Neural Networks
Gated Recurrent Units (GRUs)
Binary Classification
Model Saving and Loading
Deep Learning Project Deployment Workflow
Future Improvements
Tkinter GUI Interface
Streamlit Web Application
Attention Mechanism
Bidirectional GRU
LSTM Comparison
Transformer-Based Classification
Real-Time News Scraping
Explainable AI Visualization
