Movie Genre Classification

Overview

This project classifies movies into genres based on textual plot descriptions. The dataset consists of movie plots and corresponding genre labels, processed to train a machine-learning model for classification.

Dataset

The dataset can be downloaded from Kaggle:
IMDB Genre Classification Dataset

Dataset Files

train_data.txt: Contains movie plots and their respective genres.

test_data.txt: Contains movie plots without genre labels (used for evaluation).

test_data_solution.txt: Contains actual genre labels for test data.

description.txt: Provides an overview of the dataset.

Project Workflow

Data Loading: Read the dataset files and parse movie plots and genre labels.

Preprocessing:

Convert text to lowercase.

Remove special characters and stopwords.

Tokenize words.

Feature Engineering: Convert text into numerical features using TF-IDF Vectorization.

Model Training: Train a multi-label classification model using Logistic Regression with OneVsRestClassifier.

Evaluation: Assess model performance using precision, recall, and F1-score.

Requirements

Ensure you have the following Python libraries installed:

pip install pandas numpy nltk scikit-learn

Additionally, download NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('punkt')

Running the Code

Place the dataset files in the appropriate directory.

Run the Python script to train and evaluate the model.

python movie_genre_classification.py

The script will print classification metrics to assess performance.

Future Improvements

Experiment with deep learning models (LSTMs, Transformers like BERT).

Use word embeddings (Word2Vec, GloVe) instead of TF-IDF.

Deploy the model using a web API for real-time classification.

Author

Developed by Prashanth Reddy.
