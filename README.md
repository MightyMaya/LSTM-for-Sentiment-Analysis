# LSTM for Sentiment Analysis
###  *Using IMDB movie reviews*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1zgEkwcCkkZwmOhoivt9-0cMcRoVYSBQy?usp=sharing]

## Project Description 
Sentiment analysis on movie reviews using a Long Short-Term Memory (LSTM) model. The project contains 2 notebooks, one for training the model and one for directly using the trained model.

## Dataset
Dataset used is the IMDB dataset for movie reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Training 
1. Text is preprocessed by removing HTML tags, converting to lowercase, removing stop words, and lemmatizing.
2. Text is then tokenized and padded to fixed length sequences.
3. The labels are encoded as 0 or 1 for negative and positive sentiment, respectively.
4. The data is split into training and testing sets.
5. An LSTM model is built and trained on the training data.
6. The model is then saved and evaluated on the testing data
7. Finally, a function is defined to predict the sentiment of a given text.

