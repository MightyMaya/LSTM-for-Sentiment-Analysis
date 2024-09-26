# LSTM for Sentiment Analysis
###  *Using IMDB movie reviews*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/drive/folders/1zgEkwcCkkZwmOhoivt9-0cMcRoVYSBQy?usp=sharing]

## Project Description 
Sentiment analysis on movie reviews using a Long Short-Term Memory (LSTM) model. The project contains 2 notebooks, one for training the model and one for directly using the trained model.

## Dataset
Dataset used is the IMDB dataset for movie reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Required Libraries
```
tensorflow
nltk
pandas
numpy
sklearn
matplotlib
ipython
keras
```
## Training
1. Text is preprocessed by removing HTML tags, converting to lowercase, removing stop words, and lemmatizing.
```
def preprocess_text(text):
  # Remove HTML tags
  text = re.sub('<[^<]+?>', '', text)
  # Convert to lowercase
  text = text.lower()
  # Tokenize the text
  tokens = nltk.word_tokenize(text)
  # Remove stopwords and punctuation
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
  # Lemmatize the words
  tokens = [lemmatizer.lemmatize(token) for token in tokens]
  # Join the tokens back into a string
  processed_text = ' '.join(tokens)
  return processed_text
```
2. Text is then tokenized and padded to fixed length sequences.
```
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

Tokenizer=Tokenizer(num_words=5000)
Tokenizer.fit_on_texts(processed_text)
tokens = Tokenizer.texts_to_sequences(processed_text)

x = pad_sequences(tokens , maxlen=100)
```
3. The labels are encoded as 0 or 1 for negative and positive sentiment, respectively.
```
labels = reviews['sentiment']
encoded_labels = [1 if label == "positive" else 0 for label in labels]
```
4. An LSTM model is built and trained on the training data.
5. The model is then saved and evaluated on the testing data


