#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, download
from nltk.tokenize import word_tokenize

# Ensure the NLTK data is downloaded
download('stopwords')
download('punkt')
download('averaged_perceptron_tagger')
download('wordnet')
download('omw-1.4')

# Define a function to clean and preprocess text
def clean_text(text):
    if isinstance(text, float):
        return ""  # Handle missing values by returning an empty string
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def get_wordnet_pos(treebank_tag):
    """Convert POS tag to a format recognized by the WordNet lemmatizer."""
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun

def preprocess_data(df, text_column):
    # Apply text cleaning
    df[text_column] = df[text_column].apply(clean_text)
    
    # Tokenization
    df['tokens'] = df[text_column].apply(word_tokenize)
    
    # POS tagging
    df['pos_tags'] = df['tokens'].apply(pos_tag)
    
    # Lemmatization and stopword removal
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def lemmatize_and_remove_stopwords(tokens_with_pos):
        lemmatized_tokens = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tokens_with_pos if word not in stop_words
        ]
        return lemmatized_tokens
    
    df['tokens'] = df['pos_tags'].apply(lemmatize_and_remove_stopwords)
    
    return df

# Load the datasets
train_df = pd.read_csv('samsum-train.csv')
valid_df = pd.read_csv('samsum-validation.csv')
test_df = pd.read_csv('samsum-test.csv')

# Print the column names to verify the text column name
print("Training Data Columns:", train_df.columns)
print("Validation Data Columns:", valid_df.columns)
print("Test Data Columns:", test_df.columns)

# Assuming the text column is named 'dialogue'
text_column = 'dialogue'

# Preprocess each dataset
train_df = preprocess_data(train_df, text_column)
valid_df = preprocess_data(valid_df, text_column)
test_df = preprocess_data(test_df, text_column)

# Save the cleaned data
train_df.to_csv('samsum_train_cleaned.csv', index=False)
valid_df.to_csv('samsum_valid_cleaned.csv', index=False)
test_df.to_csv('samsum_test_cleaned.csv', index=False)

# Print a sample of the cleaned data
print("Training Data Sample:")
print(train_df.head())
print("\nValidation Data Sample:")
print(valid_df.head())
print("\nTest Data Sample:")
print(test_df.head())

