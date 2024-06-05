import os
import re
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download('punkt')

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def remove_unwanted_characters(text):
    unwanted_chars = 'ء'  # Add more unwanted special characters here if needed
    arabic_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    arabic_text = re.sub(f'[{unwanted_chars}]', '', arabic_text)  # Remove unwanted special Arabic characters
    return arabic_text

def process_arabic_text(text):
    text = remove_unwanted_characters(text)
    text = text.strip()
    tokens = word_tokenize(text)
    stemmer = ISRIStemmer()
    lemmatized_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

input_directory = '/Users/Tosha/Desktop/Text counter/Text preparation/Output'
output_directory = '/Users/Tosha/Desktop/Text counter/Corpuses/Comparison visualisation Output'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

documents = []
filenames = []

file_list = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
for filename in tqdm(file_list, desc='Processing files'):
    input_file_path = os.path.join(input_directory, filename)
    output_file_path = os.path.join(output_directory, filename)
    text = read_text_from_file(input_file_path)
    processed_text = process_arabic_text(text)
    write_text_to_file(processed_text, output_file_path)

    documents.append(processed_text)
    filenames.append(filename)

print("Text processing completed.")

# Sorting texts
sorted_indices = sorted(range(len(filenames)), key=lambda i: int(filenames[i][:4]))
filenames = [filenames[i] for i in sorted_indices]
documents = [documents[i] for i in sorted_indices]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Visualization magic
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(cosine_similarities, xticklabels=filenames, yticklabels=filenames, annot=True, cmap='coolwarm')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.title('NEWS 2005-2022/300K Heatmap', pad=20)
plt.tight_layout(pad=2.0)
plt.show()