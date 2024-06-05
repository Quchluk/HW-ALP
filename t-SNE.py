import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

numeric_filenames = [(i, filenames[i]) for i in range(len(filenames)) if filenames[i][:4].isdigit()]
non_numeric_filenames = [(i, filenames[i]) for i in range(len(filenames)) if not filenames[i][:4].isdigit()]

sorted_numeric_indices = sorted(numeric_filenames, key=lambda x: int(x[1][:4]))
sorted_non_numeric_indices = sorted(non_numeric_filenames, key=lambda x: x[1])  # Sort non-numeric filenames alphabetically

sorted_indices = [i for i, _ in sorted_numeric_indices + sorted_non_numeric_indices]

filenames = [filenames[i] for i in sorted_indices]
documents = [documents[i] for i in sorted_indices]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

tsne = TSNE(n_components=2, random_state=42, perplexity=5)  # Set perplexity to 5
tsne_result = tsne.fit_transform(pca_result)

# Scatter plot of PCA results
plt.figure(figsize=(30, 20))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', label='Documents (PCA)')
for i, filename in enumerate(filenames):
    plt.annotate(filename, (pca_result[i, 0], pca_result[i, 1]), fontsize=15)  # Adjust annotation font size
plt.title('PCA Visualization of TF-IDF Vectors of Texts from Various Genres (7th to 21st Century)', fontsize=24)  # Adjust title font size
plt.xlabel('PCA Component 1', fontsize=20)  # Adjust x-label font size
plt.ylabel('PCA Component 2', fontsize=20)  # Adjust y-label font size
plt.legend(fontsize=18)  # Adjust legend font size
plt.show()

# Scatter plot of t-SNE results
plt.figure(figsize=(30, 20))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='green', label='Documents (t-SNE)')
for i, filename in enumerate(filenames):
    plt.annotate(filename, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=15)  # Adjust annotation font size
plt.title('t-SNE Visualization of TF-IDF Vectors of Cosine Similarity of Texts from Various Genres (7th to 21st Century)', fontsize=24)  # Adjust title font size
plt.xlabel('t-SNE Component 1', fontsize=20)  # Adjust x-label font size
plt.ylabel('t-SNE Component 2', fontsize=20)  # Adjust y-label font size
plt.legend(fontsize=18)  # Adjust legend font size
plt.show()