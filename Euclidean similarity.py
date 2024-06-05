import os
import re
import nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

nltk.download('punkt')

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
s
def remove_unwanted_characters(text):
    unwanted_chars = 'ء'
    arabic_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    arabic_text = re.sub(f'[{unwanted_chars}]', '', arabic_text)
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

# Progress bar magic
file_list = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
for filename in tqdm(file_list, desc='Processing files'):
    input_file_path = os.path.join(input_directory, filename)
    output_file_path = os.path.join(output_directory, filename)

    # Read, process, and write the text
    text = read_text_from_file(input_file_path)
    processed_text = process_arabic_text(text)
    write_text_to_file(processed_text, output_file_path)

    documents.append(processed_text)
    filenames.append(filename)

print("Text processing completed.")

def extract_leading_number(s):
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else float('inf')

sorted_indices = sorted(range(len(filenames)), key=lambda i: extract_leading_number(filenames[i]))
filenames = [filenames[i] for i in sorted_indices]
documents = [documents[i] for i in sorted_indices]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)


euclidean_dist = euclidean_distances(tfidf_matrix, tfidf_matrix)
euclidean_similarities = 1 / (1 + euclidean_dist)

# Visualization magic
plt.figure(figsize=(18, 15), dpi=200)
heatmap = sns.heatmap(euclidean_similarities, xticklabels=filenames, yticklabels=filenames, annot=True, cmap='coolwarm', annot_kws={"size": 6})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
plt.title('Euclidean similarity of Texts from Various Genres (7th to 21st Century) + Cosine Similarity Newscrawl-osian 2018/100k + whole corpus)', pad=40, fontsize=16)
plt.tight_layout(pad=2.0)
plt.show()