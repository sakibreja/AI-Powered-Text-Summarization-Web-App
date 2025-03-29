import streamlit as st
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Load the data

def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/tennis.csv')
    return df

# Text Summarization Function
def text_summarization(text):
    sentences = sent_tokenize(text)

    word_embeddings = {}
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vector = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vector.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vector[i].reshape(1, 100),
                                                   sentence_vector[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = ""
    for i in range(5):
        summary += ranked_sentences[i][1] + " "
    return summary

# Main Streamlit App
def main():
    st.title("Text Summarization")

    # Load data
    df = load_data()

    # Input
    input_text = st.text_area("Enter your text to summarize")

    # Button to summarize
    if st.button("Summarize Now"):
        if input_text:
            summary = text_summarization(input_text)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
