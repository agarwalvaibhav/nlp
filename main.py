#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.cluster.util import cosine_distance
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import summarize
import numpy as np
import argparse as ap
import re
import networkx as nx

def sentence_similarity_word_frequency(sent1, sent2):
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

# Extract word vectors
def get_word_embeddings():
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def get_sentence_vectors(sentences, word_embeddings):
    #create sentence vector
    sentence_vectors = []
    for i in sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

def pos_tagging(sentences):
    pos = [pos_tag(word_tokenize(s)) for s in sentences]
    return pos

def get_wordnet_pos(pos):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(lemma, pos_sentence, stop_words):
    lwords = list()
    for w in pos_sentence:
        if w[0] in stop_words or not w[0].isalpha():
            continue
        lwords.extend([w[0], lemma.lemmatize(w[0], pos=get_wordnet_pos(w[1]))])
    return list(set(lwords))

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    sm1 = np.zeros((len(sentences), len(sentences)))
    sm2 = np.zeros([len(sentences), len(sentences)])

    #create sentence vector using word_embeddings
    word_embeddings = get_word_embeddings()
    sentence_vectors = get_sentence_vectors(sentences, word_embeddings)

    #generate lemma words to generate similarity index based on word frequency
    lemma = WordNetLemmatizer()
    pos_sentences = pos_tagging(sentences)
    lemma_sentences = [lemmatize(lemma, s, stop_words) for s in pos_sentences]

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j: #ignore if both are same sentences
                continue
            sm1[i][j] = sentence_similarity_word_frequency(sentences[i], sentences[j])

            #based on word embeddings
            sm2[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    return [sm1, sm2]

def remove_stopwords(sentence, stop_words=[]):
    words = word_tokenize(sentence)
    sen = " ".join([i for i in words if i.isalpha() and i not in stop_words])
    return sen

def generate_summary(sentences, n=50):
    top_n = int(n/100*len(sentences))
    stop_words = stopwords.words('english')
    clean_sentences = [remove_stopwords(s, stop_words) for s in sentences]

    #Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(clean_sentences, stop_words)

    for sm in sentence_similarity_martix:
        summarize_text = []

        #Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sm)
        scores = nx.pagerank(sentence_similarity_graph)
        
        #Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
        for i in range(top_n):
            summarize_text.append(ranked_sentence[i][1])
        
        #output the summarize text
        print("\nSummary:\n\n", "\n".join(summarize_text))
    return


def read_article(file_name):
    file = open(file_name, "r")
    article_text = file.readlines()[0]

    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    sentences = sent_tokenize(article_text)
    words = word_tokenize(article_text)

    return words, sentences

if __name__ == '__main__':
    """ Capture below inputs from the user
    """
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--top_n", type=int, help="%age summary",
                        choices = range(10, 101, 10), default=50)
    parser.add_argument("-i", "--input", type=str, help="input text file name",
                        default='input.txt')
    # parse the arguments received
    args = parser.parse_args()
    n = args.top_n
    ip = args.input
    
    #Read text and split it
    words, sentences =  read_article(ip)

    #summary using prebuilt util
    print("\nSummary using gensim:\n")
    print(summarize((' ').join(sentences), ratio=n/100))
    generate_summary( sentences, n)
