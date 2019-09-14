#Contains algos to perform
1. Generate Text Summary
2. Generate fill in the blank questions from text content
3. Find relations in the data
4. Decode the reference
5. Predict next possible word (n-gram)
6. Create keyword filter    


#Main steps involved
1. Tokenization (word, sentence)
2. lemma    
3. NER
4. POS tagging

#All algos based on NLTK
1. Use nltk.download('puntk') - Tokenizer library
2. Use nlt.download('stopwords') - we'll use list of english stopwords only
3. Use nltk.download('wordnet') - for lemmatizer
4. Use nltk.download('averaged_perceptron_tagger') - for POS

#Fetch word embeddings from:
http://nlp.stanford.edu/data/glove.6B.zip (822MB)
