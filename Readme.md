#Contains NLP algos to perform
1. Generate Text Summary
    - fetch sentence similarity score based on sentence vectors generated using:
        -- Word frequency
        -- Word embeddings
2. Find relations in the data - To be Done
3. Decode the reference - To be Done
4. Generate fill in the blank questions from text content - To be Done
5. Predict next possible word (n-gram) - To be Done
6. Create keyword filter - To be Done

Base logic:
1. Text wrangling
2. POS tagging
3. Lemmatize
4. NER

#All algos based on NLTK
1. Use nltk.download('puntk') - Tokenizer library
2. Use nlt.download('stopwords') - we'll use list of english stopwords only
3. Use nltk.download('wordnet') - for lemmatizer
4. Use nltk.download('averaged_perceptron_tagger') - for POS

#Fetch word embeddings from:
http://nlp.stanford.edu/data/glove.6B.zip (822MB)
