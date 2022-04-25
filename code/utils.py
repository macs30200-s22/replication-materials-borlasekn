import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) 
         if word not in stop_words] for doc in texts]

def lda_analysis(tweets_dict, num_topics=3):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'eng', 'nan', 'st', 'en', 'de',
                      'hhh', 'la', 'yo', 'spo', 'got'])
    data = tweets_dict['text']
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words, stop_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
    id2word=id2word,
    num_topics=num_topics, random_state=42)
    #Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    
    return lda_model, doc_lda, corpus, id2word

def gen_wordcloud(text_lst):
    # generate wordcloud
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # put text into wordcloud
    text = " ".join(text_lst)
    wordcloud.generate(text)
    # Visualize the word cloud
    return wordcloud