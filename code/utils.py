import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import nltk
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models
from collections import Counter
from nltk.stem import WordNetLemmatizer
import numpy as np


lemmatizer = WordNetLemmatizer()
SID = SentimentIntensityAnalyzer()

def sentiment_score(text):
    return SID.polarity_scores(text)['compound']

def sentiment_comparison(df_1, df_2):
    # Sentiment Analysis
    fm = df_1['text'].apply(sentiment_score)
    # Sentiment Analysis
    fd = df_2['text'].apply(sentiment_score)
    return fm, fd

def get_topic_keywords(lda_model, texts):
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in texts for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])     
    return df

def get_frequent_hashtags(data):
    n = 30
    full_words = data['hashtags']
    full_words = list(full_words)
    full_words = [lemmatizer.lemmatize(i) for i in full_words if isinstance(i, str) and i != " "]
    full_words = " ".join(full_words)
    full_words = full_words.split(" ")
    full_words = [w for w in full_words if w]

    word_list = full_words
    counts = dict(Counter(word_list).most_common(n))
    labels, values = zip(*counts.items())

    labels = [i for i in reversed(labels)]
    values = [i for i in reversed(values)]

    indexes = np.arange(len(labels))
    
    return indexes, values, labels

def word_frequency(data):
    n = 30
    full_words = data['text']
    full_words = list(full_words)
    full_words = [lemmatizer.lemmatize(i) for i in full_words if isinstance(i, str)]
    full_words = " ".join(full_words)
    full_words = full_words.split(" ")
    stop = stopwords.words('english') + ['…', "it's", "like", 'im', '1st', '30', 'retweet', 'ki', 'it’s', 'you’re'
                                        'they’re', 'i’m', '\u2063', "they're", "“they’re", 'they’re', 'choday', 'get', 'got',
                                        'h', 'ea', '12']
    full_words = [w for w in full_words if w not in stop and w]

    word_list = full_words
    counts = dict(Counter(word_list).most_common(n))
    labels, values = zip(*counts.items())

    labels = [i for i in reversed(labels)]
    values = [i for i in reversed(values)]

    indexes = np.arange(len(labels))
    
    return indexes, values, labels

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) 
         if word not in stop_words] for doc in texts]

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def get_dominant_topics(df_topic_sents_keywords, lda_model, corpus):
    t0_coord = []
    t1_coord = []
    t2_coord = []
    for i in lda_model[corpus]:
        t0_coord.append(i[0][1])
        t1_coord.append(i[1][1])
        t2_coord.append(i[2][1])

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic['T0 Coords'] = t0_coord
    df_dominant_topic['T1 Coords'] = t1_coord
    df_dominant_topic['T2 Coords'] = t2_coord
    
    return df_dominant_topic

def lda_analysis(tweets_df, num_topics=3):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'eng', 'nan'])
 
    data = tweets_df['text'].values.tolist()
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
     # Print the Keyword in the 10 topics
     #pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return lda_model, doc_lda, corpus, id2word, texts

def gen_wordcloud(text_lst):
    # generate wordcloud
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # put text into wordcloud
    text = " ".join(text_lst)
    wordcloud.generate(text)
    # Visualize the word cloud
    return wordcloud