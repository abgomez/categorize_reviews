# Abel Gomez
# Categorize Hotel Reviews.
# Identify POS tag patterns that can be use to group and categorize text reviews.

import matplotlib.pyplot as plt
import itertools
import pandas
import numpy
import spacy
import nltk
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem  import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag_sents
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from collections import Counter
from itertools import islice
from nltk.tag import pos_tag
from spacy import displacy

"""
POS tag list:

    CC coordinating conjunction
    CD cardinal digit
    DT determiner
    EX existential there (like: "there is" ... think of it like "there exists")
    FW foreign word
    IN preposition/subordinating conjunction
    JJ adjective 'big'
    JJR adjective, comparative 'bigger'
    JJS adjective, superlative 'biggest'
    LS list marker 1)
    MD modal could, will
    NN noun, singular 'desk'
    NNS noun plural 'desks'
    NNP proper noun, singular 'Harrison'
    NNPS proper noun, plural 'Americans'
    PDT predeterminer 'all the kids'
    POS possessive ending parent's
    PRP personal pronoun I, he, she
    PRP$ possessive pronoun my, his, hers
    RB adverb very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP particle give up
    TO to go 'to' the store.
    UH interjection errrrrrrrm
    VB verb, base form take
    VBD verb, past tense took
    VBG verb, gerund/present participle taking
    VBN verb, past participle taken
    VBP verb, sing. present, non-3d take
    VBZ verb, 3rd person sing. present takes
    WDT wh-determiner which
    WP wh-pronoun who, what
    WP$ possessive wh-pronoun whose
    WRB wh-abverb where, when
"""

ps = PorterStemmer()
#chunking rules
#pattern = 'noun_phrase: {<DT>?<JJ>*<NN>}' #noun phrase
pattern = 'verb_phrase: {<RB.?>*<VB.?>*<NNP>+<NN>?}' #noun phrase

# tokenize text reviews.
def stemming_tokenizer(str_input):
    words = word_tokenize(str_input.lower())
    words = [word for word in words if word.isalpha() and len(word) > 3]
    words = [ps.stem(word) for word in words]
    return words

# count number of tags
def count_tags(text_tags):
    tag_count = {}
    for word, tag in text_tags:
        if tag in tag_count:
            tag_count[tag] += 1
        else:
            tag_count[tag] = 1
    return tag_count


#load reviews
#original = pandas.read_csv('truthful.csv', encoding="utf-8")
original = pandas.read_csv('deceptive.csv', encoding="utf-8")

###terms count
count_vectorizer = CountVectorizer(stop_words='english', tokenizer=stemming_tokenizer)
count = count_vectorizer.fit_transform(original.text)
truthful_count = pandas.DataFrame(count.toarray(), columns=count_vectorizer.get_feature_names())
###tfidf
tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=stemming_tokenizer, use_idf=True)
tf = tfidf_vectorizer.fit_transform(original.text)
truthful_tfidf = pandas.DataFrame(tf.toarray(), columns=tfidf_vectorizer.get_feature_names())

###get term frequency, we want to know what is commom what is not.
word_count = truthful_count.sum(axis = 0, skipna = True)
tfidf = pandas.DataFrame(truthful_tfidf.sum(axis = 0))
values = tfidf.values
tfidf.sort_values(0).head(20).plot(kind='bar')


###named entity recognition
tagged_text = original['text'].str.split().map(pos_tag)
tagged_text = pandas.DataFrame(tagged_text)
tagged_text['tag_counts'] = tagged_text['text'].map(count_tags)
tag_set = list(set([tag for tags in tagged_text['tag_counts'] for tag in tags]))
for tag in tag_set:
    tagged_text[tag] = tagged_text['tag_counts'].map(lambda x: x.get(tag, 0))
###find chunks
cp = nltk.RegexpParser(pattern)
chunked = []
for s in tagged_text['text']:
    chunked.append(cp.parse(s))
noun_rule_count = 0
for item in chunked:
    noun_rule_count += str(item).find('chunk')

