# Genuine and Deceptive Hotel Reviews
The openness of the platform allows people to give their review without taking in consideration the impact that the review could have. Due to this openness and the amount of information that platform holds it is impossible to verify every single review, this causes an issue of misinformation. Not all the reviews can be considered trustful, deceptive users can take advantage of the platform and they can influence buyer's decision.  The impact of misinformation could be both positive and negative. Even in a positive impact, misinformation is bad for the consumer and it should be separate from trustful reviews. Since review platforms handles large amount of data, data mining techniques are required to clean up the and eliminate bad reviews.


## Dataset
Our model focus in a genuine and deceptive hotel reviews corpus available at [kaggle](https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus). The dataset has five columns 'deceptive', 'hotel', 'polarity', 'source' and 'text'. In total, there are 1600 reviews where half of them are truthful and the remaining half are deceptive. The reviews are about 20 hotels; 80 reviews for each of them. There are 800 positive and 800 negative reviews as labeled in the column 'polarity'. The use of this dataset will allow us to easily categorize the components of each type of reviews.

## Methodology
We implement text extraction aproaches based in standard data mining techniqiues useful to parse and process unstructured text. Our model utilize Information Retrieval (IR), Natural Language Processing (NLP), and Information Extraction from text (IE) to gather high quality information required to define patterns in text reviews.

## Project Files
To process the unstructured text reviews, our model implements different python libraries. The following describes in a high level implementation in our model.

[chunking.py](https://github.com/abgomez/categorize_reviews/blob/master/chunking.py): The goal of chunking.py is to identify stand alone tags and group them with a syntax analysis to generate phrases such noun, verbs, and adjective phrases. To identify the groups and the association of tags, chunking implements syntax rules. Chunking rules have specific purpose and features, our model consider two rules.  A noun phrase rules expressed as `{<DT>?<JJ>*<NN>}` and a verb/adverb phrase rule expressed as `{<RB.?>*<VB.?>*<NNP>+<NN>`.

Complete detail of the categorize model can be found at [Reviews Model](https://sites.google.com/view/dataminingspring2019/home)
