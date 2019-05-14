import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from apyori import apriori
from itertools import chain
from timeit import default_timer as timer
import pyfpgrowth
stop_words = ['also', 'hotel', 'stay', 'stayed', 'room','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

#%% Dataset loading
data = []
with open('deceptive-opinion.csv') as file:
    data = file.readlines()

reviews = []
for i in range(1, len(data)):
    if data[i] == '"\n':
        continue
    reviews.append(data[i])

#%% Dataset preprocessing
ap = []
dec = []
gen = []
start = timer()
for i in range(len(reviews)):
    rev = reviews[i].split(',')
    deceptive = rev[0]
    hotel = rev[1]
    polarity = rev[2]
    source = rev[3]
    text = rev[4]
    arr = text.split(' ')

    good_words = []
    for word in arr:
        word = word.lower()
        word = word.strip('" \'.,')
        if len(word) < 2:
            continue
        if word in stop_words:
            continue

        good_words.append(word)

    if deceptive == 'deceptive':
        dec.append(good_words)
    else:
        gen.append(good_words)
        ap.append(good_words)

print("Preprocessing took ", timer() - start)
raise Exception

#%% FP-Growth (Genuine)
start = timer()
patterns = pyfpgrowth.find_frequent_patterns(gen, 15)
fp_rules_gen = pyfpgrowth.generate_association_rules(patterns, 0.7)
fp_rules_gen = sorted(fp_rules_gen.items(), key=lambda kv: kv[1][1], reverse=True)
print("fp-growth took ", timer() - start, "len rules", len(fp_rules_gen))

#%% FP-Growth (Deceptive)
start = timer()
patterns = pyfpgrowth.find_frequent_patterns(dec, 15)
fp_rules_dec = pyfpgrowth.generate_association_rules(patterns, 0.7)
fp_rules_dec = sorted(fp_rules_dec.items(), key=lambda kv: kv[1][1], reverse=True)
print("fp-growth took ", timer() - start, "len rules", len(fp_rules_dec))

#%% FP-Growth printing for report in Latex table format
fp_rules = fp_rules_gen
idx = 0
rules = 0
while idx < len(fp_rules) and rules < 15:
    lst = fp_rules[idx]
    lhs = lst[0]

    idx += 1
    if len(lhs) != 2: # Only rules of length 3 or fewer (without repetition)
        continue

    rhs = lst[1][0]
    confidence = lst[1][1]

    if len(rhs) == 0:
        continue

    rule = "%s->%s" % (lhs, rhs)
    print('%s & %.3f \\\ \hline' % (rule, confidence))
    rules += 1


#%% FP-Growth Printing
fp_rules = fp_rules_gen
for key in fp_rules.keys():
    # if len(key) != 2:
    #     continue
    val = fp_rules[key]
    r2 = val[0]
    confidence = val[1]
    print('%s->%s, %.3f' % (key, r2, confidence))

#%% Apriori (Genuine)
start = timer()
ap_rules_gen = list(apriori(gen, min_confidence=0.7, min_support=0.01))
print("apriori took ", timer() - start)

ap_rules_gen = sorted(ap_rules_gen, key=lambda x: x[2][0].confidence, reverse=True)

#%% Apriori (Deceptive)
start = timer()
ap_rules_dec = list(apriori(dec, min_confidence=0.7, min_support=0.01))
print("apriori took ", timer() - start)

ap_rules_dec = sorted(ap_rules_dec, key=lambda x: x[2][0].confidence, reverse=True)

#%% Ariori rule printing (for report) in Latex table format
# Rule, Support, Lift, Confidence
ap_rules = ap_rules_gen
for i in range(15):
    obj = ap_rules[i]
    pair = obj[0]
    items = [x for x in pair]
    if len(items) < 2:
        continue

    rule = "%s -> %s" % (items[0], items[1])
    support = obj.support
    lift = obj.ordered_statistics[0].lift
    confidence = obj.ordered_statistics[0].confidence
    print("%s & %.3f & %.3f & %.3f \\\ \hline" % (rule, support, lift, confidence))

#%% Apriori rule printing (only rules of length 2)
ap_rules = ap_rules_gen
for item in ap_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    if len(items) < 2:
        continue

    if item[2][0][2] < 0.8: # confidence
        continue
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")