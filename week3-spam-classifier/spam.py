import nltk

from nltk.corpus import stopwords

import glob
import re

from collections import Counter 
import os
f = os.listdir("ex03-data/spam-train")
# All files and directories ending with .txt and that don't begin with a dot:
file_spam= glob.glob("ex03-data/spam-train/*.spam.txt")
#print(f"these are spam files \n{file_spam}")
stop_words= set(stopwords.words('english'))
spam_language = set()
spam_total_counter= Counter()
def read_spam_msg(directory):
    file_spam = glob.glob(f"{directory}/*.spam.txt")
    for file in file_spam:
        with open(file, "r", encoding='latin-1') as f:  # Use 'with' for safe file handling
            file_content = f.read()

        words= re.findall(r'\b\w+', file_content.lower())
        for word in words:
            if word not in stop_words:
                spam_language.add(word)
                spam_total_counter.update([word])


read_spam_msg("ex03-data/spam-train")

#print("\nVocabulary (spam_language):\n", spam_language)

def average_frequency_spam(spam_language, spam_total_counter):
    avg_frq_dict= {}
    no_words_spam= sum(spam_total_counter.values())
    for word in spam_language:
        avg_freq = spam_total_counter[word]/no_words_spam
        avg_frq_dict[word]= avg_freq
    return avg_frq_dict

    

dict_spam=average_frequency_spam(spam_language, spam_total_counter)
print(dict_spam)



ham_language = set()
ham_total_counter= Counter()
file_ham = glob.glob("ex03-data/spam-train/*.ham.txt")
for file in file_ham:
    with open(file, "r", encoding="ISO-8859-1") as f:  # Use 'with' for safe file handling
        file_content = f.read()

    each_file = Counter()
    words= re.findall(r'\b\w+', file_content.lower())
    ham_language.update(words)
    ham_total_counter.update(words)

no_words_ham= sum(ham_total_counter.values()) 




#print("\nVocabulary (spam_language):\n", spam_language)
#print("\nVocabulary (ham_sentences):\n", ham_total_counter)

#print("\nWord counts in each file (spam_sentences):\n", spam_sentences)
#print("\nWord counts in each file (ham_sentences):\n", ham_sentences)