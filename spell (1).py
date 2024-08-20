import os
import pandas as pd
# Importing all required libraries for this task.
import nltk
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
from itertools import chain
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import *
from nltk.corpus import wordnet as wn
import time
from tqdm import tqdm
from difflib import SequenceMatcher
import sys
from PyQt6.QtWidgets import QLabel,QApplication,QLineEdit,QLayout, QMainWindow, QTableView, QGridLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

lemmatizer = WordNetLemmatizer()


# -

import nltk


# +
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-grams Dashboard")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QGridLayout(central_widget)

        label = QLabel()
        label.setText(correct_mod)

        button1 = QPushButton("Generate correct sentence")
        button1.clicked.connect(correct_mod)
        layout.addWidget(button1)

        self.searchbar = QLineEdit()
        self.searchbar.setPlaceholderText("Enter text to correct")
        layout.addWidget(self.searchbar)
        self.searchbar.returnPressed.connect(correct_mod)

        data = self.searchbar.text

        self.canvas = None
        layout.addWidget(self.canvas)

    def generate_ngrams(self,n):
        if n == 'one':
            freq = unigram('me')
            self.canvas = plot_freq(freq)
            self.layout.addWidget(self.canvas)
        else:
            freq = bigram('you did')
            self.canvas = plot_freq(freq)
            self.layout.addWidget(self.canvas)
            
        
# -

def plot_freq(freq):
    fig, ax = plt.subplots()
    ax.bar(freq.keys(), freq.values())
    canvas = FigureCanvas(fig)
    return canvas


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# +
def parsing(sent):  
    """Parsing the sentence to corrected and original and storing in the dictionary."""
    loriginal = []
    lcorrected = []
    lcorr = []
    indexes = []
    cnt = 0
    
    for i in sent:
        if '|' in i:
            # Splitting the sentence on '|'
            str1 = i.split('|')
            # Previous word to '|' is storing in loriginal list.
            loriginal.append(str1[0])
            # Next word to '|' is storing in lcorrected list.
            lcorrected.append(str1[1])
            #Noting down the index of error.
            indexes.append(cnt)
        
        else:
            # If there is no '|' in sentence, sentence is stored in loriginal and lcorrected as it is.
            loriginal.append(i)
            lcorrected.append(i)
        cnt = cnt+1
        
    #Loading to loriginal, lcorrected and index list to dictionary.      
    dictionary = {'original': loriginal, 'corrected': lcorrected, 'indexes': indexes}
    
    return dictionary

def preprocessing():
    """Loading the data from 'holbrook.txt' and passing to parsing function to get parssed sentences. 
    Returning the whole dictionary as data."""
    avdata = []
    
    # Reading the txt file
    text_file = open("holbrook.txt", "r")
    lines = []
    for i in text_file:
        lines.append(i.strip())
    
    # Word tokenizing the sentences
    sentences = [nltk.word_tokenize(sent) for sent in lines]
    
    # Calling a parse function to get corrected, original sentences.
    for sent in sentences:
        avdata.append(parsing(sent))
    
    return avdata

# Calling preprocessing function
data = preprocessing()

# Testing
print(data[2])
assert(data[2] == {
 'original': ['I', 'have', 'four', 'in', 'my', 'Family', 'Dad', 'Mum', 'and', 'siter', '.'],
'corrected': ['I', 'have', 'four', 'in', 'my', 'Family', 'Dad', 'Mum', 'and', 'sister', '.'],
 'indexes': [9]
 })

# Splitting the data to test 100 lines and remaining training lines
test = data[:100]
train = data[100:]


# +
# Splitting the data to test - first 100 lines and remaining training lines
def test_train_split():
    """Splitting the data to test - first 100 lines and remaining training lines."""
    test = data[:100]
    train = data[100:]
    
    # Seperating the train original, test original, test corrected and train corrected from dictionary to list.
    train_corrected = [elem['corrected'] for elem in train]
    tokenizer = RegexpTokenizer(r'\w+')
    test_corrected = [elem['corrected'] for elem in test]
    test_original = [elem['original'] for elem in test]
    
    # Removing all special characters from the list.
    test_original = [tokenizer.tokenize(" ".join(elem)) for elem in test_original]
    test_corrected = [tokenizer.tokenize(" ".join(elem)) for elem in test_corrected]
    train_corrected = [tokenizer.tokenize(" ".join(elem)) for elem in train_corrected]
    
    return test_corrected, test_original, train_corrected

# Test and Training data.
test_corrected, test_original, train_corrected = test_train_split()


# +
def unigram(words):
    """This function returns a unigram frequency for a given word."""
    doc = []
    words = words.lower()
    for i in train_corrected:
        doc.append(" ".join(i).lower())

    doc = " ".join(doc)    
    doc = nltk.word_tokenize(doc)
    
    # Calculates frequency distribution.
    unig_freq = nltk.FreqDist(doc)
    
    # This gives word count - which is not used (for future modification)
    tnum_unig = sum(unig_freq.values())
    
    return unig_freq[words]

    
def bigram(words):
    """This function returns a bigram frequency for a given words."""
    doc = []
    
    # This function get words in string, hence converting string of 2 words to tuple.
    words = words.split(" ")
    words[0] = words[0].lower()
    words[1] = words[1].lower()
    words = tuple(words)
    
    for i in train_corrected:
        doc.append(" ".join(i))
        
    doc = " ".join(doc)    
    doc = doc.lower()
    
    #Calculating Bigrams for given words.
    tokens = nltk.wordpunct_tokenize(doc)
    bigrams=nltk.collocations.BigramCollocationFinder.from_words(tokens)
    biag_freq = dict(bigrams.ngram_fd)
    
    # This gives totla bigram count - which is not used (for future modification)
    tnum_bg = sum(biag_freq.values())
    
    # If there is no such bigram return 0
    try:
        return biag_freq[words]
    except KeyError:
        return 0, 0


# +
from nltk.metrics.distance import edit_distance

# Edit distance returns the number of changes to transform one word to another
#print(edit_distance("hello", "hi"))

# +
def get_candidates(token):
    
    """Get nearest word for a given incorrect word."""
    doc = []

    for i in train_corrected:
        doc.append(" ".join(i))

    doc = " ".join(doc)
    doc = nltk.word_tokenize(doc)
    unig_freq = nltk.FreqDist(doc)
    unique_words = list(unig_freq.keys())

    # Calculate distance between two words
    s = []
    for i in unique_words:
        t = edit_distance(i, token)
        s.append(t)
    
    # Store the nearest words in ordered dictionary
    dist = dict(zip(unique_words, s))
    dist_sorted = dict(sorted(dist.items(), key=lambda x:x[1]))
    minimal_dist = list(dist_sorted.values())[0]
    
    keys_min = list(filter(lambda k: dist_sorted[k] == minimal_dist, dist_sorted.keys()))
    
    return keys_min

#print(get_candidates("minde"))

# +
# This is to calculate unigram and bigram probabilities in correct function
doc = []

for i in train_corrected:
    doc.append(" ".join(i).lower())

doc = " ".join(doc)
doc = nltk.word_tokenize(doc)
unig_freq = nltk.FreqDist(doc)
unique_words = list(unig_freq.keys())

cf_biag = nltk.ConditionalFreqDist(nltk.bigrams(doc))
cf_biag = nltk.ConditionalProbDist(cf_biag, nltk.MLEProbDist)

# +
def tense(suggestion, sentence):    
    """Tense Detection"""
    tag = dict(nltk.pos_tag(sentence)).values()
    past_tense = ['VBN', 'VBD']
    conti_tense = ['VBG']
    
    # If sentence is past tense append ed and check if it is valid word
    if any(x in tag for x in past_tense):
        sug = []
        for a in suggestion:
            if a.lower()+'ed' in unique_words:
                sug.append(a+'ed')
        for aelem in sug:
            suggestion.append(aelem)
            
    # If sentence is past tense append ed and check if it is valid word
    if any(x in tag for x in conti_tense):
        sug = []
        for b in suggestion:
            if b.lower()+'ing' in unique_words:
                sug.append(b+'ing')
        for belem in sug:
            suggestion.append(belem)
        
    return suggestion

def named_entity(sentence):
    """Named Entity Detection using nltk.pos_tag and nltk.ne_chunk"""
    l = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(sentence)):
        # If any named tag like PERSON, ORGANIZATION or GEOLOCATION append to list.
          if hasattr(chunk, 'label'):
            l.append(' '.join(c[0] for c in chunk))
    
    if len(l) != 0:
        l = " ".join(l)
        l = l.split(" ")
        
    return l

#print(named_entity(['I', 'live', 'at', 'Boar', 'Parva', 'it', 'is', 'near', 'Melton', 'and', 'Bridgebrook', 'and', 'Smallerden']))


def word_forms_new(suggest):
    """Taking different forms of words using derivationally related forms"""
    sug_form = []
    for w in suggest:
        forms = set()
        for i in wn.lemmas(w):
            forms.add(i.name())
            for j in i.derivationally_related_forms():
                forms.add(j.name())
        
        for a in list(forms):
            sug_form.append(a)
    
    for q in sug_form:
        suggest.append(q)
    
    word_forms = []
    [word_forms.append(i) for i in suggest if not i in word_forms]
    return word_forms

def conditions(corrected, cr_ind):
    """Common word - Oclock is not detecting. Hence handling manually but not necessary"""
    
    if 'oclock' in corrected:
        ind = corrected.index('oclock')
        corrected = list(map(lambda x: str.replace(x, "oclock", "clock"), corrected))
        corrected.insert(ind, 'o')
        return corrected
        
    return corrected
word_forms_new(['wait', 'said', 'laid', 'paid', 'wad', 'waited'])

def sentence_sentence_similarity(sentence1):
    """Sentence - Sentence similarity using sequence matcher. We can also use cosine similarity but not implemented"""
    correc = []
    for d in train_corrected:
        ratio = SequenceMatcher(None, " ".join(d), " ".join(sentence1)).ratio()
        if ratio > 98:
            correc.append(d)
    
    if len(correc) == 1:
        return correc[0]
    else:
        return []

sentence_sentence_similarity(['1'])
# -

import nltk
nltk.download('averaged_perceptron_tagger')

import nltk
nltk.download('maxent_ne_chunker')

import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')


# +
def correct_mod(sentence):
    sts = ['oclock']
    corrected = []
    cnt = 0
    indexes = []
    #To check stemmed word in dictonary or not
    stemmer = PorterStemmer()
    status = 0
    #This will extract all named entities of a sentence
    n_en = named_entity(sentence)
    
    for i in sentence:
        # Check for sentence similarity
        corr = sentence_sentence_similarity(i)
        if len(corr) == 1:
            return corr
        # Ignoring digits like page number and lemmatizing the word and check if it is present in dictionary and use words.words() for word validation.
        elif i.lower() not in unique_words and not i.isdigit() and lemmatizer.lemmatize(i.lower()) not in unique_words and i not in n_en and i not in sts and i not in wn.words() and stemmer.stem(i) not in wn.words():
            indexes.append(cnt)
            if len(get_candidates(i)) > 1:
                # Get words forms, tense detection for suggested sentence
                suggestion = get_candidates(i)
                suggestion = tense(suggestion, sentence)
                wd_fms = word_forms_new(suggestion)
                suggestion = wd_fms

                prob = []
                
                # Bigram probabilities
                for sug in suggestion:

                    # Check the misspelled word is first or last word of the sentence
                    if ((cnt != 0) and (cnt != len(sentence)-1)):

                        try:
                            p1 = cf_biag[sug.lower()].prob(sentence[cnt+1].lower())
                            p2 = cf_biag[corrected[len(corrected)-1].lower()].prob(sug.lower())
                            p = p1 * p2
                            prob.append(p)   
                        except:
                            prob.append(0)
                    else:
                        #If mispelled word is last word of a sencence take probaility of previous word
                        if cnt == len(sentence)-1:
                            try:
                                p2 = cf_biag[corrected[len(corrected)-1].lower()].prob(sug.lower())
                                prob.append(p2)
                            except:
                                prob.append(0)


                        elif cnt == 0:
                            #If mispelled word is first word of a sencence take probaility of next word
                            try:
                                p1 = cf_biag[sug.lower()].prob(sentence[cnt+1].lower())
                                prob.append(p1)
                            except:
                                prob.append(0)
              
                if len(suggestion[prob.index(max(prob))]) > 1:
                    corrected.append(suggestion[prob.index(max(prob))])
                else:
                    corrected.append(suggestion[prob.index(max(prob))])

            else:
                corrected.append(get_candidates(i)[0])

        else:
            corrected.append(i)

        cnt = cnt+1
        # Manula hadling 'Oclock'
        corrected = conditions(corrected, indexes)
    
    fin = sentence_sentence_similarity(corrected)
    if len(fin) != 0:
        return fin
    else:
        return " ".join(corrected)



