# Named-Entity-Recognition
Chinese NER Demo Project Basing on LSTM and Aiming at General Concepts

## 1. Corpus Resource
We applied tagged Chinese corpus, PFR corpus provided at http://www.icl.pku.edu.cn

PFR corpus is made from the text of People's Daily (first half of year 1998). Each word in the article is marked with its part of speech. The corpus has now 26 basic word class tags (noun n, time t, place s, position f, number m, quantifier q, distinguishing word b, pronoun r, verb v, adjective a, state word z , adverb d, preposition p, conjunction c, auxiliary word u, modal word y, interjection e, fictitious words o, idiom i, idiom l, abbreviation j, the former component h, the latter component k, morpheme g, etc.)

Additionally, from the perspective of application, it has **4 more special tags**: Name nr, place ns, orgnization nt, other proper noun nz.

## 2. Format of Input Set (Independent Variable of Model)

It is a vector with 1 dimension only, generated from the corpus.

Through a python dictionary *(MAPS_Word_to_Index and MAPS_Word_to_POS)*, every single Chinese word from the corpus corresponds to *its par of speech* with **an distinct integer**.

**Hence, the input vector would be a vector with every integer ranging from 0 to *len(MAPS_Word_to_Index)*.**

Any other character corresponds to *"unknown" tag* with **a certain integer (to be more specific, it would be the number of known word +1)**.

## 3. Format of Output Set (Dependent Variable of Model)

It is a matrix with 6 dimensions:

    Column 0 means "Not an Entity"
    Column 1 means "Orgnization"
    Column 2 means "Place"
    Column 3 means "Other Entities"
    Column 4 means "Name of Someone"
    Column 5 means "Abbreviation"


Only **0 and 1** are used for building this matrix, 0 representinng "no" and 1 "yes".


## 4. Framwork of LSTM Neural Network

TBC...


## 5. Demo Http Server
### 5.1 "Bottle" Package with a Template 

TBC...

### 5.3 Jieba Manual Dictionary

We developeed a manual dictionary when generating the input set, including all of possible entity words that showed up in our corpus.

It is beneficial when cutting input sentences.

TBC...

### 5.3

TBC...
