Named-Entity-Recognition
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

We apply the same model as what is listed on Keras Documentaion.

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embeddin    from keras.layers import LSTM

    model = Sequential()
    model.add(Embedding(max_Features, output_dim=256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

We modify its entrance and exit only. Therefore it is a squential model with one-dimension input and 6-dimension output.

This framwork performed well on our training materials (accuracy = 100% when epoches = 5) thus there is no need for further modification unless we go to next step: transfer learning.

## 5. Demo Http Server
### 5.1 "bottle" Package with a Template 

Author: LJQ

Package: bottle

**Set up Your Server**

    python pyHttpServer.py function.py [port]

For example:

    python pyHttpServer.py NER_v4.py 11187

**Variables:**

     name: a string for the title of the webpage
     desc: a string for description of the webpage (below the title)
     examples：a list containing strings for demonstration/test/exhition
     port: an integer for dafault port (this variable can be omitted)

**Function Run(param):**

    param: a string from webpage
    return: return value of your task, where carriage return could be written as "\n"

**Notes: Variables needed initializing must be global variables.**

### 5.3 Jieba Manual Dictionary

We developeed a manual dictionary when generating the input set, including all of possible entity words that showed up in our corpus.

It is beneficial when cutting input sentences.
