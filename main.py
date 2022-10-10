#Google's speech recognition 
import speech_recognition as sr 

#nltk package(natural language toolkit)
import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
from nltk.tokenize import WordPunctTokenizer


import string
import os
import time

#flair model for sentiment nalysis
from flair.models import TextClassifier
from flair.data import Sentence

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

# Create tokenizer and stemmer
tokenizer =WordPunctTokenizer()


def flair_prediction(x):
    sia = TextClassifier.load('en-sentiment')
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"

def match(a, b):
    tokens_a = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    #accuracy
    ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))

    return (ratio*100)

#predefined paragraph
a="Our Team Data Pirates is here to demonstrate working of code snippet that shows accuracy speed and sentiment analysis of a speech which we made  after completing the main task of creating a wireframe and Powerpoint presentation"

r = sr.Recognizer()
# read the audio data from the default microphone
with sr.Microphone() as source:
 r.adjust_for_ambient_noise(source)
 print("start")
 start=time.time()
 audio_data = r.listen(source)
 end=time.time()
 
 #recognizing audio
 print("Recognizing...")
 try:
  b=r.recognize_google(audio_data,language="en-US")
  print(b)
 except sr.UnknownValueError:
   print('Unable to recognize the audio')
 except sr.RequestError as e: 
   print("Request error from Google Speech Recognition service; {}".format(e))

#words in speech
tokens_c=[token.lower().strip(string.punctuation) for token in tokenizer.tokenize(b) \
                    if token.lower().strip(string.punctuation)]

duration_m=(end-start)/60 #minutes
words=len(tokens_c) #number of words
speed=(words/duration_m) #wpm


print("ACCURACY: " + str(match(a,b))) #accuracy

print("TIME IN MINUTES: " + str(duration_m)) #time
print("WORDS IN SPEECH: " + str(words)) #words

print("SPEED OF SPEECH: " + str(speed)) #speed
print("SENTIMENT ANALYSIS: " + (flair_prediction(b))) #sentiment analysis


