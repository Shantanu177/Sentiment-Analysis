from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

text = open('paragraph.txt',encoding="utf-8").read().lower()
clean_text = text.translate(str.maketrans('','',string.punctuation))

tokenized_words= word_tokenize(clean_text,"english") 

final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)


emotion_list=[]
with open('emotions.txt','r') as file:
    for line in file:
        line = line.strip().replace("'","").replace(",","")
        word , emotion = line.split(":")
        if word in final_words:
            emotion_list.append(emotion) 

# print(emotion_list)

w = Counter(emotion_list)
# print(w)
def Sentiment_Analyzer(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    # print(score)
    neg = score['neg']
    pos = score['pos']
    if neg> pos :
        print('Negative Sentiment')
    elif pos>neg:
        print('Positive Sentiment')
    else:
        print('Nuetral Vibe')

Sentiment_Analyzer(clean_text)

fig , ax1 = plt.subplots()
ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()