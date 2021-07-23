import numpy as np
import pandas as pd

#reading the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t')
dataset

#text preprocessing
import re #regular expression
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

data = []
for i in range(0,1000):
    review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=''.join(review)
    data.append(review)

#data = ['wow place love','awesome place','wow awesome place']
# wow place love awesome
#awesome love place wow
#  0       1    1     1
#  1       0    1     0
#  1       0    1     1

#TF-IDF = 

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(data).toarray()

import pickle
pickle.dump(cv,open('cv.pkl','wb'))

y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#modeling building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#initialize the sequential model
model = Sequential()
#input layer
model.add(Dense(units=979,kernel_initializer='random_uniform',activation='relu'))
#hidden layer
model.add(Dense(units=1500,kernel_initializer='random_uniform',activation='relu'))
#output layers
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100)
model.save('NLP.h5')

y_pred = model.predict(x_test)

#random prediction
text = "Wow.. it was amazing tasty food"
review = re.sub('[^a-zA-Z]',' ',review)
review=review.lower()
review=review.split()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review=''.join(review)
y_p=model.predict(cv.transform([text]))
y_p=y_p>0.5
print(y_p)








