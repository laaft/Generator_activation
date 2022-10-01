# code contributed by: AMAN VASHISTHA (2019KUCP1011) for Assignment 2
# submitted to Dr. Basant Agarwal

# importing the libraries for preprocessing and importing 
# the dataset from the csv file
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# read the file named as data.csv
data = pd.read_csv('/data.csv')

# converting the data into small case
def preprocess_data(data):
    data['title'] = data['title'].str.strip().str.lower()
    return data

# preprocessing
data = preprocess_data(data)


# training and testing the data with test size 25%
x = data['title']
y = data['polarity']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

# transforming into the array
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

# creating the model and printing its score
model = MultinomialNB()
model.fit(x, y)
print("Score =", model.score(x_test, y_test))

# the sentences which will going to be predicted
predict_arr = [
    ['i was very amazed after movie'],
    ['i do not like this at all'],
    ['it was awesome in there!'],
    ['awful scenes!'],
    ['loved it, will go to watch it again!']
]

# predict the class of these sentences (0 for negative, 1 for positive)
for x in predict_arr:
  print(x," =",end="")
  print(model.predict(vec.transform(x)))
  