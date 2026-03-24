import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("spam.tsv", sep="\t", names=["label", "message"])

# convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# lowercase text
df['message'] = df['message'].str.lower()

# features and labels
X = df['message']
y = df['label']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# vectorization
vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# predict
y_pred = model.predict(X_test_vec)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# test custom message
# take user input
user_input = input("Enter a message: ")

# preprocess
user_input = user_input.lower()

# convert to vector
user_vec = vectorizer.transform([user_input])

# predict
prediction = model.predict(user_vec)

# output
if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Not Spam")
