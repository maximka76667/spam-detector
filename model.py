import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('spam.csv', usecols=["v1", "v2"], encoding='latin-1')
data.columns = ['label', 'message']
print(data)

# Separate into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2)

# Convert training message data into numeric values
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train a model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Get predictions
X_test_counts = vectorizer.transform(X_test)
predictions = model.predict(X_test_counts)

# Calculate accuracy score
print("Accuracy:", accuracy_score(y_test, predictions))