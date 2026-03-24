import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# NLTK Preprocessing Function
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    
    # Remove non-alphabet characters
    text = re.sub(r'[^a-z]', ' ', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords + stemming
    words = [ps.stem(word) for word in words if word not in stopwords_set]
    
    # Join back to string
    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

# Show cleaned samples
print("Sample cleaned messages:")
print(df['clean_text'].head())

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label_num']

print("\nTF-IDF vectorization complete")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Split data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom message prediction loop
while True:
    msg = input("\nEnter a message to check (or type 'exit' to quit): ")
    if msg.lower() == 'exit':
        break
    test_clean = preprocess(msg)
    vec = vectorizer.transform([test_clean])
    pred = model.predict(vec)[0]
    
    if pred == 0:
        print("Prediction: Ham")
    else:
        print("Prediction: Spam")
