:

📩 SMS Spam Detection using Machine Learning

This project is a Machine Learning-based SMS Spam Detection system built using Python and Natural Language Processing (NLP). It is designed to classify SMS messages into two categories: Spam and Ham (Not Spam).

📌 Project Overview

Spam messages are a major issue in modern communication. Manually identifying them can be time-consuming and inefficient.
This project provides an automated solution by applying NLP techniques and a machine learning model to accurately detect spam messages.

✨ Key Features
Text preprocessing using NLP techniques
Conversion of text to lowercase for uniformity
Removal of stopwords and unnecessary characters
Word stemming for better text normalization
Feature extraction using TF-IDF Vectorization
Classification using Multinomial Naive Bayes
Model evaluation with accuracy and classification report
Real-time prediction through user input
🛠️ Technologies Used
Python
Pandas
NLTK
Scikit-learn
⚙️ Working Process
Load and clean the SMS dataset
Preprocess the text (tokenization, stopword removal, stemming)
Convert text data into numerical features using TF-IDF
Train the model using Multinomial Naive Bayes
Evaluate model performance on test data
Predict whether new messages are spam or ham
▶️ How to Run the Project
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Run the Program
python main.py
Step 3: Test the Model
Enter any SMS message when prompted
The system will classify it as Spam or Ham
📂 Dataset

The project uses a publicly available SMS dataset containing labeled messages:

Spam → Unwanted or promotional messages
Ham → Normal messages
📊 Model Evaluation

The model is evaluated using:

Accuracy Score
Precision, Recall, and F1-Score (via classification report)
🎯 Learning Outcomes
Understanding the basics of Natural Language Processing (NLP)
Applying TF-IDF for feature extraction
Building and evaluating a machine learning model
Improving Python programming and data preprocessing skills
👤 Author

Yash Jain
B.Tech Computer Science Engineering Student
