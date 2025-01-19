import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# step no 2
reviews = ["the product is excellent and works perfectly",
           "the product is not good, very disoppointed",
           "terrible product and waste of money",
           "I love this product and it is amazing"]
sentiments = np.array([1,0,0,1])
# step no 3
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)
# step no 4
classifier = MultinomialNB()
classifier.fit(x, sentiments)
# step no 5
def classify_new_review(review):
    review_vectorized = vectorizer.transform([review])
    prediction = classifier.predict(review_vectorized)
    if prediction[0] == 1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"
# step no 6
user_review = input("Enter your review: ")
result = classify_new_review(user_review)
print(f"The review '{user_review}' is classified as '{result}'")
