
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("horoscope_saved.csv")

df.head(5)

df['date'] = pd.to_datetime(df['date'])
categories = df['category'].unique()
models = {}

for category in categories:
    category_data = df[df['category'] == category]

    X = category_data[['sign', 'date']]
    y = category_data['horoscope']

    vectorizer = CountVectorizer()
    X_text = vectorizer.fit_transform(X['sign'])  # Vectorize zodiac signs
    X_date = X['date'].dt.dayofyear.values.reshape(-1, 1)  # Extract day of the year as a numeric feature

    X_combined = pd.concat([pd.DataFrame(X_text.toarray()), pd.DataFrame(X_date)], axis=1)

    model = MultinomialNB()
    model.fit(X_combined, y)

    models[category] = (vectorizer, model)

user_sign = input("Enter your zodiac sign: ").lower()
user_date = pd.to_datetime(input("Enter the date (YYYY-MM-DD): "))

for category, (vectorizer, model) in models.items():
    user_sign_vectorized = vectorizer.transform([user_sign])

    user_dayofyear = user_date.dayofyear
    user_combined = pd.concat([pd.DataFrame(user_sign_vectorized.toarray()), pd.DataFrame([user_dayofyear])], axis=1)

    predicted_horoscope = model.predict(user_combined)

    print("Predicted horoscope for", user_sign.capitalize(), "on", user_date.strftime('%Y-%m-%d'), "in", category, "category:", predicted_horoscope[0])

