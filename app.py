import streamlit as st
import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# Load FAQ data
faq = pd.read_csv("faq.csv")

questions = faq['Question']
answers = faq['Answer']

# Convert text into vectors
vectorizer = CountVectorizer()

question_vectors = vectorizer.fit_transform(questions)

# Streamlit UI
st.title("🤖 AI FAQ Chatbot")

st.write("Ask any question related to the FAQ dataset.")

# User input
user_input = st.text_input("You:")

if st.button("Get Answer"):

    if user_input.strip() == "":
        st.warning("Please enter a question.")

    else:
        # Convert user question into vector
        user_vector = vectorizer.transform([user_input])

        # Calculate similarity
        similarity = cosine_similarity(user_vector, question_vectors)

        # Find best match
        best_match = similarity.argmax()

        response = answers[best_match]

        # Display response
        st.success("Chatbot Response:")
        st.write(response)