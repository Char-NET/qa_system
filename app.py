from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Load and process data
questions = []
answers = []

with open("data.txt", "r") as file:
    for line in file:
        if "|" in line:
            q, a = line.strip().split("|")
            questions.append(q)
            answers.append(a)

# Train vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Function to get best answer
def get_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, X)
    index = np.argmax(similarity)
    return answers[index]

# Routes
@app.route("/")
def home():
    return render_template("index.html", questions=random.sample(questions,5))

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json["question"]
    answer = get_answer(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)