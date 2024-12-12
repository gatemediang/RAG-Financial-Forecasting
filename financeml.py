from flask import Flask, request, jsonify
import os
import pandas as pd
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from constants import *


# Initialize Gemini
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.0-pro")

openai_api_key = OPEN_KEY

# Fix: Initialize Flask app
app = Flask(__name__)

# Fix: Initialize embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

file_name = "all_stocks_5yr.csv"


def load_and_process_csv(file_name):
    data = pd.read_csv(file_name)
    data["text"] = data.apply(
        lambda row: f"Stock {row['Name']} on date {row['date']} opening price {row['open']} closing price {row['close']}.",
        axis=1,
    )
    texts = data["text"].tolist()
    return texts


def create_and_store_embeddings(texts):
    vectors = embeddings.embed_documents(texts)
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def retrieve(query, vector_store, k=20):
    return vector_store.similarity_search(query, k=k)


def askquestion(question, vectorstore):
    topdocs = retrieve(question, vectorstore)
    top_contexts = [doc.page_content for doc in topdocs]
    top_context = " ".join(top_contexts)

    prompttemplate = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert question answering system. I'll give you a question and context, and you'll return the answer.
        Context: {context}
        Query: {question}""",
    )
    argumented_prompt = prompttemplate.format(context=top_context, question=question)

    response = model.generate_content(argumented_prompt)
    return response.text


# Endpoint for handling question answering requests
@app.route("/api/question-answering", methods=["POST"])
def question_answering():
    try:
        # Extract data from the request
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data.get("question")

        # Load and process data from CSV
        if not os.path.exists(file_name):
            return jsonify({"error": f"CSV file {file_name} not found"}), 500

        texts = load_and_process_csv(file_name)
        vector_store = create_and_store_embeddings(texts)
        response = askquestion(question, vector_store)

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
