import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI model and embeddings
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()
app = Flask(__name__)
# Load your daily notes
# Assuming your notes are in text files in a 'daily_notes' directory
daily_notes = []
NOTES_DIR = "daily_notes"
for filename in os.listdir(NOTES_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(NOTES_DIR, filename), "r") as file:
            daily_notes.append(file.read())

# Create vector store from your notes
vectorstore = DocArrayInMemorySearch.from_texts(daily_notes, embedding=embeddings)

# Set up retriever
retriever = vectorstore.as_retriever()

# Create prompt template
TEMPLATE = """
Answer the question based on the context below. If you can't 
answer the question based on the context, reply "I don't know".
Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(TEMPLATE)

# Set up the chain
setup = RunnableParallel(context=retriever, question=RunnablePassthrough())
chain = setup | prompt | model | StrOutputParser()


# Function to query your notes
def query_notes(question):
    return chain.invoke(question)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    answer = query_notes(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)