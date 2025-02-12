import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API Key (replace with your actual key or set it as an environment variable)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Load the model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to("cuda").eval()

# Optional: Convert model to FP16 for faster inference
model.half()

# Load FAISS index and metadata
faiss_index_path = "./faiss_index/"
faiss_index_file = f"{faiss_index_path}faiss_index.bin"
metadata_file = f"{faiss_index_path}metadata.json"

index = faiss.read_index(faiss_index_file)

# Move FAISS index to GPU if available
if torch.cuda.is_available():
    index = faiss.index_cpu_to_all_gpus(index)

# Load metadata
with open(metadata_file, "r") as f:
    all_metadata = json.load(f)

# Function to get embeddings for a query
def get_query_embedding(query_text):
    inputs = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():  # Disable gradients for inference
        embeddings = model(**inputs).last_hidden_state[:, 0]  # Extract [CLS] token embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize for cosine similarity
    return embeddings.half().cpu().numpy()  # Convert to FP16, move to CPU for FAISS

# Function to search FAISS index
def search_faiss(query_embedding, k=3):
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Function to query OpenAI GPT model (Updated for openai>=1.0.0)
def ask_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Change to "gpt-3.5-turbo" for lower cost
        messages=[
            {"role": "system", "content": "You are an AI assistant helping users with document-based questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0. # Smaller the temperature is, the more factual the response will be. Otherwise, higher values increase randomness, making responses more diverse and creative.
    )
    return response.choices[0].message.content

# Chatbot loop
while True:
    query_text = input("You: ")
    if query_text.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Measure the time of embedding the query, retrieval and generating answers
    total_start_time = time.time()

    query_embedding = get_query_embedding(query_text)
    distances, indices = search_faiss(query_embedding, k=3)  # Retrieve top 3 chunks

    retrieved_contexts = "\n".join([all_metadata[idx]["text"] for idx in indices[0]])

    # Construct a final prompt for GPT
    final_prompt = f"""Use the following retrieved context to answer the question accurately:

    Context:
    {retrieved_contexts}

    Question: {query_text}
    """

    # Get response from GPT
    gpt_response = ask_gpt(final_prompt)

    total_time = time.time() - total_start_time
    print(f"⏱ Total retrieval + response time: {total_time:.4f} seconds")

    print("\nBot:", gpt_response)
    print("-" * 50)


    ### Sample questions
    # 1. How to simulate a future climate with changing temporal covariance while largely retaining non-Gaussian features of the observations?
    # 2. How is the tropical cyclone best track data produced?
