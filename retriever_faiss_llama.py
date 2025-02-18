import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import json
import time
import os
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
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
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)

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

def ask_llama(prompt):
    # Format prompt correctly as a string
    system_prompt = "You are an AI assistant helping users with document-based questions."
    formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    response = pipe(formatted_prompt, max_new_tokens=512)
    # Extract the raw generated text
    full_response = response[0]["generated_text"]

    # Remove everything before "Assistant:"
    if "Assistant:" in full_response:
        assistant_response = full_response.split("Assistant:")[-1].strip()
    else:
        assistant_response = full_response.strip()
    # Extract generated text
    return assistant_response
# Chatbot loop
while True:
    query_text = input("You: ")
    if query_text.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Measure retrieval time
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
    gpt_response = ask_llama(final_prompt)

    total_time = time.time() - total_start_time
    print(f"⏱ Total retrieval + response time: {total_time:.4f} seconds")
    retrieved_filenames = "\n".join([all_metadata[idx]["filename"] for idx in indices[0]])
    print(retrieved_filenames)
    print("\nBot:", gpt_response)
    print("-" * 50)
