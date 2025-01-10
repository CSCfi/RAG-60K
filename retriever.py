import faiss
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
# Load the FAISS index
#index = faiss.read_index("faiss_index.index")
index = faiss.read_index("./index/combined_faiss_index.index")
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
# Load the metadata DataFrame from the pickle file
#all_metadata_df = pd.read_pickle("metadata.pkl")
all_metadata_df = pd.read_pickle("./metadata/combined_metadata.pkl")
# Define paths and parameters

model_name  = "sentence-transformers/all-MiniLM-L6-v2"  # Replace with preferred model

# Load the tokenizer and model once to avoid re-loading
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map='auto')
model.eval()
# Function to generate embeddings for a batch
def get_embeddings(text_batch):
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        #model_output = model(**inputs)
        model_output = model(**inputs).last_hidden_state[:, 0, :].cpu()
        model_output = torch.nn.functional.normalize(model_output, p=2, dim=1).numpy()
        #norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return model_output
# Function to perform search with an optional metadata filter
def search_with_metadata(query_text, k=3, metadata_filter=None):
    query_embedding = get_embeddings([query_text]).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    results = []
    for i, idx in enumerate(indices[0]):
        result_metadata = all_metadata_df.iloc[idx]  # Use DataFrame to retrieve metadata

        # Apply metadata filter if provided
        if metadata_filter:
            if all(result_metadata[key] == value for key, value in metadata_filter.items()):
                results.append((distances[0][i], result_metadata.to_dict()))
        else:
            results.append((distances[0][i], result_metadata.to_dict()))

    return results

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
import os

# # Set up your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "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"  # Set your actual API key here

# # Initialize OpenAI model with LangChain
# llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Use "gpt-3.5-turbo" if you prefer GPT-3.5


# Define a prompt template for question answering with retrieved context
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Use the context provided to answer the question concisely.

Context: {context}

Question: {question}

Answer:"""
)

from langchain.llms import HuggingFaceHub

# Set up Hugging Face API Key for hosted Mistral model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the Mistral model (replace with "mistral/mistral-7B" or "mistral/mixtral" for hosted versions)
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature": 0.7})
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


# Function to perform question-answering using retrieved context and GPT model
def answer_query_with_openai(query_text, k=3, metadata_filter=None):
    # Step 1: Retrieve relevant documents using FAISS search
    results = search_with_metadata(query_text, k=k, metadata_filter=metadata_filter)

    # Step 2: Concatenate the context from retrieved documents
    context = " ".join([meta['Content'] for _, meta in results])  # Adjust 'text' to the actual content field

    # Step 3: Generate the answer using LangChain LLM Chain
    answer = qa_chain.run(context=context, question=query_text)

    return answer, results  # Return the answer and results for transparency

# Example usage
query_text = "What are the Limitations of the PMF analysis as applied to VOC measurements?"
#metadata_filter = None
metadata_filter = {"source": "acp-11-2399-2011.pdf"}  # Adjust to your filter requirements
answer, retrieved_results = answer_query_with_openai(query_text, k=3, metadata_filter=metadata_filter)

# Display results
print("Answer:", answer)
print("Retrieved Documents:")
for distance, meta in retrieved_results:
    print('*' * 40)
    print(f"Distance: {distance}\nMetadata: {meta}")
