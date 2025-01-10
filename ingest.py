import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pandas as pd
import pickle
import os
os.environ["PYTORCH_SDP_BACKEND"] = "disable"  # Disables the SDPA backend to fallback on alternatives
import faiss

# load the data
json_file_path = "../../data/extracted_texts.json"

# the data currently used is from the scientific publications from https://publications.copernicus.org/open-access_journals/open_access_journals_a_z.html, We manage to download 60712 pdfs. This json file is obtained by extracting the text from those pdfs where headers, footers, and references are removed.

# The json is 3.2G, the data inside looks like the samples listed below, it is a list having 60712 entries with each entry consists of filename and the actual text of the filename,
#  [
#     {'filename': paper1, 'text': 'this is the first paper content'},
#     {'filename': paper2, 'text': 'this is the second paper content'},
#     {'filename': paper3, 'text': 'this is the third paper content'},
#     ...

# ]
# the total number of tokens in this large json file is 723,178,588, less than 1B tokens.

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2" # open source and light weight, can be explored if this embedding model does not fetch the relevant content
faiss_index_path = "./index/" # path to store the vectorstore
metadata_path = "./metadata/" # path to store the metadata

os.makedirs(faiss_index_path, exist_ok=True)
os.makedirs(metadata_path, exist_ok=True)

# the text in each paper is quite long, here we split it, chunk_size and overlap size can be explored too.
chunk_size = 256
overlap = 50
bs = 512*12 # adjust based on the clusters gpu memory limit
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define a Dataset class
class JSONTextDataset(Dataset):
    def __init__(self, documents, chunk_size=1000, overlap=20):
        """
        Args:
            json_file_path (str): Path to the JSON file.
            chunk_size (int): Maximum size of each chunk of text.
            overlap (int): Number of overlapping characters between chunks.
        """
        # Load the JSON data
        with open(json_file_path, 'r') as file:
            self.data = json.load(file)

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        # Process the data into chunks
        self.processed_data = self._process_data()

    def _process_data(self):
        """
        Process the raw data and split the text into chunks while keeping the filename.
        """
        processed_data = []
        for entry in self.data:
            filename = entry.get('filename')
            text = entry.get('text', '')

            # Split the text using RecursiveCharacterTextSplitter
            text_chunks = self.text_splitter.split_text(text)

            # Append filename and each chunk to the processed data
            for chunk in text_chunks:
                processed_data.append((filename, chunk))

        return processed_data

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        return self.processed_data[idx]

# Main model inference function with Data Distributed Parrallel (DDP), details about DDP can be found from pytorch documentation
def main():
    dist.init_process_group(backend='nccl')


    local_rank = int(os.environ['LOCAL_RANK'])
    print(local_rank)
    torch.cuda.set_device(local_rank)

    verbose = dist.get_rank() == 0  # print only on global_rank==0

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name).to(device)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    # Dataset and DataLoader with Distributed Sampler
    dataset = JSONTextDataset(json_file_path, chunk_size, overlap)
    sampler = DistributedSampler(dataset) # this DistributedSampler function ensures each model get a unique batch of data samples
    dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4,pin_memory=True,shuffle=False)

    # Initialize the faiss vectorstore, maybe needs to be changed if using faiss-gpu.
    index = faiss.IndexFlatL2(model.module.config.hidden_size)


    all_sources = []
    all_content = []

    for filename, texts in tqdm(dataloader):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        # model inference
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu()
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy()
        # collect the embeddings to the vectorstore, append the metadata info too
        index.add(normalized_embeddings)
        all_sources.extend(filename)
        all_content.extend(texts)
    # Save metadata for each rank
    metadata_df = pd.DataFrame({"source": all_sources, "Content": all_content})
    metadata_df.to_pickle(f"{metadata_path}rank_{local_rank}.pkl")
    faiss.write_index(index, f"{faiss_index_path}rank_{local_rank}.index")

    # Finalize the distributed process
    dist.destroy_process_group()

if __name__ == "__main__":


    main()
