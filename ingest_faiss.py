import json
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import faiss
import numpy as np
import re

# define the path of the data
json_file_path = "/scratch/project_462000824/data/extracted_texts.json"

# The data currently used is from the scientific publications from https://publications.copernicus.org/open-access_journals/open_access_journals_a_z.html, We manage to download 60712 pdfs. This json file is obtained by extracting the text from those pdfs where headers, footers, and references are removed.

# The json is 3.2G, the data inside looks like the samples listed below, it is a list of dictionaries, having 60712 entries with each entry consists of two keys "filename" and "text" and their corresponding values.
#  [
#     {'filename': paper1.pdf, 'text': 'this is the first paper content, this length of each paper varies.'},
#     {'filename': paper2.pdf, 'text': 'this is the second paper content'},
#     {'filename': paper3.pdf, 'text': 'this is the third paper content'},
#     ...

# ]
# the total number of tokens in this large json file is 723,178,588, less than but close to 1B tokens.

embedding_model_name = 'Alibaba-NLP/gte-base-en-v1.5' # "sentence-transformers/all-MiniLM-L6-v2"
#embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1024 # this number should not to exceed the token limit of the model. The larger chunk size is, the less number of the embedding vectors will end up with.
overlap = 50
bs = 128 # can be adjusted based on the GPU memory
faiss_index_path = "./faiss_index/"  # FAISS index directory
os.makedirs(faiss_index_path, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Dataset class
class JSONTextDataset(Dataset):
    def __init__(self, json_file_path=json_file_path, chunk_size=1000, overlap=20):
        with open(json_file_path, 'r') as file:
            self.data = json.load(file)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        self.processed_data = self._process_data()

    def _process_data(self):
        processed_data = []
        for entry in self.data:
            filename = entry.get('filename')
            text = entry.get('text', '')
            # Fix broken words (remove hyphen and join with next word)
            text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

            # Remove unnecessary newline characters that split words or sentences
            text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)

            # Fix any multiple spaces into a single space
            text = re.sub(r'\s+', ' ', text)

            # Remove any extra newlines that split paragraphs or sentences
            text = re.sub(r'\n+', '\n', text)

            text_chunks = self.text_splitter.split_text(text)
            for idx, chunk in enumerate(text_chunks):
                processed_data.append((filename, idx,chunk))
        return processed_data
    # the processed_data is a list of tuples
    # [
    # ("paper_1.pdf", 3, "this is the content of the paper_1.pdf and chunk id 3"),
    # ("paper_2.pdf", 4, "this is the content of the paper_2.pdf and chunk id 4"),
    # ...
    # ]


    def __len__(self):
        print('the total length of the data from the data class is:', len((self.processed_data)))
        return len(self.processed_data)

    def __getitem__(self, idx):

        return self.processed_data[idx] # This returns one sample of the dataset, it is a tuple with length of 3 ("paper_1.pdf", 3, "this is the content of the paper_1.pdf from chunk id 3")


# Main function
def main():
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    # define the model and its tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True).to(device)
    # use pytorch DDP, copy the model to each gpu id
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    dataset = JSONTextDataset(json_file_path, chunk_size, overlap)

    #this will distribute data across multiple GPUs. It ensures that each process (GPU) gets a unique subset of the dataset without overlap, preventing redundant computations
    sampler = DistributedSampler(dataset)

    def collate_fn(batch):
        filenames, indices, texts = zip(*batch)
        return list(filenames), torch.tensor(indices, dtype=torch.long), list(texts)

    dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=8,
                            pin_memory=True, shuffle=False, collate_fn=collate_fn)


    local_embeddings = []
    local_metadatas = []

    for batch in tqdm(dataloader, desc=f"Processing Rank {rank}") if rank == 0 else dataloader:
        filenames, chunk_indices, texts = batch
        # filesnames: ["paper_1.pdf",..., "paper_128.pdf"]
        # chunk_indices: [tensor(1),..., tensor(4)]
        # texts: ["this is the content of paper_1 chunk 1", ..., "this is the content of paper_128 chunk 4"]

        chunk_indices = [int(idx) for idx in chunk_indices]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:, 0]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        local_embeddings.extend(normalized_embeddings.cpu().numpy())
        metadatas = [{"filename": filename, "chunk_index": chunk_idx,"text": text} for filename, chunk_idx,text in zip(filenames, chunk_indices,texts)]
        local_metadatas.extend(metadatas)

    #  local_metadatas: list of dictionaries
    # [
    #     {"filename": paper_1.pdf, "chunk_index": 1, "text": this is the content of paper_1 chunk 1},
    #     ...
    #     ,
    #     {"filename": paper_n.pdf, "chunk_index": 8, "text": this is the content of paper_n chunk 8},
    # ]


    local_embeddings = np.array(local_embeddings)
    print(f"this is rank {rank}, the shape of the local_embeddings is {local_embeddings.shape}")

    dist.barrier()  # Ensures all processes complete before shutdown
    np.save(f'{faiss_index_path}data_{rank}.npy', local_embeddings) # save
    metadata_path = f"{faiss_index_path}metadata_rank_{rank}.json"
    with open(metadata_path, "w") as f:
        json.dump(local_metadatas, f)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
