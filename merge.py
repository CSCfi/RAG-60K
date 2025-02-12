import os
import json
import faiss
import numpy as np
import time
# Configuration
faiss_index_path = "./faiss_index/"
faiss_index_file = f"{faiss_index_path}faiss_index.bin"
metadata_file = f"{faiss_index_path}metadata.json"

# Step 1: Load all .npy embedding files
embedding_files = sorted([f for f in os.listdir(faiss_index_path) if f.startswith("data_") and f.endswith(".npy")])
metadata_files = sorted([f for f in os.listdir(faiss_index_path) if f.startswith("metadata_rank_") and f.endswith(".json")])

# Step 2: Concatenate all embeddings
all_embeddings = []
for file in embedding_files:
    embeddings = np.load(os.path.join(faiss_index_path, file))
    all_embeddings.append(embeddings)

all_embeddings = np.vstack(all_embeddings)  # Shape: (total_chunks, embedding_dim)
print(f"Total embeddings shape: {all_embeddings.shape}")

# Step 3: Load and merge all metadata
all_metadata = []
for file in metadata_files:
    with open(os.path.join(faiss_index_path, file), "r") as f:
        all_metadata.extend(json.load(f))

print(f"Total metadata entries: {len(all_metadata)}")

# Step 4: Create and save FAISS index

ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)
embedding_dim = all_embeddings.shape[1]
cpu_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance-based FAISS index
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
start_time = time.time()  # Start timer
gpu_index.add(all_embeddings)              # add vectors to the index
end_time = time.time()  # End timer
# Print Time Taken
time_taken = end_time - start_time
print(f"Time taken to add {all_embeddings.shape[0]} vectors: {time_taken:.4f} seconds")

print(gpu_index.ntotal)

# Move the index to CPU before saving
cpu_index_saved = faiss.index_gpu_to_cpu(gpu_index)

# Save the CPU index to disk
faiss.write_index(cpu_index_saved, faiss_index_file)
print(f"FAISS index saved to {faiss_index_file}")

# Step 5: Save merged metadata
with open(metadata_file, "w") as f:
    json.dump(all_metadata, f)

print(f"Metadata saved to {metadata_file}")
