import faiss
import numpy as np
import pandas as pd

from glob import glob

faiss_index_files = sorted(glob("./index/rank_*.index"))  # Match all index files that start with 'faiss_index_' and end with '.index'
metadata_files = sorted(glob("./metadata/rank_*.pkl"))  # Match all index files that start with 'faiss_index_' and end with '.index'
#breakpoint()
# Initialize the first index
index = faiss.read_index(faiss_index_files[0])

# Load the embeddings from the first index
embedding_dim = index.d  # Get the dimension of the embeddings

# Now add embeddings from the other indices
for index_file in faiss_index_files[1:]:
    # Load the index
    temp_index = faiss.read_index(index_file)

    # Ensure that the dimensions match
    if temp_index.d != embedding_dim:
        raise ValueError("Embedding dimensions do not match.")

    # Add embeddings from temp_index into the main index
    index.add(temp_index.reconstruct_n(0, temp_index.ntotal))  # Reconstruct and add vectors to the index


# Concatenate metadata from each file
metadata_list = [pd.read_pickle(metafile) for metafile in metadata_files]
metadata = pd.concat(metadata_list, ignore_index=True)
# Step 4: Save the combined FAISS index
faiss.write_index(index, "./index/combined_faiss_index.index")

# Step 5: Save the combined metadata
metadata.to_pickle("./metadata/combined_metadata.pkl")

print("FAISS index and metadata merged and saved successfully!")
#breakpoint()
