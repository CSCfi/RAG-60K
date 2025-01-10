# RAG-60K

This is a incomplete Git repo, trying to build a RAG pipeline with 60K pdfs.


## Build the vectorstore with DDP
`sbatch run_devg.sh`

## Merge the vectorstore
`python merge.py`

## Retrieve the vectorstore
`sbatch run_smallg.sh`

Notes: The retriever.py is a messy, needs to be written in a better way. Also, should check how to use FAISS-GPU to retrieve faster. LUMI now supports FAISS-GPU.

Final notes: There should be better or smarter way of dealing with large amount of data using RAG.
