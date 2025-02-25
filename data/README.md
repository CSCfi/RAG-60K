# Data preparation for LLM

This repo contains scripts which download the scientific publications and extract the text from open access journal copernicus https://publications.copernicus.org/open-access_journals/open_access_journals_a_z.html.

## Get all the links of the articles

```
sbatch download_links.sh
```

## Download the PDFs of the articles given the links obtained from above

```
sbatch pdf_download.sh
```

All the articles in the PDF format are stored in the **copernicus** folder. It is quite large, the data will not be uploaded here. We manage to download 60,712 pdfs.

## Extract the text content from the articles

```
sbatch extract_text.sh
```

The output of this command is a json file which is obtained by extracting the text from those pdfs where headers, footers, and references are removed.

The json file is 3.2G in size, the data inside looks like the samples listed below, it is a list having 60,712 entries with each entry consists of filename and the actual text of the filename,
```
[
     {'filename': paper1, 'text': 'this is the first paper content'},
     {'filename': paper2, 'text': 'this is the second paper content'},
     {'filename': paper3, 'text': 'this is the third paper content'},
     ...

]
```
The total number of tokens in this large json file is 723,178,588, less than 1B tokens.
