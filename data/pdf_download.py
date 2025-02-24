import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm


def download_pdf(url, folder_path):
    """
    Download a PDF from the given URL and saves it to the specified folder.

    Parameters
    ----------
    url: str
         The URL of the PDF file.
    folder_path: str
         The folder where the PDF will be saved.

    """
    try:
        topic_name = url.split("//")[1].split(".")[0]
        article_name = url.split("/")[-4:-1]  # Extract the article name from the URL
        article_name = "-".join(article_name) + ".pdf"
        article_name = topic_name + "-" + article_name
        response = requests.get(url + article_name)
        if response.status_code == 200:
            with open(os.path.join(folder_path + article_name), "wb") as file:
                file.write(response.content)
                # print(f"Downloaded: {article_name}")
        else:
            print(f"Failed to download {article_name}. HTTP Status Code: {response.status_code}")

    except Exception as e:
        print(f"Error downloading {url}: {e}")


def ensure_folder_exists(folder_path):
    """Ensure the download folder exists or create it if it doesn't."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def fast_download_pdfs(urls, folder_path, max_workers=5):
    """
    Download multiple PDFs in parallel using threads.

    Parameters
    ----------
    urls: list
          List of PDF URLs to download.
    folder_path: str
          The folder where PDFs should be saved.
    max_workers: int
          Number of worker threads to use for downloading.

    """
    ensure_folder_exists(folder_path)

    # Use ThreadPoolExecutor for parallel downloading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the download tasks
        futures = [executor.submit(download_pdf, url, folder_path) for url in urls]

        # Process each future as they are completed
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
            # for future in as_completed(futures):
            try:
                future.result()  # Raises an exception if the thread encountered one
            except Exception as e:
                print(f"Error during download: {e}")


all_download_links = []
for i in range(7):
    df = pd.read_csv("./csv_files/file" + str(i) + ".csv", header=None)
    links = df[0].values.tolist()
    all_download_links += links
# Remove duplicate URLs
pdf_urls = list(set(all_download_links))
# breakpoint()
download_folder = "./copernicus/"

# Fast download with 5 parallel workers
fast_download_pdfs(pdf_urls, download_folder, max_workers=48)
