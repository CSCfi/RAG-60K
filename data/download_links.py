import csv
import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_sublinks_with_conditions(base_url, condition_func=None):
    """
    Fetch and returns all sublinks from the base URL that meet the given condition.

    Parameters
    ----------
    base_url: str
         The main URL to scrape for sublinks.
    condition_func:
         A function that takes a URL as input and returns True if the URL should be included.

    Returns
    -------
    List of sublinks that meet the condition.

    """
    # Send request to the base URL
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all 'a' tags with 'href' attribute
    all_sublinks = [urljoin(base_url, link["href"]) for link in soup.find_all("a", href=True)]
    all_sublinks = [link for link in all_sublinks if ".png" not in link and ".jpg" not in link]
    # breakpoint()
    # If no condition function is provided, return all links
    if condition_func is None:
        return all_sublinks

    # Apply the condition to filter the links
    filtered_sublinks = [link for link in all_sublinks if condition_func(link)]

    return filtered_sublinks


def article_condition(link):
    # breakpoint()
    return (
        (link.endswith(".copernicus.org") or link.endswith(".copernicus.org/") or link.endswith("articles/"))
        and ("www" not in link.split("//")[1])
        and ("meetings" not in link)
    )


def article_issue_condition(link):
    return ("article" in link and "issue" in link) and ".pdf" not in link


def last_three_number_condition(link):
    last_three_parts = link.split("/")[-4:-1]
    return all(re.match(r"^\d+$", part) for part in last_three_parts)


# Example usage
ba_url = "https://publications.copernicus.org/open-access_journals/journals_by_subject.html"
article_pages = get_sublinks_with_conditions(ba_url, article_condition)
output_dir = "./csv_files/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_download_links = []
counter = 6  # 0-6
# Print the filtered sublinks
for link_1 in tqdm(article_pages[counter * 4 : counter * 4 + 4]):
    print(link_1)
    if "articles" in link_1:
        issue_link = get_sublinks_with_conditions(link_1, article_issue_condition)
    else:
        issue_link = get_sublinks_with_conditions(link_1 + "/articles/", article_issue_condition)

    # breakpoint()

    for link_2 in tqdm(issue_link):
        article_link = get_sublinks_with_conditions(link_2, last_three_number_condition)
        filtered_urls = [url for url in article_link if url != "javascript:void(0)" or "html" not in url]
        # breakpoint()
        all_download_links += filtered_urls

with open(output_dir + "file" + str(counter) + ".csv", "w") as f:
    wr = csv.writer(f, delimiter="\n")
    wr.writerow(all_download_links)
