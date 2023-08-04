"""Processing pipeline for Coher GenQ.

Scrapping contexts from webpages and vectorize them using Cohere \
    Embedding model.
Add meta data to vectors and upload them to Pinecone vector database.
"""


import json
import requests
import logging
import cohere
import pinecone
from tqdm.auto import trange
from bs4 import BeautifulSoup


def get_text(url):
    """Send request to webpage and scrape text."""
    try:
        # Send a GET request to the URL
        try:
            response = requests.get(url)
            response.raise_for_status()  # Exception for 4xx and 5xx codes
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception:
            content = open(url, "r", encoding="utf-8").read()
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")

        # Find all the text inside paragraph tags
        paragraphs = soup.find_all("p")

        # Extract and concatenate the text from the paragraphs
        article_text = "\n\n".join(p.get_text() for p in paragraphs)

    except requests.exceptions.RequestException as e:
        logging.error("Error fetching the Wikipedia article:")
        logging.erro(e)
        article_text = None

    except Exception as e:
        logging.error("An error occurred:")
        logging.error(e)
        article_text = None

    return article_text


def get_contexts(links):
    """Scrape text from webpages and return combined contexts."""
    articles = []
    for link in links:
        article = get_text(link)
        if article:
            articles.append(article)
    return "\n\n".join(articles).split(sep="\n\n")


def get_embeds(contexts, api, embedding_model="embed-english-v2.0"):
    """Embed contexts using Cohere."""
    response = api.embed(texts=contexts, model=embedding_model)
    return response.embeddings


def upsert_embeds(pinecone_access, contexts, embeds):
    """Upload embeddings to pinecone vector database."""
    index_name = "cohere-gqa"
    batch_size = 32
    # connect to pinecone
    pinecone.init(
        api_key=pinecone_access["api_key"], environment=pinecone_access["env"]
    )
    # create index if not there
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))
    # connect to index
    index = pinecone.Index(index_name)
    num_current_vectors = index.describe_index_stats()["total_vector_count"]
    # create ids for querys
    ids = [
        str(i) for i in range(num_current_vectors,
                              len(contexts) + num_current_vectors)
    ]
    contexts = [{"text": c} for c in contexts]
    assert len(ids) == len(embeds) == len(contexts), "Data Missmacth ..."
    # upload data to index
    data = list(zip(ids, embeds, contexts))
    for i in trange(0, len(data), batch_size):
        index.upsert(data[i: i + batch_size])


def main(tokens_path):
    """Run the processing pipeline."""
    logging.info("Loading Access Tokens ...")
    with open(tokens_path) as f:
        access_tokens = json.load(f)
    logging.info("Connecting to Cohere API ...")
    api = cohere.Client(access_tokens["cohere"]["api_key"])
    links = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    ]
    logging.info("Scrapping Data ...")
    contexts = get_contexts(links)
    logging.info("Embedding Contexts ...")
    embeddings = get_embeds(contexts, api)
    logging.info("Populating Pinecone Index ...")
    upsert_embeds(access_tokens["pinecone"], contexts, embeddings)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    tokens_path = "./access_tokens.json"
    main(tokens_path)
