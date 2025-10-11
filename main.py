import argparse
import json
import logging
import os
import time
from urllib.parse import urlparse

import config
from src.crawler import PoliteCrawler
from src.indexer import Indexer
from src.qa_engine import QAEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_domain_dir_name(url: str) -> str:
    """Creates a filesystem-safe directory name from a URL's domain."""
    try:
        netloc = urlparse(url).netloc
        return netloc
    except:
        return "default_site"


def setup_directories(base_path: str):
    """Ensures that the necessary base directory for a context exists."""
    os.makedirs(base_path, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "KonduitCrawler",
        description="A RAG based web-crawler to answer queries related to provided websites",
    )
    subparser = parser.add_subparsers(
        dest="command", required=True, help="Available Commands"
    )

    # === Crawler Arguments ===
    crawler_parse = subparser.add_parser(
        name="crawler_cli", help="Crawl a website to gather content"
    )
    crawler_parse.add_argument(
        "start_url", type=str, help="URL for the crawler to access"
    )
    crawler_parse.add_argument(
        "--max_pages",
        type=int,
        help="The max number of pages to crawl through",
        default=config.CrawlerConfig.MAX_PAGES,
    )
    crawler_parse.add_argument(
        "--politeness",
        type=int,
        help="The number of milliseconds to wait for before the next request",
        default=config.CrawlerConfig.CRAWL_DELAY,
    )

    # === Indexer Arguments ===
    index_parse = subparser.add_parser(
        name="indexer_cli", help="Index content crawled from the given url"
    )
    index_parse.add_argument(
        "start_url", type=str, help="The base URL of the content to index."
    )
    index_parse.add_argument(
        "--chunk_size",
        type=int,
        default=config.IndexerConfig.CHUNK_SIZE,
        help="The size length of text chunks",
    )
    index_parse.add_argument(
        "--chunk_overlap",
        type=float,
        default=config.IndexerConfig.CHUNK_OVERLAP,
        help="The percentage of chunk overlap",
    )

    # === Q&A Arguments ===
    ask_parser = subparser.add_parser(
        "ask_cli", help="Ask questions related to indexed content from provided url."
    )
    ask_parser.add_argument(
        "start_url", type=str, help="The URL context for your question."
    )
    ask_parser.add_argument(
        "question",
        type=str,
        help="The question to ask, will return refusal if beyond scope of crawled content",
    )
    ask_parser.add_argument(
        "--top_k",
        type=int,
        default=config.QaConfig.TOP_CHUNKS,
        help="The number of chunks to retrieve based on similarity to question asked",
    )

    args = parser.parse_args()

    domain_dir_name = get_domain_dir_name(args.start_url)
    context_base_path = os.path.join("./data", domain_dir_name)
    crawled_output_path = os.path.join(context_base_path, "crawled_content.json")
    vector_store_path = os.path.join(context_base_path, "vector_store")

    setup_directories(context_base_path)

    start_time = time.time()

    if args.command == "crawler_cli":
        try:
            logging.info(f"Initializing Crawler for {args.start_url}")
            crawl = PoliteCrawler(
                start_url=args.start_url,
                max_pages=args.max_pages,
                delay_ms=args.politeness,
            )
            crawled_data = crawl.run()
            with open(crawled_output_path, "w", encoding="utf-8") as f:
                json.dump(crawled_data, f, indent=2, ensure_ascii=False)
            logging.info(
                f"Successfully Crawled Data, Output saved to {crawled_output_path}"
            )
        except (ValueError, ConnectionError) as e:
            logging.error(f"Crawler Initialization Failed, {e}")
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")

    elif args.command == "indexer_cli":
        try:
            logging.info(f"Starting Indexing Using ChromaDB for {args.start_url}")
            indexer = Indexer(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                input_file=crawled_output_path,
                vector_store_path=vector_store_path,
            )
            indexer.create_index()
            logging.info("Indexer processing complete.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during indexing: {e}")

    elif args.command == "ask_cli":
        try:
            logging.info(
                f"\nProcessing Query\nContext URL: {args.start_url}\nQuery: {args.question}"
            )
            engine = QAEngine(vector_store_path=vector_store_path, top_k=args.top_k)
            response_json = engine.answer_question(args.question)
            print(json.dumps(response_json, indent=2))
        except FileNotFoundError as e:
            logging.error(
                f"Initialization failed: {e}. Please ensure you have run the indexer for this URL first."
            )
        except Exception as e:
            logging.error(f"An unexpected error occurred during the Q&A process: {e}")

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Command '{args.command}' finished in {duration:.2f} seconds.")
