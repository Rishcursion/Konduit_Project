import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

import config
import numpy as np
from src.crawler import PoliteCrawler
from src.indexer import Indexer
from src.qa_engine import QAEngine

# --- Enhanced Logging Setup ---
# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a file handler to save logs to a file
file_handler = logging.FileHandler("cli.log", mode="a")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create a stream handler to display logs in the console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


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


def save_results(filepath: str, command: str, data: dict):
    """Saves the results of a command to a JSON file, appending new results."""
    logging.info(f"Saving results for '{command}' command to {filepath}")

    timestamp = datetime.now().isoformat()
    new_entry = {command: data}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    if "runs" not in existing_data:
        existing_data["runs"] = []
    existing_data["runs"].append({timestamp: new_entry})

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)


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
        "--max-pages",
        dest="max_pages",
        type=int,
        help="The max number of pages to crawl through",
        default=config.CrawlerConfig.MAX_PAGES,
    )
    crawler_parse.add_argument(
        "--politeness",
        dest="politeness",
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
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=config.IndexerConfig.CHUNK_SIZE,
        help="The size length of text chunks",
    )
    index_parse.add_argument(
        "--chunk-overlap",
        dest="chunk_overlap",
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
        "--top-k",
        dest="top_k",
        type=int,
        default=config.QaConfig.TOP_CHUNKS,
        help="The number of chunks to retrieve based on similarity to question asked",
    )
    ask_parser.add_argument(
        "--compact",
        action="store_true",
        help="Summarize the source snippets for a more compact output.",
    )

    # === Evaluation Command ===
    eval_parser = subparser.add_parser(
        "eval_cli",
        help="Evaluate a list of queries stored in a json file over a single URL for evaluation",
    )
    eval_parser.add_argument(
        "start_url",
        type=str,
        help="The URL context to evaluate against.",
    )
    eval_parser.add_argument(
        "--eval-file",
        dest="eval_file",
        type=str,
        required=True,
        help="The JSON file containing a list of queries (e.g., eval_questions.json)",
    )
    eval_parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=config.QaConfig.TOP_CHUNKS,
        help="Number of chunks to retrieve for context during evaluation.",
    )

    args = parser.parse_args()

    domain_dir_name = get_domain_dir_name(args.start_url)
    context_base_path = os.path.join("./data", domain_dir_name)
    crawled_output_path = os.path.join(context_base_path, "crawled_content.json")
    vector_store_path = os.path.join(context_base_path, "vector_store")
    results_path = os.path.join(context_base_path, "results.json")

    setup_directories(context_base_path)

    start_time = time.time()
    result_data = {}

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
            result_data = {
                "status": "Success",
                "pages_crawled": len(crawled_data),
                "output_file": crawled_output_path,
            }
        except Exception as e:
            logging.error(f"An unexpected error occurred during crawl: {e}")
            result_data = {"status": "Failed", "error": str(e)}

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
            result_data = {
                "status": "Success",
                "input_file": crawled_output_path,
                "vector_store": vector_store_path,
            }
        except Exception as e:
            logging.error(f"An unexpected error occurred during indexing: {e}")
            result_data = {"status": "Failed", "error": str(e)}

    elif args.command == "ask_cli":
        try:
            logging.info(
                f"\nProcessing Query\nContext URL: {args.start_url}\nQuery: {args.question}"
            )
            engine = QAEngine(vector_store_path=vector_store_path, top_k=args.top_k)
            response_json = engine.answer_question(args.question)
            
            result_data = response_json

            if args.compact:
                logging.info("Compact output enabled. Summarizing source snippets.")
                for source in response_json.get("sources", []):
                    snippet = source.get("snippet", "")
                    source["snippet"] = snippet[:150] + "..." if len(snippet) > 150 else snippet
            
            print(json.dumps(response_json, indent=2))
            logging.info("Query Resolved")
        except Exception as e:
            logging.error(f"An unexpected error occurred during the Q&A process: {e}")
            result_data = {"status": "Failed", "error": str(e)}

    elif args.command == "eval_cli":
        try:
            with open(args.eval_file, "r") as f:
                questions = json.load(f)

            logging.info(f"Evaluating RAG crawler with {len(questions)} queries")
            engine = QAEngine(vector_store_path=vector_store_path, top_k=args.top_k)
            
            latencies = []
            total_input_tokens = 0
            total_output_tokens = 0
            total_tokens_used = 0
            
            for question in questions:
                response = engine.answer_question(question)
                latencies.append(response["timings"]["total_ms"])
                if response.get("usage"):
                    total_input_tokens += response["usage"].get("prompt_token_count", 0)
                    total_output_tokens += response["usage"].get("candidates_token_count", 0)
                    total_tokens_used += response["usage"].get("total_token_count", 0)

                logging.info(
                    f"Question: {question}\n"
                    f"Answer: {response['answer'][:50]}...\n"
                    f"Time Taken: {response['timings']['total_ms']} ms"
                )
            
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            
            logging.info("--- Evaluation Complete ---")
            logging.info(f"Total Queries: {len(latencies)}")
            logging.info(f"Average Latency: {np.mean(latencies):.2f} ms")
            logging.info(f"p50 (Median) Latency: {p50_latency:.2f} ms")
            logging.info(f"p95 Latency: {p95_latency:.2f} ms")
            logging.info("--- Token Usage ---")
            logging.info(f"Total Input Tokens: {total_input_tokens}")
            logging.info(f"Total Output Tokens: {total_output_tokens}")
            logging.info(f"Total Tokens Used in Batch: {total_tokens_used}")

            result_data = {
                "status": "Success",
                "total_queries": len(latencies),
                "average_latency_ms": np.mean(latencies),
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "total_tokens_used": total_tokens_used,
            }

        except FileNotFoundError:
            logging.error(f"Evaluation file not found at {args.eval_file}")
            result_data = {"status": "Failed", "error": f"Evaluation file not found at {args.eval_file}"}
        except Exception as e:
            logging.error(f"An unexpected error occurred during evaluation: {e}")
            result_data = {"status": "Failed", "error": str(e)}

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Command '{args.command}' finished in {duration:.2f} seconds.")
    
    result_data["total_duration_seconds"] = str(round(duration, 2))
    save_results(filepath=results_path, command=args.command, data=result_data)
