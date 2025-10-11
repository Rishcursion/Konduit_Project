import os

from dotenv import load_dotenv


# DEFAULT CRAWLER SETTINGS TO USE
class CrawlerConfig:
    CRAWL_DELAY = 1000  # in milliseconds
    MAX_PAGES = 50  # As per project guidelines
    USER_AGENT = "KonduitCrawler/1.0"


# DEFAULT INDEXER SETTINGS
class IndexerConfig:
    CHUNK_SIZE = 800  # choosing initial lower chunk size as per project guidelines
    CHUNK_OVERLAP = 0.125  # 12.5% overlap initially
    VECTOR_STORE_LOC = "data/vector"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Q&A Settings
class QaConfig:
    TOP_CHUNKS = 6


class Env:
    load_dotenv(".env")
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
