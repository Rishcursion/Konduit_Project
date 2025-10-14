import json
import logging
import os
import time

import config
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma

logging.basicConfig(
    filename="rag_indexer.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Indexer:
    """
    Initializes the indexer, which embeds content crawled from a URL
    for vector-based similarity retrieval.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: float,
        input_file: str,
        vector_store_path: str,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap_ratio = chunk_overlap if 0 < chunk_overlap < 1 else 0.125
        self.input_file = input_file
        self.vector_store_path = vector_store_path

        logging.info(f"Loading Embedding Model: {config.IndexerConfig.EMBEDDING_MODEL}")
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=config.IndexerConfig.EMBEDDING_MODEL
        )

        logging.info("Initializing Text Splitter")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_overlap_ratio * self.chunk_size),
            length_function=len,
        )

    def _load_data(self) -> list[Document]:
        """
        Loads Crawled JSON Data for processing into LangChain Documents.
        """
        data = {}
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            logging.error(
                f"Input File Not Found At Provided Location: {self.input_file}, Run crawler before indexing"
            )
            return []

        documents = []
        for url, text in data.items():
            doc = Document(page_content=text, metadata={"source_url": url})
            documents.append(doc)

        logging.info(f"Loaded {len(documents)} documents from {self.input_file}")
        return documents

    def create_index(self) -> None:
        """
        Orchestrates loading, chunking, embedding, and storage of provided data,
        with performance logging and progress indication.
        """
        start_time = time.time()

        documents = self._load_data()
        if not documents:
            logging.warning("Documents list is empty. Aborting indexing.")
            return

        logging.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

        logging.info(
            f"Creating vector store at: {self.vector_store_path}. This may take a while..."
        )

        # Use a generator with tqdm to show progress for the embedding process

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=self.vector_store_path,
        )

        end_time = time.time()
        duration = end_time - start_time

        logging.info(
            f"Successfully created and persisted vector store at {self.vector_store_path}"
        )
        logging.info(f"Total vectors in store: {vector_store._collection.count()}")
        logging.info(f"Indexing process completed in {duration:.2f} seconds.")
