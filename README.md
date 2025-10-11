# Konduit Assignment
This project is an end-to-end Retrieval-Augmented Generation (RAG) system built to fulfill the requirements of the Konduit engineering assignment. The system can crawl a target website, index its content into a searchable knowledge base, and answer questions based only on the information it has gathered.

https://docs.google.com/document/d/1D6kzwef0R2kKsN_bEOkc8KUoZ9hrf80EBjbnJlwxW0I/edit?tab=t.0


# Project Scope
- The current project incorporates three elements outlined in the project description:
    - The Web Crawler to extract website content, while respecting robots.txt and staggering api requests so as to be polite.
    - The Indexer, to handle embedding and initializing the vector store for further query resolution
    - The QA engine, that utilizes the aforementioned vectore store and the Gemini API to resolve user API queries

# Project Directory Setup
/
├── main.py             # Main entrypoint and CLI orchestrator using argparse.
├── config.py           # Centralized configuration for all modules.
├── requirements.txt    # Project dependencies for easy environment setup.
├── .env                # For storing secret API keys (e.g., GOOGLE_API_KEY).
├── data/               # Root directory for all persistent data.
│   └── <domain_name>/  # A unique directory is created for each crawled site.
│       ├── crawled_content.json
│       └── vector_store/
└── src/
    ├── crawler.py      # Contains the PoliteCrawler class.
    ├── indexer.py      # Contains the Indexer class for embedding and storage.
    └── qa_engine.py    # Contains the QAEngine class for RAG logic.

# Getting Started
1. Setup
Clone the repository via http and navigate to project root directory. It is recommended to create a python virtual environment for reproducibility.
`
python -m venv venv
source venv/bin/activate #if  using linux
venv\bin\activate #if using Windows 
`
Install the required extra libraries for compatibility
`
pip install -r requirements.txt
`

2. Configuration
create a .env file in your root project directory, and add your Google AI studio API key:
`
GOOGLE_API_KEY = <google_api_key>
`
modify other settings in config.py as per preference.

# Usage
The application is a CLI with three main commands to be used sequentially for a specific URL.

## Step 1: Crawl A Website
Crawls a specified website URL and stores the retrieved data under the specified local storage option
`
python main.py crawler_cli https://<domain_name>.<domain_extension>?/<domain sub-routes> --max_pages 30 --politeness=1000
`
## Step 2: Indexing 
The next step is to process the crawled data into a vector store,for top k-chunk retrival for enhancing response accuracy/contextual understanding
`
python main.py indexer_cli https://<domain_name>.<domain_extension>?/<domain sub-routes> --chunk_size 800 --chunk_overlap 0.125
`
Note: The same route used in Step 1 should be mentioned in the indexing, to ensure the same folder is picked up.

## Step 3: Ask your query
The final step is to resolve your query based on the LLM's understanding and the vectore database's similarity retrival.
`
python main.py ask_cli https://<domain_name>.<domain_extension>?/<domain sub-routes> <query> --top_k 6
`

# Design Choices And TradeOffs:
- Dynamic Contexts: One design choice was to create a seperate sub directory for each URL queried to improve knowledge retention and reduce API calls, which focuses on aligning with the politness factor of web crawling, albeit at the cost of implementing context switching for each context call.
- Chunking Strategy: A chunk size of 800 characters with a 10% overlap was chosen as a default. This size is large enough to capture the semantic context of most paragraphs while being small enough for efficient processing by the embedding model. The overlap helps maintain continuity for concepts that span chunk boundaries.
- Embedding Model: A HuggingFace SentenceTransformerEmbedding model that runs locally was chosen over a Google Gemini-based Embedding Transformer due to cost-limitations, with the trade-off of computational time and complexity.
- Vector Storage: ChromaDB was selected due to its relative simplicity compared to Pinecone and other Vector databases, as well as for its local-based file-persistance and LangChain integration, making it easy to run without external database infrastructure.
- Grounded Generation: The Q&A prompt explicitly instructs the LLM to answer only from the provided context and to refuse if the answer is not present. This prioritizes accuracy and safety over trying to be helpful with outside knowledge.

