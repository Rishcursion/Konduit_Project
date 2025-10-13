import json
import logging
import re
import socket
import time
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class PoliteCrawler:
    """
    A class to crawl a website politely and efficiently, staying within a specified domain.
    """

    def __init__(self, start_url: str, max_pages: int, delay_ms: int):
        """
        Initializes the crawler with its configuration.
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.delay_seconds = delay_ms / 1000

        # --- State Variables ---
        self.visited_urls = set()
        self.url_queue = [self.start_url]
        self.crawled_data = {}

        # --- Pre-flight Checks during initialization ---
        if not self._is_valid_url(self.start_url):
            raise ValueError(
                f"Invalid starting URL: '{self.start_url}'. Must include a scheme (e.g., https://)."
            )

        self.start_domain = urlparse(self.start_url).netloc
        if not self._domain_exists():
            raise ConnectionError(
                f"Domain '{self.start_domain}' does not appear to exist or DNS lookup failed."
            )

        self._setup_robot_parser()

    def _is_valid_url(self, url: str) -> bool:
        """Checks if a URL is syntactically valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _domain_exists(self) -> bool:
        """Checks if the crawler's start domain has a DNS entry."""
        try:
            socket.gethostbyname(self.start_domain)
            return True
        except socket.gaierror:
            return False

    def _clean_text(self, text: str) -> str:
        """
        A more sophisticated text cleaning pipeline.
        - Normalizes Unicode and removes strange artifacts.
        - Preserves meaningful paragraph breaks while collapsing unnecessary newlines.
        - Cleans up whitespace for better readability.
        """
        # 1. Normalize Unicode and fix common artifacts
        # Replace the incorrectly encoded pilcrow sign (¶) and right single quote (’), etc.
        text = text.replace("\u00c2\u00b6", "")
        text = text.replace("\u00e2\u0080\u0099", "'")
        text = text.replace(" \n", "\n")  # Fix newlines preceded by spaces

        # 2. Preserve paragraph breaks, but collapse in-line newlines.
        # This is the key step to avoid losing paragraph structure.
        # We replace double newlines (paragraph breaks) with a unique placeholder.
        text = re.sub(r"\n{2,}", "__PARAGRAPH_BREAK__", text)

        # Now, we can safely replace all remaining single newlines with spaces.
        text = text.replace("\n", " ")

        # Restore the paragraph breaks.
        text = text.replace("__PARAGRAPH_BREAK__", "\n\n")

        # 3. Collapse excessive whitespace into a single space.
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def _setup_robot_parser(self):
        """Initializes the robot parser."""
        robots_url = urljoin(self.start_url, "/robots.txt")
        self.robot_parser = robotparser.RobotFileParser()
        self.robot_parser.set_url(robots_url)
        self.robot_parser.read()

    def run(self) -> dict:
        """
        Executes the main crawling loop and returns the crawled data.
        """
        logging.info(
            f"Starting crawl for {self.start_url} with domain lock on '{self.start_domain}'."
        )

        while self.url_queue and len(self.visited_urls) < self.max_pages:
            current_url = self.url_queue.pop(0)

            if current_url in self.visited_urls:
                continue

            self.visited_urls.add(current_url)
            logging.info(
                f"Crawling ({len(self.visited_urls)}/{self.max_pages}): {current_url}"
            )

            if not self.robot_parser.can_fetch("*", current_url):
                logging.warning(f"Blocked by robots.txt: {current_url}")
                continue

            try:
                response = requests.get(current_url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")
                clean_text = (
                    soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                )
                clean_text = self._clean_text(clean_text)
                self.crawled_data[current_url] = clean_text

                self._find_and_queue_links(soup, current_url)

            except requests.RequestException as e:
                logging.error(f"Failed to fetch {current_url}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred at {current_url}: {e}")

            time.sleep(self.delay_seconds)

        logging.info(f"Crawl finished. Visited {len(self.visited_urls)} pages.")
        return self.crawled_data

    def _find_and_queue_links(self, soup: BeautifulSoup, current_url: str):
        """Finds, cleans, validates, and queues new links from a page."""
        for link_tag in soup.select("a[href]"):
            raw_link = link_tag.get("href")

            # --- FIX FOR THE TypeError ---
            # Ensure raw_link is a string before proceeding
            if raw_link and isinstance(raw_link, str):
                abs_url = urljoin(current_url, raw_link)
                # Clean the URL (remove fragments like #section)
                abs_url = urlparse(abs_url)._replace(fragment="").geturl()

                # CRITICAL: Check if link is in the same domain and not yet seen
                if (
                    urlparse(abs_url).netloc == self.start_domain
                    and abs_url not in self.visited_urls
                    and abs_url not in self.url_queue
                ):
                    self.url_queue.append(abs_url)
