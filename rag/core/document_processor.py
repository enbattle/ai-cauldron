"""
Document processing module for PDF parsing and text chunking.

Supports both naive (fixed-size) and production (semantic) chunking strategies.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal

import pdfplumber
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    text: str
    chunk_id: int
    metadata: dict


@dataclass
class ProcessedDocument:
    """Container for a fully processed document."""

    chunks: List[DocumentChunk]
    metadata: dict
    total_chunks: int


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict = None) -> List[DocumentChunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunking(ChunkingStrategy):
    """Naive chunking: fixed character count, no semantic awareness."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict = None) -> List[DocumentChunk]:
        """Split text into fixed-size chunks."""
        if metadata is None:
            metadata = {}

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata={**metadata, "start_pos": start, "end_pos": end},
                    )
                )
                chunk_id += 1

            start = end - self.chunk_overlap

        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    Production chunking: respects sentence boundaries and semantic units.

    Attempts to create chunks that preserve meaning by:
    - Not splitting mid-sentence
    - Respecting paragraph boundaries
    - Maintaining min/max chunk sizes
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter (can be enhanced with spaCy/nltk)."""
        # Basic sentence splitting on common punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: dict = None) -> List[DocumentChunk]:
        """Split text into semantic chunks respecting sentence boundaries."""
        if metadata is None:
            metadata = {}

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds max_chunk_size, finalize current chunk
            if current_size + sentence_len > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata={**metadata, "num_sentences": len(current_chunk)},
                    )
                )
                chunk_id += 1

                # Handle overlap: keep last few sentences
                if self.chunk_overlap > 0:
                    overlap_text = ""
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if len(overlap_text) + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_text = " ".join(overlap_sentences)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space

            # If we've reached target chunk_size, consider finalizing
            if current_size >= self.chunk_size:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            metadata={**metadata, "num_sentences": len(current_chunk)},
                        )
                    )
                    chunk_id += 1

                    # Handle overlap
                    if self.chunk_overlap > 0:
                        overlap_text = ""
                        overlap_sentences = []
                        for s in reversed(current_chunk):
                            if len(overlap_text) + len(s) <= self.chunk_overlap:
                                overlap_sentences.insert(0, s)
                                overlap_text = " ".join(overlap_sentences)
                            else:
                                break
                        current_chunk = overlap_sentences
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0

        # Add any remaining text as final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata={**metadata, "num_sentences": len(current_chunk)},
                    )
                )

        return chunks


class DocumentProcessor:
    """Main document processor supporting multiple PDF parsers and chunking strategies."""

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        pdf_parser: Literal["pypdf", "pdfplumber"] = "pdfplumber",
    ):
        self.chunking_strategy = chunking_strategy
        self.pdf_parser = pdf_parser

    def _extract_text_pypdf(self, pdf_path: str) -> tuple[str, dict]:
        """Extract text using PyPDF."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        metadata = {
            "num_pages": len(reader.pages),
            "parser": "pypdf",
        }
        return text, metadata

    def _extract_text_pdfplumber(self, pdf_path: str) -> tuple[str, dict]:
        """Extract text using pdfplumber (better for tables/complex layouts)."""
        text = ""
        num_pages = 0

        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        metadata = {
            "num_pages": num_pages,
            "parser": "pdfplumber",
        }
        return text, metadata

    def process_pdf(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a PDF file: extract text and chunk it.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            ProcessedDocument containing chunks and metadata
        """
        # Extract text
        if self.pdf_parser == "pypdf":
            text, metadata = self._extract_text_pypdf(pdf_path)
        else:
            text, metadata = self._extract_text_pdfplumber(pdf_path)

        metadata["source_file"] = pdf_path

        # Clean text
        text = self._clean_text(text)

        # Chunk text
        chunks = self.chunking_strategy.chunk(text, metadata)

        return ProcessedDocument(
            chunks=chunks,
            metadata=metadata,
            total_chunks=len(chunks),
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text: remove excess whitespace, fix encoding issues."""
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


def create_processor(
    strategy: Literal["fixed", "semantic"],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    **kwargs,
) -> DocumentProcessor:
    """
    Factory function to create a DocumentProcessor with the specified strategy.

    Args:
        strategy: 'fixed' for naive chunking, 'semantic' for production
        chunk_size: Target size for chunks
        chunk_overlap: Overlap between chunks
        **kwargs: Additional parameters for chunking strategy

    Returns:
        Configured DocumentProcessor
    """
    if strategy == "fixed":
        chunking_strategy = FixedSizeChunking(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "semantic":
        chunking_strategy = SemanticChunking(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", 2000),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return DocumentProcessor(
        chunking_strategy=chunking_strategy,
        pdf_parser=kwargs.get("pdf_parser", "pdfplumber"),
    )
