import gc
from pathlib import Path
from typing import List

import torch
from docling.chunking import HybridChunker
from langchain.schema import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from tqdm.auto import tqdm

from ..constant import (
    CHUNKER_MODEL,
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    PDF_INPUT_DIR,
    PROCESSING_SESSION_ID,
)
from ..utils.logging import get_logger

logger = get_logger("rag.preprocess.pdf")
from .cleaner import TextCleaner
from .metadata_gen import ChunkMetadataGenerator


def process_pdfs() -> List[Document]:
    """PDF processing with BAAI BGE-M3 and enhanced error handling"""

    # Memory optimization: Force CPU usage and clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    text_cleaner = TextCleaner()
    metadata_generator = ChunkMetadataGenerator(PROCESSING_SESSION_ID)

    # Find PDF files
    pdf_files = list(Path(PDF_INPUT_DIR).glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files for processing")

    if not pdf_files:
        logger.error(f"No PDF files found in {PDF_INPUT_DIR}")
        return []

    all_documents = []

    # Initialize chunker with BAAI BGE-M3
    logger.info(f"Initializing chunker with {CHUNKER_MODEL}")
    chunker = None
    try:
        chunker = HybridChunker(
            tokenizer=CHUNKER_MODEL,
            max_tokens=MAX_CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
        )
        logger.milestone("Chunker initialized successfully")

    except Exception as e:
        logger.error(f"Chunker initialization failed: {e}")
        logger.error(
            "Please check if the model is available or use a different tokenizer"
        )
        return []

    # Process each PDF file
    pbar = tqdm(total=len(pdf_files), desc="Processing PDFs")

    for pdf_file in pdf_files:
        pbar.set_description(f"Processing {pdf_file.name}")
        pdf_file_str = str(pdf_file)

        try:
            # Use DoclingLoader with chunker
            loader = DoclingLoader(
                file_path=pdf_file_str,
                export_type=ExportType.DOC_CHUNKS,
                chunker=chunker,
            )

            # Load documents - already chunked
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} chunks from {pdf_file.name}")

            # Process each chunk
            file_chunks_processed = 0

            for i, doc in enumerate(docs):
                # Clean text
                cleaned_content, cleaning_metadata = text_cleaner.clean_text(
                    doc.page_content
                )

                if cleaned_content and len(cleaned_content) > 50:
                    estimated_tokens = text_cleaner.estimate_tokens(cleaned_content)

                    # Create metadata
                    metadata = metadata_generator.create_metadata(
                        doc=doc,
                        source_file=pdf_file_str,
                        chunk_index=i,
                        cleaning_metadata=cleaning_metadata,
                        estimated_tokens=estimated_tokens,
                    )

                    # Update document with cleaned content and metadata
                    doc.page_content = cleaned_content
                    doc.metadata = metadata

                    all_documents.append(doc)
                    file_chunks_processed += 1

            del loader
            logger.info(f"{pdf_file.name}: {file_chunks_processed} chunks created")
            pbar.update(1)

        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            pbar.update(1)
            continue

    del chunker
    pbar.close()

    logger.milestone(f"Processing Complete: {len(all_documents)} total chunks created")
    return all_documents
