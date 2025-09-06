"""
RAG System Preprocessor - Document Processing Stage
===================================================

Process PDF documents and prepare them for RAG system.
All configuration is loaded from config.json.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src import (
    HAS_PREPROCESS,
    preview_documents,
    process_pdfs,
    require_preprocess,
    save_documents,
)
from src.utils.logging import preprocess_logger as logger

# Ensure we have preprocessing components
if not HAS_PREPROCESS:
    require_preprocess()


def main():
    """Main preprocessing pipeline"""
    logger.milestone("Starting RAG Document Preprocessing")

    try:
        logger.info("Beginning PDF document processing")
        docs = process_pdfs()

        if not docs:
            logger.error("No documents were processed successfully")
            return

        logger.info(f"Successfully processed {len(docs)} documents")

        logger.info("Saving processed documents")
        save_documents(documents=docs)

        logger.info("Generating document preview")
        preview_documents(documents=docs)

        logger.milestone(
            f"Document preprocessing completed successfully - {len(docs)} documents ready"
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
