import datetime
import gc
import hashlib
from pathlib import Path
from typing import Any, Dict

from langchain.schema import Document

from ..constant import CHUNKER_MODEL


class ChunkMetadataGenerator:
    """Generates core metadata for chunks"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chunk_counter = 0

    def generate_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
        return f"CHUNK_{self.session_id}_{file_hash}_{chunk_index:04d}"

    def generate_document_id(self, source_file: str) -> str:
        """Generate unique document ID"""
        file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
        return f"DOC_{self.session_id}_{file_hash}"

    def create_metadata(
        self,
        doc: Document,
        source_file: str,
        chunk_index: int,
        cleaning_metadata: Dict[str, Any],
        estimated_tokens: int,
    ) -> Dict[str, Any]:
        """Create chunk metadata"""
        gc.collect()
        self.chunk_counter += 1

        # Generate IDs
        chunk_id = self.generate_chunk_id(source_file, chunk_index)
        document_id = self.generate_document_id(source_file)

        # Extract file information
        file_path = Path(source_file)
        content = doc.page_content
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Core metadata
        metadata = {
            # Essential IDs
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "global_index": self.chunk_counter,
            # Processing info
            "session_id": self.session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "chunker_model": CHUNKER_MODEL,
            # Source info
            "source_file": source_file,
            "source_filename": str(file_path.name),
            # Content metrics
            "content_length": len(content),
            "estimated_tokens": estimated_tokens,
            "word_count": cleaning_metadata.get("word_count", 0),
            "content_hash": content_hash,
            # # Future use placeholders
            # "summary": None,
            # "keywords": None,
            # "embedding_id": None,
            # "last_accessed": None,
        }

        return metadata
