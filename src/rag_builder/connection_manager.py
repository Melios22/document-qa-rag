"""
Milvus Connection Manager
========================

Simple connection manager for Milvus with Docker support and HNSW indexing.
"""

from typing import Optional, Tuple

from pymilvus import MilvusClient

from ..constant import (
    DENSE_INDEX_CONFIG,
    DENSE_INDEX_FALLBACK_CONFIG,
    DENSE_SEARCH_FALLBACK_PARAMS,
    DENSE_SEARCH_PARAMS,
    MILVUS_DOCKER_URI,
    MILVUS_URI,
    USE_DOCKER_MILVUS,
)


class MilvusConnectionManager:
    """Manages Milvus connections with Docker support"""

    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self.supports_hnsw = False

    def get_client(self) -> Tuple[MilvusClient, bool]:
        """Get Milvus client with automatic connection management"""
        if self.client is not None:
            return self.client, self.supports_hnsw

        # Try Docker connection first if enabled
        if USE_DOCKER_MILVUS:
            try:
                print("🐳 Connecting to Docker Milvus...")
                self.client = MilvusClient(uri=MILVUS_DOCKER_URI)
                # Test connection
                self.client.list_collections()
                self.supports_hnsw = True
                print("✅ Docker Milvus connected (HNSW enabled)")
                return self.client, True
            except Exception as e:
                print(f"⚠️ Docker connection failed: {e}")

        # Fallback to local file-based connection
        print("📁 Using local file-based Milvus...")
        self.client = MilvusClient(uri=str(MILVUS_URI))
        self.supports_hnsw = False
        print("✅ Local Milvus connected")
        return self.client, False

    def get_index_config(self):
        """Get appropriate index configuration"""
        if self.supports_hnsw:
            return DENSE_INDEX_CONFIG, DENSE_SEARCH_PARAMS
        else:
            return DENSE_INDEX_FALLBACK_CONFIG, DENSE_SEARCH_FALLBACK_PARAMS


# Global connection manager
_connection_manager = MilvusConnectionManager()


def get_milvus_client() -> Tuple[MilvusClient, bool]:
    """Get the global Milvus client instance"""
    return _connection_manager.get_client()


def get_index_config():
    """Get appropriate index configuration for current connection"""
    return _connection_manager.get_index_config()
