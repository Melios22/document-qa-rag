"""
BGE-M3 Encoder (dense + sparse)
===============================

Lightweight wrapper around FlagEmbedding's BGEM3FlagModel to return both
dense and sparse representations as shown in the instructions (dense_vecs, sparse_vecs).
"""

from typing import Dict, List, Optional, Union

from FlagEmbedding import BGEM3FlagModel


class BGEM3Encoder:
    """Encode text with BGE-M3 producing dense and sparse vectors."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        use_fp16: bool = False,  # Set to False to avoid dtype mismatches
        max_length: int = 512,
        batch_size: int = 32,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = BGEM3FlagModel(
            model,
            device=device,
            use_fp16=use_fp16,
            trust_remote_code=trust_remote_code,
        )

    def encode(
        self, text_or_texts: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Encode text(s) to dense and sparse vectors.

        Args:
            text_or_texts: Single text string or list of texts
            batch_size: Batch size for encoding

        Returns:
            Dictionary with 'dense_vecs' and 'lexical_weights' keys
        """
        try:
            if isinstance(text_or_texts, str):
                texts = [text_or_texts]
            else:
                texts = text_or_texts

            if not texts:
                raise ValueError("No texts provided for encoding")

            out = self.model.encode(
                sentences=texts,
                batch_size=batch_size or self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            # Ensure consistent data types (convert to float32 if needed)
            if "dense_vecs" in out:
                import numpy as np

                out["dense_vecs"] = out["dense_vecs"].astype(np.float32)

            # Validate output structure
            if "dense_vecs" not in out or "lexical_weights" not in out:
                raise ValueError("Invalid output from BGE-M3 model")

            # out keys: "dense_vecs": np.ndarray [n, d], "lexical_weights": List[Dict[token_id->weight]]
            return out

        except Exception as e:
            print(f"❌ Error encoding with BGE-M3: {e}")
            # Return empty structure on error
            import numpy as np

            return {
                "dense_vecs": np.array([]).astype(np.float32),
                "lexical_weights": [],
            }

    def encode_query(self, text: str) -> Dict[str, List]:
        """Encode a single query text."""
        return self.encode(text)
