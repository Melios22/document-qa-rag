import gc
import re
import unicodedata
from typing import Any, Dict

from ..constant import CHARS_PER_TOKEN


class TextCleaner:
    """Vietnamese text cleaner optimized for RAG preprocessing."""

    def __init__(self):
        # Allow Vietnamese letters + digits + punctuation
        self.vietnamese_chars = (
            "àáạảãâầấậẩẫăằắặẳẵ"
            "èéẹẻẽêềếệểễ"
            "ìíịỉĩ"
            "òóọỏõôồốộổỗơờớợởỡ"
            "ùúụủũưừứựửữ"
            "ỳýỵỷỹ"
            "đ"
            "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
            "ÈÉẸẺẼÊỀẾỆỂỄ"
            "ÌÍỊỈĨ"
            "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
            "ÙÚỤỦŨƯỪỨỰỬỮ"
            "ỲÝỴỶỸ"
            "Đ"
        )

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode (NFC) for consistency."""
        return unicodedata.normalize("NFC", text)

    def clean_text(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Clean text and return metadata for RAG preparation."""
        if not text or not text.strip():
            return "", {"was_empty": True}

        gc.collect()
        original_length = len(text)

        # Unicode normalization
        text = self.normalize_unicode(text)

        # Basic noise cleanup
        text = re.sub(r"[ \t]+", " ", text)  # collapse spaces/tabs
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # normalize paragraph breaks
        text = re.sub(r"[.]{3,}", "...", text)  # reduce ellipses
        text = re.sub(r"[-]{3,}", "—", text)  # convert long dashes
        text = re.sub(r"[,]{2,}", ",", text)  # collapse commas

        # Remove stray non-text characters (OCR artifacts, control chars)
        vietnamese_escaped = re.escape(self.vietnamese_chars)
        allowed_pattern = f"[a-zA-Z0-9{vietnamese_escaped}\\s.,;:!?()\\[\\]{{}}\"'\\-_/\\\\+=%&@#$\\n\\t]"
        text = "".join(ch for ch in text if re.match(allowed_pattern, ch))

        # Normalize line breaks around punctuation (fix OCR issues)
        text = re.sub(r"\s*([.,;:!?])\s*", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        cleaned_text = text.strip()

        # Metadata for tracking
        cleaning_metadata = {
            "was_empty": False,
            "original_length": original_length,
            "cleaned_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "paragraphs": cleaned_text.count("\n\n") + 1,
            "estimated_tokens": self.estimate_tokens(cleaned_text),
        }

        return cleaned_text, cleaning_metadata

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (tune per tokenizer)."""
        if not text:
            return 0
        return len(text) // CHARS_PER_TOKEN + 1
