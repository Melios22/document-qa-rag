import gc
import re
from typing import Any, Dict

from ..constant import CHARS_PER_TOKEN


class TextCleaner:
    """Simple text cleaner"""

    def __init__(self):
        # Vietnamese diacritics set for basic detection
        self.vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ"

    def clean_text(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Clean text and return basic metadata"""
        if not text or not text.strip():
            return "", {"was_empty": True}

        gc.collect()
        original_length = len(text)

        # Basic text cleaning
        text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces and tabs
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # Normalize paragraph breaks
        text = re.sub(r"[.]{3,}", "...", text)  # Multiple periods
        text = re.sub(r"[-]{3,}", "---", text)  # Multiple dashes
        text = re.sub(r"[,]{2,}", ",", text)  # Multiple commas

        # Keep essential characters and punctuation
        vietnamese_escaped = re.escape(self.vietnamese_chars)
        allowed_pattern = f"[a-zA-Z0-9{vietnamese_escaped}\\s.,;:!?()\\[\\]{{}}\"'\\-_/\\\\+=%%&@#$\\n\\t]"
        text = "".join(char for char in text if re.match(allowed_pattern, char))

        cleaned_text = text.strip()

        # Basic metadata
        cleaning_metadata = {
            "was_empty": False,
            "original_length": original_length,
            "cleaned_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }

        return cleaned_text, cleaning_metadata

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        if not text:
            return 0
        return len(text) // CHARS_PER_TOKEN + 1
