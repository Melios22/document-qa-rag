"""
Logging utility for the RAG system.
Sets up structured logging according to config.json specifications.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.constant import (
    LOG_FILE_BUILD,
    LOG_FILE_ERRORS,
    LOG_FILE_PREPROCESS,
    LOG_FILE_RETRIEVAL,
    PROJECT_ROOT,
    config,
)


class RAGLogger:
    """Centralized logging for the RAG system"""

    _instances = {}
    _qa_log_file = PROJECT_ROOT / "logs" / "qa_history.log"

    def __new__(cls, name: str):
        if name not in cls._instances:
            cls._instances[name] = super(RAGLogger, cls).__new__(cls)
            cls._instances[name]._initialized = False
        return cls._instances[name]

    def __init__(self, name: str):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self._initialized = True

    def _setup_logger(self):
        """Set up logger with appropriate handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set logging level from config
        log_level = getattr(
            logging, config["logging"]["level"].upper(), logging.WARNING
        )
        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(config["logging"]["format"])

        # Determine log file based on logger name
        if "preprocess" in self.name.lower():
            log_file = LOG_FILE_PREPROCESS
        elif "build" in self.name.lower() or "rag" in self.name.lower():
            log_file = LOG_FILE_BUILD
        elif "retriev" in self.name.lower() or "search" in self.name.lower():
            log_file = LOG_FILE_RETRIEVAL
        else:
            log_file = LOG_FILE_ERRORS

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Console handler for errors only
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        console_handler.setLevel(logging.ERROR)
        self.logger.addHandler(console_handler)

        # Error handler
        if log_file != LOG_FILE_ERRORS:
            error_handler = logging.FileHandler(LOG_FILE_ERRORS, encoding="utf-8")
            error_handler.setFormatter(formatter)
            error_handler.setLevel(logging.ERROR)
            self.logger.addHandler(error_handler)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

    def milestone(self, message: str, **kwargs):
        """Log important milestones - always logged regardless of level"""
        # Log to file at critical level
        self.logger.critical(f"MILESTONE: {message}", **kwargs)

        # Show milestone to console with visual indicator
        print(f"✅ {message}")

    @classmethod
    def log_qa_pair(
        cls, question: str, answer: str, retrieval_info: Optional[Dict[str, Any]] = None
    ):
        """Log question-answer pairs with metadata"""
        # Ensure QA log directory exists
        cls._qa_log_file.parent.mkdir(parents=True, exist_ok=True)

        qa_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "retrieval_info": retrieval_info or {},
        }

        with open(cls._qa_log_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(qa_entry, ensure_ascii=False, indent=2)
                + "\n"
                + "=" * 80
                + "\n"
            )


def get_logger(name: str) -> RAGLogger:
    """Get a logger instance for the specified component"""
    return RAGLogger(name)


# Create specialized loggers for different components
preprocess_logger = get_logger("rag.preprocess")
build_logger = get_logger("rag.build")
retrieval_logger = get_logger("rag.retrieval")
main_logger = get_logger("rag.main")
