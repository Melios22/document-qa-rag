import os
from typing import List, Tuple

from dotenv import load_dotenv

# import google as genai
from google.genai import Client, types
from langchain.schema import Document

# IBM Watson imports
from langchain_ibm import WatsonxRerank, WatsonxLLM as Wastonx


class GeminiLLM:
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        load_dotenv()
        self.model_id = model_id
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("API key is required")
        self.llm = Client(api_key=self.api_key)
        self.config = types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(
                max_steps=5,
                stop_sequences=["\n"],
            ),
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.llm.models.generate_content(
                model=self.model_id,
                prompt=prompt,
                config=self.config,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error generating response: {e}"

class WatsonxLLM:
    def __init__(
        self,
        model_id: str = "",
        project_id: str = None,
        api_key: str = None,
        url: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        load_dotenv()
        self.model_id = model_id
        self.project_id = project_id or os.getenv("WASTONX_PROJECT_ID")
        self.api_key = api_key or os.getenv("WASTONX_API")
        self.url = url or os.getenv("WASTONX_URL")
        
        self.llm = Wastonx(
            model_id=self.model_id,
            url=self.url,
            apikey=self.api_key,
            project_id=self.project_id,
            params={
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
            },
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"❌ IBM LLM error: {e}")
            return f"Error generating response: {e}"


class WatsonxReRanker:
    def __init__(self, model_id: str, project_id: str=None, api_key: str=None, url: str=None):
        load_dotenv()
        self.model_id = model_id
        self.project_id = project_id or os.getenv("WASTONX_PROJECT_ID")
        self.api_key = api_key or os.getenv("WASTONX_API")
        self.url = url or os.getenv("WASTONX_URL")
        
        if not self.project_id or not self.api_key or not self.url:
            raise ValueError("Project ID, API key, and URL are required for Watsonx ReRanker")
        
        self.reranker = WatsonxRerank(
            model_id=self.model_id,
            project_id=self.project_id,
            api_key=self.api_key,
            url=self.url
        )

    def rerank_documents(self, query: str, documents: List[Document] | List[Tuple[Document, float]], k: int = 5) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        elif isinstance(documents[0], tuple):
            documents = [doc for doc, _ in documents]
            
        try:
            reranked_docs = self.reranker.compress_documents(
                documents=documents,
                query=query,
            )

            # Return top-k results
            # IBM reranker doesn't return explicit scores, so if needed, you can use rank-based scoring
            return [(doc, 1.0) for doc in reranked_docs[:k]]
        except Exception as e:
            print(f"❌ Watsonx ReRanker error: {e}")
            return []
        
        