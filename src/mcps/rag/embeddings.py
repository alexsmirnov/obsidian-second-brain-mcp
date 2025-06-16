"""
Embedding service implementations for the RAG search system.
"""

import logging
import asyncio
from typing import List, Optional
import numpy as np

from .interfaces import IEmbeddingService

logger = logging.getLogger("mcps")


class HuggingFaceEmbedding(IEmbeddingService):
    """HuggingFace embedding service using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._model_loaded = False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self._model_loaded:
            await self._load_model()
        
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self._generate_embedding_sync, 
                text
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            # Return zero vector as fallback
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding generation."""
        return self.model.encode(text, convert_to_numpy=True)
    
    async def _load_model(self):
        """Load the embedding model."""
        try:
            # Import here to avoid dependency issues if not installed
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                self.model_name
            )
            self._model_loaded = True
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise


class OpenAIEmbedding(IEmbeddingService):
    """OpenAI embedding service."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API."""
        if not self.client:
            await self._initialize_client()
        
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Default dimension for text-embedding-3-small
    
    async def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            import os
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
            
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise


class MockEmbedding(IEmbeddingService):
    """Mock embedding service for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash."""
        # Simple hash-based mock embedding
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        embedding = []
        for i in range(0, min(len(text_hash), self.dimension * 2), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        return embedding[:self.dimension]


class BatchEmbeddingService:
    """Wrapper for batch embedding generation."""
    
    def __init__(self, embedding_service: IEmbeddingService, batch_size: int = 32):
        self.embedding_service = embedding_service
        self.batch_size = batch_size
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Processing embedding batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            # Process batch concurrently
            batch_tasks = [
                self.embedding_service.generate_embedding(text)
                for text in batch
            ]
            
            batch_embeddings = await asyncio.gather(*batch_tasks)
            embeddings.extend(batch_embeddings)
        
        return embeddings