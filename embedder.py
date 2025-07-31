import os
import uuid
from pinecone import Pinecone, ServerlessSpec
import httpx # Changed from requests
import asyncio # Added for asyncio.gather
from typing import List, Tuple

from config import settings

class Embedder:
    def __init__(self):
        self.pinecone_api_key = settings.PINECONE_API_KEY
        self.pinecone_environment = settings.PINECONE_ENVIRONMENT
        self.index_name = settings.PINECONE_INDEX_NAME
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.ollama_base_url = "http://localhost:11434/api/embeddings"
        self.index = None
        
        # The dimension for Ollama's all-minilm is 384.
        # If you use nomic-embed-text, it's 768.
        # Ensure this matches the model you pull with Ollama!
        self.dimension = 384 # IMPORTANT: This must match your chosen Ollama embedding model's dimension

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in .env or config.")
        if not self.pinecone_environment:
            raise ValueError("PINECONE_ENVIRONMENT is not set in .env or config.")

        # Initialize httpx.AsyncClient here for reuse across requests
        self._httpx_client = httpx.AsyncClient(timeout=60.0)

        self._init_pinecone()

    def _init_pinecone(self):
        """Initializes Pinecone and creates the index if it doesn't exist."""
        try:
            pc = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            
            if self.index_name not in [idx.name for idx in pc.list_indexes().indexes]:
                print(f"Creating Pinecone index '{self.index_name}'...")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Pinecone index '{self.index_name}' created.")
            else:
                print(f"Pinecone index '{self.index_name}' already exists.")
            
            self.index = pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    async def close(self):
        """Closes the underlying httpx client connection."""
        await self._httpx_client.aclose()

    async def _get_ollama_embedding(self, text: str) -> List[float]: # Made async
        """Sends text to local Ollama server for embedding generation."""
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            response = await self._httpx_client.post(self.ollama_base_url, json=payload) # Use await with self._httpx_client
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            embedding = [float(val) for val in response.json()["embedding"]]
            return embedding
        except httpx.RequestError as e: # Catch httpx-specific errors
            print(f"Error calling Ollama embedding API for text: '{text[:50]}...' Error: {e}")
            raise

    async def embed_and_upsert_documents(self, chunks: List[str]): # Made async
        """
        Embeds a list of text chunks concurrently and upserts them to the Pinecone index.
        Each chunk is stored with its original text as metadata.
        """
        if not chunks:
            return

        print(f"Embedding {len(chunks)} chunks and upserting to Pinecone...")
        vectors_to_upsert = []

        # Concurrently get embeddings for all chunks
        embedding_tasks = [self._get_ollama_embedding(chunk) for chunk in chunks]
        try:
            embeddings = await asyncio.gather(*embedding_tasks) # Await all embedding tasks
        except Exception as e:
            print(f"Error during concurrent embedding generation: {e}")
            raise # Re-raise to indicate a failure

        for i, chunk in enumerate(chunks):
            vector_id = str(uuid.uuid4())
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[i], # Use the already fetched embedding
                "metadata": {"text": chunk}
            })
        
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {e}")
                
        print(f"Finished upserting {len(chunks)} chunks to Pinecone.")

    async def search(self, query: str, k: int = settings.TOP_K_CHUNKS) -> List[Tuple[str, float]]: # Made async
        """
        Performs a semantic search for the query in the Pinecone index.

        Args:
            query (str): The natural language query.
            k (int): The number of top similar chunks to retrieve.

        Returns:
            List[Tuple[str, float]]: A list of (chunk_text, similarity_score) for the top k chunks.
        """
        if self.index is None:
            print("Pinecone index not initialized.")
            return []

        query_embedding = await self._get_ollama_embedding(query) # Await here as _get_ollama_embedding is now async

        try:
            query_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )

            results = []
            for match in query_results.matches:
                chunk_text = match.metadata.get("text", "")
                score = match.score
                if chunk_text:
                    results.append((chunk_text, score))
            
            return results
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return [] # IMPORTANT: Ensure an empty list is returned on error

# Example Usage (for testing) - updated to use async
if __name__ == "__main__":
    async def test_embedder():
        # Ensure you have PINECONE_API_KEY and PINECONE_ENVIRONMENT set in your .env
        # And a Pinecone index named 'hackrx-policy-index' (or change in config.py)

        embedder = Embedder()
        sample_chunks = [
            "The policy covers medical expenses for inpatient treatment under Ayurveda.",
            "A grace period of thirty days is provided for premium payment after the due date.",
            "This document outlines the terms and conditions for personal loans.",
            "Maternity expenses are covered subject to a 24-month continuous coverage period.",
            "The waiting period for pre-existing diseases is thirty-six months from inception."
        ]
        
        await embedder.embed_and_upsert_documents(sample_chunks)

        query = "Does the plan cover childbirth?"
        top_chunks = await embedder.search(query, k=2) # Await search
        print("\nTop chunks for query:")
        for chunk, score in top_chunks:
            print(f"Score: {score:.4f}, Chunk: {chunk}")

        await embedder.close() # Close the client after testing

    import asyncio
    asyncio.run(test_embedder())