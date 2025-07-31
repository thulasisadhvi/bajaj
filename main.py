from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Union
import asyncio
import time
from datetime import datetime
import re
import json

# Local imports
from document_reader import read_document_from_url
from chunks import get_chunks
from embedder import Embedder
from llm_reasoner import LLMReasoner
from config import settings

# --- Pydantic Models (Define them early) ---
class DocumentInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerOutput(BaseModel):
    answers: List[str]
    status: str
    processing_info: Dict[str, Union[int, float]]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Processes large documents and makes contextual decisions for insurance, legal, HR, and compliance domains.",
    version="1.0.0",
)

# Global instances for efficiency
# These will be initialized once when the app starts
embedder = Embedder()
llm_reasoner = LLMReasoner()

print(f"MAIN.PY LOADED AT {datetime.now()}")

# --- FastAPI Lifespan Events to ensure httpx client is closed ---
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down and closing httpx client...")
    await embedder.close()


# --- Authentication Dependency ---
async def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing or invalid.")
    
    bearer_token = token.split(" ")[1]
    if bearer_token != settings.AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication token.")
    return True

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerOutput, summary="Run Queries on Documents")
async def run_submission(
    input_data: DocumentInput,
    auth_verified: bool = Depends(verify_token)
):
    """
    Processes a document from a URL, chunks it, embeds the chunks using Pinecone,
    performs semantic search for given questions, and uses an LLM to generate contextual answers.
    """
    start_time = time.perf_counter()

    document_url = str(input_data.documents)
    questions = input_data.questions

    print(f"Processing document from URL: {document_url}")

    # 1. Input Documents & Text Extraction (Handles various formats)
    text_extraction_start = time.perf_counter()
    text_content = await read_document_from_url(document_url)
    text_extraction_time = time.perf_counter() - text_extraction_start

    if not text_content:
        raise HTTPException(status_code=400, detail="Could not extract text from the provided document. Check URL or document type.")

    document_length_chars = len(text_content)
    print(f"Extracted {document_length_chars} characters from document.")

    # 2. Chunks
    chunking_start = time.perf_counter()
    document_chunks = get_chunks(text_content)
    chunking_time = time.perf_counter() - chunking_start

    if not document_chunks:
        raise HTTPException(status_code=400, detail="Could not create meaningful chunks from the document content.")
    num_chunks = len(document_chunks)
    print(f"Created {num_chunks} chunks.")

    # 3. Embedding Search (Pinecone upsert for this specific document's chunks)
    # REMOVED THE INDEX DELETION LOGIC HERE TO IMPROVE PERFORMANCE
    # The embedder is initialized once globally. It will create the index if it doesn't exist.
    # Subsequent calls will just upsert into the existing index.
    
    embedding_upsert_start = time.perf_counter()
    await embedder.embed_and_upsert_documents(document_chunks) # AWAIT HERE
    embedding_upsert_time = time.perf_counter() - embedding_upsert_start
    print("Document chunks embedded and upserted to Pinecone.")

    # 4. Process each question concurrently
    all_answers: List[str] = []
    question_processing_times: List[float] = []
    
    async def process_single_question(question: str):
        q_start_time = time.perf_counter()
        print(f"\n--- Processing Question: '{question}' ---")
        
        # 5. Clause Matching (Semantic similarity via Pinecone)
        retrieval_start = time.perf_counter()
        relevant_chunks_with_scores = await embedder.search(question, k=settings.TOP_K_CHUNKS) # AWAIT HERE
        relevant_chunks_text = [chunk for chunk, score in relevant_chunks_with_scores]
        retrieval_time = time.perf_counter() - retrieval_start

        if not relevant_chunks_text:
            print(f"No relevant chunks found for question: '{question}'")
            return f"The system could not find relevant information for the question: '{question}'.", (time.perf_counter() - q_start_time)
        
        print(f"Found {len(relevant_chunks_text)} relevant chunks.")

        # 6. Logic Evaluation (LLM Reasoning)
        llm_reasoning_start = time.perf_counter()
        answer_response = await llm_reasoner.get_reasoned_answer(question, relevant_chunks_text) # AWAIT HERE
        llm_reasoning_time = time.perf_counter() - llm_reasoning_start
        
        final_answer_text = f"The system could not generate a clear answer for: '{question}'."

        if isinstance(answer_response, dict) and "answer" in answer_response:
            final_answer_text = answer_response["answer"]
        elif isinstance(answer_response, str):
            json_match = re.search(r'\{[\s\S]*\}', answer_response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_dict = json.loads(json_str)
                    if "answer" in parsed_dict:
                        final_answer_text = parsed_dict["answer"]
                    else:
                        print(f"Warning: JSON extracted but no 'answer' key for question '{question}'. JSON: {json_str}")
                        final_answer_text = f"An error occurred while extracting the answer for question: '{question}' (missing 'answer' key in LLM JSON)."
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse extracted JSON for question '{question}': {e} - String: {json_str}")
                    final_answer_text = f"An error occurred while generating the answer for question: '{question}' (JSON parsing failed)."
            else:
                print(f"Warning: LLMReasoner returned a string without parsable JSON for question '{question}': {answer_response[:200]}...")
                final_answer_text = f"An error occurred while generating the answer for question: '{question}' (unexpected LLM response format)."
        else:
            print(f"Warning: Unexpected type from LLMReasoner for question '{question}': {type(answer_response)}")
            final_answer_text = f"An error occurred while generating the answer for question: '{question}' (unexpected LLM response type)."
        
        return final_answer_text, (time.perf_counter() - q_start_time)

    # Create and run all question processing tasks concurrently
    tasks = [process_single_question(q) for q in questions]
    results = await asyncio.gather(*tasks)

    # Unpack the results
    for answer, q_proc_time in results:
        all_answers.append(answer)
        question_processing_times.append(q_proc_time)

    total_processing_time = time.perf_counter() - start_time

    print("\n--- All questions processed. ---")

    # Prepare processing information dictionary
    processing_info = {
        "total_processing_time_ms": round(total_processing_time * 1000, 2),
        "document_length_chars": document_length_chars,
        "num_chunks": num_chunks,
        "num_questions_processed": len(questions),
        "text_extraction_time_ms": round(text_extraction_time * 1000, 2),
        "chunking_time_ms": round(chunking_time * 1000, 2),
        "embedding_upsert_time_ms": round(embedding_upsert_time * 1000, 2),
        "avg_question_processing_time_ms": round(sum(question_processing_times) / len(question_processing_times) * 1000, 2) if question_processing_times else 0,
    }
    
    if total_processing_time > 30:
        print(f"WARNING: Response time exceeded 30 seconds: {total_processing_time:.2f} seconds.")

    return JSONResponse(content={
        "answers": all_answers,
        "status": "success",
        "processing_info": processing_info
    })

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "LLM-Powered Query-Retrieval System API. Visit /docs for OpenAPI documentation."}