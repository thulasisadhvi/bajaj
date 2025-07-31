import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(__file__), '.env'), extra='ignore')

    # --- CRITICAL UPDATES HERE FOR PERFORMANCE ---
    LLM_MODEL_NAME: str = "phi3:mini"          # <-- Change this to phi3:mini for CPU optimization
    EMBEDDING_MODEL_NAME: str = "all-minilm" # <-- Change this to the CPU-only version you created
    TOP_K_CHUNKS: int = 5

    AUTH_TOKEN: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "hackrx-policy-index"
    TOGETHER_API_KEY: str

print(f"Current working directory during config load: {os.getcwd()}")
print(f"Does .env file exist in CWD? {os.path.exists('.env')}")

try:
    settings = Settings()
    print("\n--- Settings loaded successfully! ---")
    print(f"AUTH_TOKEN (first 5 chars): {settings.AUTH_TOKEN[:5]}*****")
    print(f"PINECONE_API_KEY (first 5 chars): {settings.PINECONE_API_KEY[:5]}*****")
    print(f"PINECONE_ENVIRONMENT: {settings.PINECONE_ENVIRONMENT}")
    print("------------------------------------\n")
except Exception as e:
    print(f"\n--- Error loading settings: {e} ---")
    print("This means the required fields were not found in the environment.")
    print("Check your .env file's presence in the current working directory, and its contents.")
    print("------------------------------------\n")
    raise # Re-raise the exception to still see the full traceback in Uvicorn