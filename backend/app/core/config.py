from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "AI Research Assistant API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    DEFAULT_MODEL: str = "gpt-4.1"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Search APIs
    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    GOOGLE_CSE_ID: str = Field(default="", env="GOOGLE_CSE_ID")
    TAVILY_API_KEY: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    
    # Arxiv
    ARXIV_MAX_RESULTS: int = 5
    
    # API Security
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000"
    ]
    
    # HTTP Client
    USER_AGENT: Optional[str] = Field(
        default="knowledge-synth/1.0",
        env="USER_AGENT"
    )
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'  # This allows extra fields in .env without validation errors
    )

settings = Settings()
