import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # Model Configuration
    MODEL_NAME: str = "gemini-2.0-flash-exp"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 8192
    
    # UI Configuration  
    APP_TITLE: str = "🤖 AI Q&A Bot - Powered by Gemini & LangChain"
    PAGE_ICON: str = "🚀"
    LAYOUT: str = "wide"
    
    # Memory Configuration
    MAX_MEMORY_LENGTH: int = 10
    MEMORY_RETURN_MESSAGES: bool = True
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get Google API key from environment or Streamlit secrets"""
        import streamlit as st
        
        # Try environment variable first
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Fall back to Streamlit secrets
        if not api_key:
            try:
                api_key = st.secrets.get("GOOGLE_API_KEY")
            except:
                pass
                
        return api_key

# Export configuration instance
config = AppConfig()