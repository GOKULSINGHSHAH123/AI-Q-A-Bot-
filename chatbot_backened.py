import os
from typing import Dict, Any, List, Optional
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory  # Using simpler memory
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
import asyncio
from config import config

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for Streamlit streaming"""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM"""
        self.text += token
        self.container.markdown(self.text + "▌")

class GeminiChatBot:
    """Advanced Gemini-powered chatbot with LangChain integration"""
    
    def __init__(self, api_key: str):
        """Initialize the chatbot with API key"""
        self.api_key = api_key
        self.llm = None
        self.memory = None
        self.conversation_chain = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, memory, and conversation chain"""
        try:
            # Initialize Gemini LLM with streaming
            self.llm = ChatGoogleGenerativeAI(
                model=config.MODEL_NAME,
                google_api_key=self.api_key,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                streaming=True,
                convert_system_message_to_human=True
            )
            
            # Initialize conversation memory with correct memory_key
            self.memory = ConversationBufferMemory(
                memory_key="history",  # FIXED: Use 'history' not 'chat_history'
                return_messages=False,  # Return as string, not message objects
                input_key="input"
            )
            
            # Create custom prompt template that matches memory structure
            template = """The following is a friendly conversation between a human and an AI assistant. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history}
Human: {input}
AI:"""

            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=template
            )
            
            # Create conversation chain with custom prompt
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=prompt,  # Use custom prompt
                verbose=True
            )
            
            print("✅ Chatbot components initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {str(e)}")
            raise e
    
    def get_response(self, user_input: str, stream_container=None) -> str:
        """Get response from the chatbot with optional streaming"""
        try:
            if stream_container:
                # Streaming response
                callback_handler = StreamlitCallbackHandler(stream_container)
                response = self.conversation_chain.invoke(
                    {"input": user_input},
                    {"callbacks": [callback_handler]}
                )
                return response.get('response', '')
            else:
                # Non-streaming response
                response = self.conversation_chain.invoke({"input": user_input})
                return response.get('response', '')
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            st.success("🧹 Conversation history cleared!")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        if not self.memory:
            return []
        
        messages = []
        try:
            # Get memory variables
            memory_vars = self.memory.load_memory_variables({})
            history = memory_vars.get('history', '')
            
            # Parse the history string (simple parsing)
            if history:
                # Split by Human: and AI: markers
                parts = history.split('\n')
                current_role = None
                current_content = ""
                
                for part in parts:
                    part = part.strip()
                    if part.startswith('Human:'):
                        if current_role and current_content:
                            messages.append({"role": current_role, "content": current_content.strip()})
                        current_role = "user"
                        current_content = part[6:]  # Remove 'Human:'
                    elif part.startswith('AI:'):
                        if current_role and current_content:
                            messages.append({"role": current_role, "content": current_content.strip()})
                        current_role = "assistant"
                        current_content = part[3:]  # Remove 'AI:'
                    else:
                        if current_content:
                            current_content += " " + part
                        else:
                            current_content = part
                
                # Add the last message
                if current_role and current_content:
                    messages.append({"role": current_role, "content": current_content.strip()})
                    
        except Exception as e:
            st.warning(f"Could not load conversation history: {str(e)}")
            
        return messages
    
    def add_system_context(self, context: str):
        """Add system context to improve responses"""
        # Add initial context to memory
        system_input = f"Please remember this context for our conversation: {context}"
        system_response = "I understand and will keep this context in mind for our conversation."
        
        # Add to memory manually
        self.memory.save_context(
            {"input": system_input},
            {"response": system_response}
        )

def initialize_chatbot() -> Optional[GeminiChatBot]:
    """Initialize chatbot with proper error handling"""
    api_key = config.get_api_key()
    
    if not api_key:
        st.error("🔑 **Google API Key Required!** Please set your GOOGLE_API_KEY in the sidebar.")
        st.info("👈 Enter your API key in the sidebar to get started")
        return None
    
    try:
        # Initialize chatbot
        chatbot = GeminiChatBot(api_key)
        
        # Add system context
        chatbot.add_system_context(
            "You are an intelligent AI assistant powered by Google Gemini and LangChain. Provide helpful, detailed responses and maintain conversation context."
        )
        
        return chatbot
        
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        st.info("Please check your API key and try again.")
        return None

# Export key functions
__all__ = ['GeminiChatBot', 'initialize_chatbot']