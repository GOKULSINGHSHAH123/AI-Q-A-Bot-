
import streamlit as st
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate

# Configure Streamlit page FIRST - must be the first st command
st.set_page_config(
    page_title="🤖 AI Q&A Bot - Powered by Gemini & LangChain",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

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
    MEMORY_RETURN_MESSAGES: bool = False
    
    @classmethod
    def get_api_key(cls) -> Optional[str]:
        """Get Google API key from environment or Streamlit secrets"""
        # Try environment variable first
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Fall back to Streamlit secrets
        if not api_key:
            try:
                api_key = st.secrets.get("GOOGLE_API_KEY")
            except:
                pass
                
        return api_key

# Create global config instance
config = AppConfig()

# ============================================================================
# CHATBOT BACKEND CLASSES
# ============================================================================

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
            # Initialize Gemini LLM
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
                memory_key="history",  # Use 'history' to match prompt template
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
                prompt=prompt,
                verbose=False  # Set to True for debugging
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_chatbot() -> Optional[GeminiChatBot]:
    """Initialize chatbot with proper error handling"""
    api_key = config.get_api_key()
    
    if not api_key:
        return None
    
    try:
        # Initialize chatbot
        chatbot = GeminiChatBot(api_key)
        
        # Add system context
        chatbot.add_system_context(
    "You are a helpful AI assistant. Be natural, concise, and match the user's tone. For greetings, respond briefly."
)

        
        return chatbot
        
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        return None

def render_custom_css():
    """Add custom CSS for better UI"""
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 10px;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
    
    .metrics-container {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    
    .element-container {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar() -> Optional[str]:
    """Render sidebar with API key input and controls"""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "🔑 Google API Key",
            type="password",
            placeholder="Enter your Gemini API key...",
            help="Get your free API key from Google AI Studio: https://aistudio.google.com",
            key="api_key_input"
        )
        
        # Set API key in environment
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input
        
        st.markdown("---")
        
        # Model configuration
        st.markdown("### 🎛️ Model Settings")
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=config.TEMPERATURE,
            step=0.1,
            help="Higher values make responses more creative",
            key="temperature_slider"
        )
        
        max_tokens = st.selectbox(
            "Max Response Length",
            options=[1000, 2000, 4000, 8000],
            index=3,
            help="Maximum tokens in response",
            key="max_tokens_select"
        )
        
        # Update config
        config.TEMPERATURE = temperature
        config.MAX_TOKENS = max_tokens
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        if st.button("🧹 Clear Chat History", type="secondary", key="clear_history"):
            if 'chatbot' in st.session_state and st.session_state.chatbot:
                st.session_state.chatbot.clear_memory()
                st.session_state.messages = []
                st.rerun()
        
        if st.button("🔄 Reset Bot", type="secondary", key="reset_bot"):
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot
            st.session_state.messages = []
            st.rerun()
        
        # Information panel
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
        <h4>💡 Tips for Best Results</h4>
        <ul>
        <li>Ask specific, clear questions</li>
        <li>Use conversation context</li>
        <li>Try different temperature settings</li>
        <li>Experiment with follow-up questions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        return api_key_input

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Apply custom CSS
    render_custom_css()
    
    # Custom header
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI Q&A Bot</h1>
        <p>Powered by Google Gemini 2.0 Flash & LangChain</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get API key from sidebar
    api_key = render_sidebar()
    
    # Initialize chatbot if not exists
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    # Check if chatbot is ready
    if not st.session_state.chatbot:
        st.warning("⚠️ Please configure your Google API key in the sidebar to start chatting!")
        
        # Display getting started info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Get API Key**: Visit [Google AI Studio](https://aistudio.google.com)
            2. **Create Account**: Sign in with your Google account
            3. **Generate Key**: Click "Get API key" → "Create API key in new project"
            4. **Copy Key**: Paste it in the sidebar input field
            5. **Start Chatting**: Ask any question!
            
            ### ✨ Features
            - **Memory**: Remembers conversation context
            - **Streaming**: Real-time response generation  
            - **Advanced AI**: Latest Gemini 2.0 Flash model
            - **Professional UI**: Clean, responsive interface
            """)
        
        with col2:
            st.markdown("""
            ### 💭 Example Questions
            
            **Technical:**
            - "Explain quantum computing in simple terms"
            - "Write a Python function to sort a list"
            - "What's the difference between AI and ML?"
            
            **Creative:**
            - "Write a short story about a robot"
            - "Create a marketing slogan for a tech startup"
            - "Explain blockchain like I'm 5 years old"
            
            **Problem Solving:**
            - "Help me debug this code snippet"
            - "How do I optimize this algorithm?"
            - "Best practices for API design"
            """)
        
        return
    
    # Display chat metrics
    if st.session_state.messages:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💬 Messages", len(st.session_state.messages))
        with col2:
            st.metric("🤖 Model", config.MODEL_NAME.split('-')[0].title())
        with col3:
            st.metric("🌡️ Temperature", f"{config.TEMPERATURE}")
        with col4:
            st.metric("📝 Max Tokens", config.MAX_TOKENS)
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything... 💭", key="chat_input"):
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            
            # Create placeholder for streaming
            message_placeholder = st.empty()
            
            # Show thinking animation
            with st.spinner("🧠 Thinking..."):
                time.sleep(0.5)  # Brief pause for UX
            
            try:
                # Get streaming response
                response = st.session_state.chatbot.get_response(
                    prompt, 
                    stream_container=message_placeholder
                )
                
                # Update placeholder with final response
                message_placeholder.markdown(response)
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_response = f"❌ **Error**: {str(e)}\n\nPlease check your API key and try again."
                message_placeholder.markdown(error_response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_response
                })

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with ❤️ using <strong>Streamlit</strong>, <strong>LangChain</strong> & <strong>Google Gemini</strong><br>
        <small>Intern Assignment • AI Q&A Bot • {datetime.now().year}</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
