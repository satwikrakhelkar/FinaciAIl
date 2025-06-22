import streamlit as st
import requests
import json
from typing import Dict, List, Optional
import time

# Page configuration
st.set_page_config(
    page_title="HuggingFace AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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
    
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .sidebar-info {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HuggingFaceChatbot:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api-inference.huggingface.co/models"
    
    def query_model(self, model_name: str, prompt: str, parameters: Dict = None) -> Dict:
        """Query a Hugging Face model via API"""
        url = f"{self.base_url}/{model_name}"
        
        # Default parameters
        default_params = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
        
        if parameters:
            default_params.update(parameters)
        
        payload = {
            "inputs": prompt,
            "parameters": default_params
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_response(self, model_name: str, prompt: str, parameters: Dict = None) -> str:
        """Get formatted response from the model"""
        result = self.query_model(model_name, prompt, parameters)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        try:
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                elif "text" in result[0]:
                    return result[0]["text"].strip()
            elif isinstance(result, dict):
                if "generated_text" in result:
                    return result["generated_text"].strip()
                elif "text" in result:
                    return result["text"].strip()
            
            return str(result)
        except (KeyError, IndexError, TypeError):
            return "Sorry, I couldn't process the response properly."

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = False

def display_chat_messages():
    """Display chat messages"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>AI:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 HuggingFace AI Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Token input
        api_token = st.text_input(
            "HuggingFace API Token",
            type="password",
            help="Enter your HuggingFace API token. Get one at https://huggingface.co/settings/tokens"
        )
        
        # Model selection
        model_options = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "huggingface/CodeBERTa-small-v1",
            "gpt2",
            "gpt2-medium",
            "distilgpt2"
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose the HuggingFace model for conversation"
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        max_tokens = st.slider("Max New Tokens", 50, 500, 150)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1)
        
        # Configure API button
        if st.button("🔧 Configure API", type="primary"):
            if api_token:
                st.session_state.chatbot = HuggingFaceChatbot(api_token)
                st.session_state.api_configured = True
                st.success("API configured successfully!")
            else:
                st.error("Please enter your API token")
        
        # Model info
        st.markdown("""
        <div class="sidebar-info">
            <h4>💡 Tips:</h4>
            <ul>
                <li>Get your free API token from HuggingFace</li>
                <li>Different models have different strengths</li>
                <li>Adjust temperature for creativity</li>
                <li>Lower temperature = more focused responses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.api_configured:
        st.warning("⚠️ Please configure your HuggingFace API token in the sidebar to start chatting.")
        
        # Instructions
        st.markdown("""
        ### How to get started:
        1. **Get API Token**: Visit [HuggingFace Tokens](https://huggingface.co/settings/tokens) and create a new token
        2. **Enter Token**: Paste your token in the sidebar
        3. **Select Model**: Choose your preferred AI model
        4. **Start Chatting**: Click "Configure API" and begin your conversation!
        
        ### Popular Models:
        - **DialoGPT**: Great for conversational AI
        - **BlenderBot**: Excellent for open-domain conversations
        - **FLAN-T5**: Good for instruction-following tasks
        - **GPT-2**: Classic language model for text generation
        """)
    else:
        # Display chat messages
        display_chat_messages()
        
        # Chat input
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Your message:",
                    placeholder="Type your message here...",
                    key="user_input"
                )
            
            with col2:
                send_button = st.button("Send 🚀", type="primary")
        
        # Process user input
        if send_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # Show loading spinner
            with st.spinner("AI is thinking... 🤔"):
                # Prepare parameters
                parameters = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "return_full_text": False
                }
                
                # Get AI response
                ai_response = st.session_state.chatbot.get_response(
                    selected_model, 
                    user_input, 
                    parameters
                )
            
            # Add AI response to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response
            })
            
            # Rerun to update the display
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Built with ❤️ using Streamlit and HuggingFace API</p>
        <p><small>Make sure to keep your API token secure and never share it publicly!</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()