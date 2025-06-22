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
    
    .model-category {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        font-weight: bold;
        color: #2e7d32;
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
        
        # Default parameters - adjusted for Qwen models
        default_params = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
        
        # Special handling for Qwen models
        if "qwen" in model_name.lower():
            default_params.update({
                "max_new_tokens": 200,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.1
            })
        
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

def get_model_options():
    """Return categorized model options"""
    return {
        "🚀 Qwen Models (Alibaba)": [
            "Qwen/Qwen2-0.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct", 
            "Qwen/Qwen1.5-0.5B-Chat",
            "Qwen/Qwen1.5-1.8B-Chat",
            "Qwen/Qwen1.5-4B-Chat",
            "Qwen/Qwen1.5-7B-Chat",
            "Qwen/Qwen-1_8B-Chat",
            "Qwen/Qwen-7B-Chat"
        ],
        "💬 Conversational Models": [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill"
        ],
        "🧠 Instruction Models": [
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl"
        ],
        "📝 Text Generation": [
            "gpt2",
            "gpt2-medium",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B"
        ],
        "🔧 Code Models": [
            "microsoft/CodeGPT-small-py",
            "Salesforce/codegen-350M-mono"
        ]
    }

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
        
        # Try to get API token from secrets first, then allow manual input as fallback
        try:
            api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
            st.success("✅ API Token loaded from secrets")
            st.info("🔒 Using secure token from deployment settings")
        except KeyError:
            st.warning("⚠️ No API token found in secrets")
            api_token = st.text_input(
                "HuggingFace API Token (Fallback)",
                type="password",
                help="Enter your HuggingFace API token. Get one at https://huggingface.co/settings/tokens"
            )
        
        # Model selection with categories
        st.subheader("🤖 Select AI Model")
        model_options = get_model_options()
        
        # Create flattened list for selectbox
        all_models = []
        for category, models in model_options.items():
            all_models.extend(models)
        
        selected_model = st.selectbox(
            "Choose Model",
            all_models,
            index=0,
            help="Choose the HuggingFace model for conversation"
        )
        
        # Show model category
        for category, models in model_options.items():
            if selected_model in models:
                st.markdown(f'<div class="model-category">{category}</div>', unsafe_allow_html=True)
                break
        
        # Model parameters
        st.subheader("🔧 Model Parameters")
        max_tokens = st.slider("Max New Tokens", 50, 500, 200 if "qwen" in selected_model.lower() else 150)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8 if "qwen" in selected_model.lower() else 0.7, 0.1)
        top_p = st.slider("Top P", 0.1, 1.0, 0.95 if "qwen" in selected_model.lower() else 0.9, 0.1)
        
        # Additional parameters for Qwen models
        if "qwen" in selected_model.lower():
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.5, 1.1, 0.05)
        
        # Configure API button
        if st.button("🔧 Configure API", type="primary"):
            if api_token:
                st.session_state.chatbot = HuggingFaceChatbot(api_token)
                st.session_state.api_configured = True
                st.success("API configured successfully!")
            else:
                st.error("Please enter your API token")
        
        # API Token status and model info
        if api_token:
            st.markdown("""
            <div class="sidebar-info">
                <h4>🔐 Security Status:</h4>
                <p>✅ API Token is configured securely</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-info">
            <h4>💡 Model Tips:</h4>
            <ul>
                <li><strong>Qwen Models:</strong> Latest from Alibaba, great for multilingual tasks</li>
                <li><strong>DialoGPT:</strong> Optimized for conversations</li>
                <li><strong>FLAN-T5:</strong> Excellent instruction following</li>
                <li><strong>BlenderBot:</strong> Open-domain conversations</li>
                <li><strong>GPT-2:</strong> Classic text generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.api_configured:
        if not api_token:
            st.error("🔑 **API Token Required**")
            st.markdown("""
            ### For Development (Local):
            1. Create a `.streamlit/secrets.toml` file in your project root
            2. Add your token: `HUGGINGFACE_API_TOKEN = "your_token_here"`
            3. Restart the application
            
            ### For Production (Streamlit Cloud):
            1. Go to your app settings in Streamlit Cloud
            2. Navigate to "Secrets" section  
            3. Add the secret as shown in the deployment guide below
            
            ### Manual Entry (Fallback):
            You can also enter your token manually in the sidebar as a fallback.
            """)
        else:
            st.warning("⚠️ Click 'Configure API' to start using the chatbot.")
        
        # Instructions
        st.markdown("""
        ### How to get started:
        1. **Get API Token**: Visit [HuggingFace Tokens](https://huggingface.co/settings/tokens) and create a new token
        2. **Secure Setup**: Use secrets for production deployment (see guide below)
        3. **Select Model**: Choose your preferred AI model (try the new Qwen models!)
        4. **Start Chatting**: Click "Configure API" and begin your conversation!
        
        ### ✨ New Qwen Models Added:
        - **Qwen2-0.5B/1.5B-Instruct**: Latest lightweight instruction models
        - **Qwen1.5-Chat Series**: Various sizes from 0.5B to 7B parameters
        - **Qwen-7B-Chat**: Large conversational model
        
        ### Popular Model Categories:
        - **🚀 Qwen Models**: State-of-the-art from Alibaba, multilingual support
        - **💬 Conversational**: DialoGPT, BlenderBot for natural conversations  
        - **🧠 Instruction**: FLAN-T5 models for following instructions
        - **📝 Text Generation**: GPT-2 variants for creative writing
        - **🔧 Code Models**: Specialized for programming tasks
        """)
    else:
        # Display current model info
        st.info(f"💬 Currently using: **{selected_model}**")
        
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
                
                # Add repetition penalty for Qwen models
                if "qwen" in selected_model.lower():
                    parameters["repetition_penalty"] = repetition_penalty
                
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
        <p><small>Now featuring Qwen models from Alibaba! Keep your API token secure.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
