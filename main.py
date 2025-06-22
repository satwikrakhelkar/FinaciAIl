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
        # Format prompt for instruct models
        if "Instruct" in model_name or "Chat" in model_name or "Qwen" in model_name:
            # Format for instruction/chat models
            if "Qwen" in model_name:
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"Human: {prompt}\nAssistant:"
        else:
            formatted_prompt = prompt
        
        result = self.query_model(model_name, formatted_prompt, parameters)
        
        if "error" in result:
            # Handle model loading errors
            if "loading" in str(result["error"]).lower():
                return "⏳ Model is loading, please wait a moment and try again..."
            return f"❌ Error: {result['error']}"
        
        try:
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    response = result[0]["generated_text"].strip()
                    # Remove the prompt from response for instruct models
                    if formatted_prompt in response:
                        response = response.replace(formatted_prompt, "").strip()
                    return response
                elif "text" in result[0]:
                    return result[0]["text"].strip()
            elif isinstance(result, dict):
                if "generated_text" in result:
                    response = result["generated_text"].strip()
                    # Remove the prompt from response for instruct models
                    if formatted_prompt in response:
                        response = response.replace(formatted_prompt, "").strip()
                    return response
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
        
        # Get API token from secrets
        try:
            api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
            st.success("✅ API Token loaded from secrets")
        except KeyError:
            st.error("❌ HUGGINGFACE_API_TOKEN not found in secrets")
            api_token = None
        except Exception as e:
            st.error(f"❌ Error loading secrets: {str(e)}")
            api_token = None
        
        # Model selection with categories
        st.subheader("🤖 Model Selection")
        
        model_categories = {
            "Qwen Models (Multilingual)": [
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2-0.5B-Instruct",
                "Qwen/Qwen2-1.5B-Instruct",
                "Qwen/Qwen2-7B-Instruct",
                "Qwen/Qwen1.5-0.5B-Chat",
                "Qwen/Qwen1.5-1.8B-Chat",
                "Qwen/Qwen1.5-4B-Chat",
                "Qwen/Qwen1.5-7B-Chat"
            ],
            "Conversational Models": [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill",
                "facebook/blenderbot-1B-distill"
            ],
            "Instruction Following": [
                "google/flan-t5-base",
                "google/flan-t5-large",
                "google/flan-t5-small"
            ],
            "General Purpose": [
                "gpt2",
                "gpt2-medium",
                "distilgpt2"
            ],
            "Code & Technical": [
                "huggingface/CodeBERTa-small-v1",
                "microsoft/CodeGPT-small-py"
            ]
        }
        
        # Flatten all models for the selectbox
        all_models = []
        for category, models in model_categories.items():
            all_models.extend(models)
        
        selected_model = st.selectbox(
            "Select Model",
            all_models,
            index=0,
            help="Choose the AI model for conversation. Qwen models excel at multilingual conversations."
        )
        
        # Show model category info
        model_category = None
        for category, models in model_categories.items():
            if selected_model in models:
                model_category = category
                break
        
        if model_category:
            st.info(f"📂 Category: {model_category}")
            
        # Model-specific information
        if "Qwen" in selected_model:
            st.markdown("""
            <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>🚀 Qwen Model Features:</strong><br>
                • Excellent multilingual support (Chinese, English, etc.)<br>
                • Strong instruction following capabilities<br>
                • Good at reasoning and conversation<br>
                • Optimized for chat and instruct tasks
            </div>
            """, unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Select Model",
            all_models,
            index=0,
            help="Choose the AI model for conversation. Qwen models excel at multilingual conversations."
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
                st.error("Please configure HUGGINGFACE_API_TOKEN in secrets")
        
        # Model info
        st.markdown("""
        <div class="sidebar-info">
            <h4>💡 Tips:</h4>
            <ul>
                <li>API token is securely loaded from secrets</li>
                <li>Qwen models are great for multilingual chats</li>
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
        if api_token:
            st.info("🔧 API token loaded successfully. Click 'Configure API' to start chatting.")
        else:
            st.warning("⚠️ Please configure your HuggingFace API token in Streamlit secrets to start chatting.")
        
        # Instructions
        st.markdown("""
        ### Configuration Required:
        1. **Add to Secrets**: Configure `HUGGINGFACE_API_TOKEN` in your Streamlit secrets
        2. **Get API Token**: Visit [HuggingFace Tokens](https://huggingface.co/settings/tokens) and create a new token
        3. **Select Model**: Choose your preferred AI model from the sidebar
        4. **Start Chatting**: Click "Configure API" and begin your conversation!
        
        ### Popular Models:
        - **Qwen Models**: Excellent multilingual models with strong reasoning
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
