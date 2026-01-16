#!/usr/bin/env python3
import streamlit as st
import torch
from bashbot import load_model, generate_command

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BashBot - Bash Command Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 BashBot - AI Bash Command Generator")
st.markdown("Generate bash commands using AI. Describe what you want to do and let BashBot generate the command!")

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Maximum number of tokens to generate"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **BashBot** is an AI-powered bash command generator that helps you quickly generate shell commands based on natural language descriptions.
    
    Simply describe what you want to do, and BashBot will generate the appropriate bash command for you.
    """)
    
    st.markdown("---")
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Running on: {device_info}")

# =========================================================
# MAIN CHAT INTERFACE
# =========================================================

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Load model on first run
if not st.session_state.model_loaded:
    try:
        with st.spinner("Loading model... This may take a moment on first run."):
            load_model()
        st.session_state.model_loaded = True
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure the model path is correct and the model files are available.")
        st.stop()

# Display chat history
st.subheader("💬 Chat History")
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input section
st.markdown("---")
col1, col2 = st.columns([1, 0.15])

with col1:
    user_input = st.text_input(
        "Enter your command request:",
        placeholder="e.g., 'List all files modified in the last 24 hours'",
        label_visibility="collapsed"
    )

with col2:
    submit_button = st.button("🚀 Generate", use_container_width=True)

# Process user input
if submit_button and user_input:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Generating command..."):
            try:
                response = generate_command(user_input, max_tokens=max_tokens)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Display the command in a code block
                st.code(response, language="bash")
                
                # Add a copy button
                st.button("📋 Copy Command", key=f"copy_{len(st.session_state.messages)}")
                
            except Exception as e:
                st.error(f"Error generating command: {str(e)}")

# Clear history button
if st.session_state.messages:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
