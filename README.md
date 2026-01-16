# 🤖 BashBot - AI Bash Command Generator

BashBot is an intelligent bash command generator powered by a fine-tuned language model. It uses natural language to understand your requirements and generates appropriate bash commands. The project includes both a CLI interface and an interactive web-based chat interface built with Streamlit.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Model File Verification & Fixing](#-model-file-verification--fixing)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Security Notes](#-security-notes)

## ✨ Features

- **CLI Interface**: Use bashbot directly from the command line
- **Web Chat Interface**: Interactive Streamlit-based chat window
- **Model Quantization**: 8-bit quantization support for efficient model loading
- **GPU/CPU Support**: Automatically detects and uses available hardware
- **Chat History**: Maintains conversation history in the web interface
- **Command Export**: Easy copy-to-clipboard functionality for generated commands

## 📁 Project Structure

```
Bash-Bot/
├── bashbot.py              # Core module with model loading and command generation
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── model/                 # Model storage directory (download required)
│   └── bashbot_merged/    # Merged model weights and configs
├── offload/               # Memory offload directory for quantization
└── README.md             # This file
```

## 🔧 Prerequisites

- **Python 3.8+** (tested with Python 3.12)
- **CUDA 12.x** (optional, for GPU acceleration)
- **At least 16GB RAM** (for model loading)
- **~15GB Disk Space** (for the model files)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Bash-Bot
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🤖 Model Setup

The BashBot model needs to be downloaded separately due to its size (~7GB).

### Download the Model

1. **Download Link**: [Model on Google Drive](https://drive.google.com/drive/folders/1GigEeew3K6wYWeflZp7rjyECICJmldrx?usp=drive_link)
   
2. **Alternative**: Ask your model provider for the `bashbot_merged` model files

### Setup Instructions

1. Download the model folder
2. Extract it into the `model/` directory:
   ```bash
   model/
   └── bashbot_merged/
       ├── config.json
       ├── tokenizer.json
       ├── tokenizer.model
       ├── model.safetensors.index.json
       ├── model-00001-of-00002.safetensors
       └── model-00002-of-00002.safetensors
   ```

3. Verify the model structure:
   ```bash
   ls -la model/bashbot_merged/
   ```

4. The model files should total approximately **7.1GB**

## 🚀 Running the Application

### Option 1: CLI Interface

Generate a single bash command from the command line:

```bash
python bashbot.py "list all files modified in the last 24 hours"
```

**Output:**
```
Prompt: list all files modified in the last 24 hours

Bashbot Output:

find . -type f -mtime -1
```

### Option 2: Web Interface (Recommended)

Start the interactive Streamlit chat interface:

```bash
streamlit run app.py
```

The app will be available at:
- **Local**: http://localhost:8501
- **Network**: http://172.22.167.193:8501 (adjust IP based on your network)

## 💬 Usage

### Web Interface

1. **Open the App**: Navigate to http://localhost:8501
2. **Enter Command Description**: Type what you want to do in natural language
   - Example: "Find all python files larger than 1MB"
3. **Click Generate**: Press the "🚀 Generate" button
4. **Copy Command**: Use the "📋 Copy Command" button to copy the generated command
5. **View History**: All commands are stored in the chat history
6. **Clear History**: Use the "🗑️ Clear Chat History" button to reset

### CLI Interface

```bash
# Single command generation
python bashbot.py "remove all empty directories recursively"

# For multi-word prompts, use quotes
python bashbot.py "search for files containing specific text in all subdirectories"
```

## 📄 File Descriptions

### `bashbot.py`
Core module that handles:
- Model and tokenizer loading
- Lazy loading with global caching
- Command generation with configurable parameters
- Configuration paths for model and offload directories
- CLI entry point

**Key Functions:**
- `load_model()`: Initialize and cache the model and tokenizer
- `get_tokenizer()`: Get cached tokenizer
- `get_model()`: Get cached model
- `generate_command(prompt, max_tokens)`: Generate bash commands

### `app.py`
Streamlit web interface featuring:
- Interactive chat window
- Chat history management
- Settings panel for token control
- Device detection (GPU/CPU)
- Model loading indicator
- Copy-to-clipboard functionality
- Clear history option

**Settings:**
- Max Tokens: Control command length (10-200 tokens)
- Displays current compute device (CUDA/CPU)

### `requirements.txt`
Python package dependencies:
- `torch`: Deep learning framework
- `transformers`: Hugging Face model library
- `bitsandbytes`: Quantization support
- `peft`: Parameter-efficient fine-tuning
- `accelerate`: Distributed computing support
- `datasets`: Dataset handling
- `streamlit`: Web UI framework
- `tqdm`: Progress bars

### `setup.sh`
Initial setup script for environment configuration

## 🔧 Model File Verification & Fixing

Before running the application, verify your model files have the correct names. Downloaded models sometimes have suffix variations (e.g., `-001`, `-002`) that cause errors.

### Verify Model Files

Check if your model files are correctly named:

```bash
# List all safetensors files in the model directory
ls -lh model/bashbot_merged/*.safetensors
```

**Expected output:**
```
-rw-r--r-- 1 user user 4.7G Jan 16 08:01 model-00001-of-00002.safetensors
-rw-r--r-- 1 user user 2.4G Jan 16 07:59 model-00002-of-00002.safetensors
```

### Fix Model Name Errors

If your files have incorrect names (e.g., `model-00001-of-00002-002.safetensors`), rename them:

```bash
# Navigate to the model directory
cd model/bashbot_merged/

# Check current file names
ls -lh model-*.safetensors

# If files have suffixes like -001 or -002, rename them:
# For the first file (larger, ~4.7GB)
mv model-00001-of-00002-002.safetensors model-00001-of-00002.safetensors

# For the second file (smaller, ~2.4GB)
mv model-00002-of-00002-001.safetensors model-00002-of-00002.safetensors

# Verify the rename
ls -lh model-*.safetensors
```

### Complete Validation Checklist

Run this script to verify everything is set up correctly:

```bash
#!/bin/bash
echo "🔍 BashBot Model Setup Verification"
echo "===================================="

# Check model directory exists
if [ -d "model/bashbot_merged" ]; then
    echo "✅ Model directory found"
else
    echo "❌ Model directory not found at model/bashbot_merged"
    exit 1
fi

# Check required files
echo ""
echo "Checking required model files..."
required_files=(
    "model/bashbot_merged/config.json"
    "model/bashbot_merged/tokenizer.json"
    "model/bashbot_merged/tokenizer.model"
    "model/bashbot_merged/model.safetensors.index.json"
    "model/bashbot_merged/model-00001-of-00002.safetensors"
    "model/bashbot_merged/model-00002-of-00002.safetensors"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING"
    fi
done

# Check total model size
echo ""
echo "Model total size:"
du -sh model/bashbot_merged/

echo ""
echo "✅ Setup verification complete!"
```

Save this as `verify_model.sh` and run:
```bash
bash verify_model.sh
```

## ⚙️ Configuration

### Model Configuration
Located in `bashbot.py`:
```python
MERGED_MODEL = "/home/sandeep-naruka/Bash-Bot/model/bashbot_merged"
OFFLOAD_DIR = "/home/sandeep-naruka/Bash-Bot/offload"
```

### Quantization Settings
The model uses 8-bit quantization with the following configuration:
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=False,
    llm_int8_enable_fp32_cpu_offload=True,
)
```

## 🐛 Troubleshooting

### Issue: Model file not found / No such file or directory
**Error Message**: `No such file or directory: /home/.../model-00001-of-00002.safetensors`

**Solutions**:
1. **Verify file names** - Check if your files have incorrect suffixes:
   ```bash
   ls -lh model/bashbot_merged/model-*.safetensors
   ```

2. **Rename files if needed** - Follow the [Model File Verification & Fixing](#-model-file-verification--fixing) section above

3. **Check file structure** - Ensure the path is correct:
   ```bash
   ls -la model/bashbot_merged/
   ```
   Should show all 6 required files (config.json, tokenizer files, model index, and 2 safetensors files)

### Issue: Repo id must be in the form 'repo_name'
**Error Message**: `Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/path/to/model'`

**Solution**: This means the model path is treated as a HuggingFace repo instead of a local path. Ensure:
1. Model directory exists at the configured path
2. All required files are present in the directory
3. Run the verification script from the section above

### Issue: Model file not found - specific file errors
**Error Pattern**: References to `-001` or `-002` suffixed files

**Solution**: Your downloaded model has incorrectly named files. Rename them:
```bash
cd model/bashbot_merged

# Check current names
ls model-*.safetensors

# Rename files (adjust names based on what you see)
# The LARGER file (~4.7GB) should be model-00001-of-00002.safetensors
# The SMALLER file (~2.4GB) should be model-00002-of-00002.safetensors

mv model-00001-of-00002-*.safetensors model-00001-of-00002.safetensors
mv model-00002-of-00002-*.safetensors model-00002-of-00002.safetensors

# Verify
ls -lh model-*.safetensors
```

### Issue: Out of memory error
**Solution**: Reduce `max_tokens` value or close other applications

### Issue: CUDA not found (if you have a GPU)
**Solution**: Install CUDA-compatible PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Streamlit port already in use
**Solution**: Kill the process or use a different port:
```bash
pkill -f "streamlit run"
# Or specify a different port:
streamlit run app.py --server.port 8502
```

### Issue: ModuleNotFoundError when importing bashbot
**Solution**: Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

### Quick Diagnostic Commands

Run these to diagnose issues:

```bash
# 1. Check Python version
python --version

# 2. Check if virtual environment is active
which python

# 3. Verify all dependencies are installed
pip list | grep -E "torch|transformers|streamlit"

# 4. Check model directory structure
find model/bashbot_merged -type f | head -20

# 5. Check available disk space
df -h | grep -E "Filesystem|home"

# 6. Check available RAM
free -h

# 7. Verify model file sizes (should match these approximately)
ls -lh model/bashbot_merged/model-*.safetensors
```

## 📊 Performance Tips

1. **GPU Usage**: Make sure CUDA is properly installed for faster inference
2. **Token Limit**: Shorter max_tokens = faster generation. Default 50 is optimal
3. **Memory**: Model uses ~7GB loaded in memory
4. **First Load**: Initial load takes ~30-60 seconds as the model is loaded into memory

## 🔐 Security Notes

- Model path is hardcoded for local development
- No API keys or credentials required for local usage
- Streamlit runs on localhost by default
- Always validate generated bash commands before execution

## 📝 License

[Add your license information here]

## 👨‍💻 Author

Sandeep Naruka

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the file descriptions
3. Verify model files are correctly placed

## 🔄 Version History

- **v1.0** (Jan 16, 2026): Initial release with CLI and web interface

---

**Last Updated**: January 16, 2026
