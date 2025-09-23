# LLM Configuration Guide

## Table of Contents
1. [Overview](#overview)
2. [Supported Providers](#supported-providers)
3. [Local Models (Ollama)](#local-models-ollama)
4. [Cloud Providers](#cloud-providers)
5. [Mac Metal Support (MLX)](#mac-metal-support-mlx)
6. [Configuration](#configuration)
7. [Testing Your Setup](#testing-your-setup)
8. [Performance Optimization](#performance-optimization)
9. [Cost Management](#cost-management)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Multi-Agent City Platform supports multiple LLM providers for maximum flexibility, performance, and cost optimization. Agents can use different models based on their requirements, from tiny local models for simple decisions to powerful cloud models for complex reasoning.

### Key Features
- **Auto-detection**: Automatically detects available LLMs and system resources
- **Fallback chain**: Gracefully falls back to available models if primary fails
- **Resource management**: Manages GPU/CPU allocation for local models
- **Multi-provider**: Mix and match providers for different agents
- **Cost tracking**: Monitor API usage and costs

---

## Supported Providers

### Provider Comparison

| Provider | Models | Speed | Cost | Privacy | Offline | GPU Required |
|----------|---------|-------|------|---------|---------|--------------|
| **Ollama** | Llama, Mistral, Phi | Fast | Free | High | Yes | Optional |
| **OpenAI** | GPT-4, GPT-3.5 | Fast | $$$ | Low | No | No |
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | Fast | $$$ | Low | No | No |
| **Google Gemini** | Gemini Pro, Ultra | Fast | $$ | Low | No | No |
| **xAI Grok** | Grok-1 | Fast | $$$ | Low | No | No |
| **OpenRouter** | 100+ models | Varies | $ | Low | No | No |
| **MLX (Mac)** | Llama, Mistral | Very Fast | Free | High | Yes | Metal GPU |

---

## Local Models (Ollama)

### Installation

#### macOS
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or with Homebrew
brew install ollama
```

#### Linux
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

#### Windows
```powershell
# Download installer from https://ollama.ai/download
# Or use WSL2 with Linux instructions
```

### Pulling Models

```bash
# Small models (< 2GB VRAM)
ollama pull tinyllama      # 1.1B parameters, very fast
ollama pull phi            # 2.7B parameters, good quality

# Medium models (4-8GB VRAM)
ollama pull mistral        # 7B parameters, excellent quality
ollama pull llama2:7b      # 7B parameters, versatile
ollama pull openchat       # 7B parameters, conversation-optimized

# Large models (16GB+ VRAM)
ollama pull llama2:13b     # 13B parameters
ollama pull mixtral        # 8x7B MoE, very capable
ollama pull llama2:70b     # 70B parameters, top quality

# Quantized versions (less VRAM)
ollama pull llama2:7b-q4_0   # 4-bit quantization
ollama pull mistral:7b-q5_1   # 5-bit quantization
```

### Auto-Configuration

The platform automatically:
1. Detects Ollama installation
2. Starts Ollama if not running
3. Pulls recommended models based on available VRAM
4. Allocates GPU layers optimally

### Manual Configuration

```bash
# Set Ollama host (if not localhost)
export OLLAMA_HOST=http://192.168.1.100:11434

# Set GPU layers (for partial GPU offload)
export OLLAMA_NUM_GPU=20  # Number of layers to offload

# Disable auto-start
export AUTO_START_OLLAMA=false

# Set custom models to auto-pull
export OLLAMA_AUTO_MODELS="tinyllama,phi,mistral:7b-instruct"
```

---

## Cloud Providers

### OpenAI

#### Setup
```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."

# Optional: Set organization
export OPENAI_ORG_ID="org-..."

# Optional: Use specific models
export OPENAI_MODELS="gpt-4-turbo-preview,gpt-3.5-turbo"
```

#### Available Models
- `gpt-4-turbo-preview` - Latest GPT-4 (128k context)
- `gpt-4` - Stable GPT-4 (8k context)
- `gpt-3.5-turbo` - Fast and cheap (4k context)
- `gpt-3.5-turbo-16k` - Extended context

### Anthropic

#### Setup
```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Set specific models
export ANTHROPIC_MODELS="claude-3-opus-20240229,claude-3-sonnet-20240229"
```

#### Available Models
- `claude-3-opus-20240229` - Most capable (200k context)
- `claude-3-sonnet-20240229` - Balanced (200k context)
- `claude-3-haiku-20240307` - Fast and cheap (200k context)

### Google Gemini

#### Setup
```bash
# Get API key from https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="..."

# Install Python SDK
pip install google-generativeai
```

#### Available Models
- `gemini-pro` - Text generation
- `gemini-pro-vision` - Multimodal (text + images)
- `gemini-ultra` - Most capable (coming soon)

### xAI Grok

#### Setup
```bash
# Get API key from https://x.ai/api
export XAI_API_KEY="xai-..."

# Grok uses OpenAI-compatible API
export GROK_BASE_URL="https://api.x.ai/v1"
```

#### Available Models
- `grok-beta` - Early access model

### OpenRouter

#### Setup
```bash
# Get API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY="sk-or-..."

# OpenRouter provides access to 100+ models
```

#### Popular Models
- `anthropic/claude-3-opus` - Via OpenRouter
- `openai/gpt-4-turbo` - Via OpenRouter
- `google/gemini-pro` - Via OpenRouter
- `meta-llama/llama-2-70b` - Via OpenRouter
- `mistralai/mixtral-8x7b` - Via OpenRouter

---

## Mac Metal Support (MLX)

### Installation

```bash
# Requires macOS 13.0+ with Apple Silicon
pip install mlx mlx-lm

# Download models
python -c "from mlx_lm import load; load('mlx-community/Llama-3-8B-4bit')"
```

### Available Models
```python
# Optimized for Apple Silicon
models = [
    "mlx-community/Llama-3-8B-4bit",
    "mlx-community/Mistral-7B-v0.1-4bit",
    "mlx-community/phi-3-mini-4bit",
    "mlx-community/gemma-7b-4bit"
]
```

### Performance
- **M1**: 7B models at 20-30 tokens/sec
- **M1 Pro/Max**: 13B models at 15-25 tokens/sec
- **M2 Ultra**: 70B models at 10-15 tokens/sec

---

## Configuration

### Environment Variables

Create a `.env` file:
```env
# Local Models
OLLAMA_HOST=http://localhost:11434
AUTO_START_OLLAMA=true
OLLAMA_AUTO_MODELS=tinyllama,mistral:7b

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_DEFAULT_MODEL=claude-3-haiku-20240307

# Google Gemini
GOOGLE_API_KEY=...
GEMINI_DEFAULT_MODEL=gemini-pro

# xAI Grok
XAI_API_KEY=xai-...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Model Selection
DEFAULT_PROVIDER=ollama  # ollama, openai, anthropic, gemini, etc.
FALLBACK_PROVIDERS=anthropic,openai  # Comma-separated fallback chain

# Resource Management
MAX_GPU_MEMORY_GB=8
MAX_CPU_MEMORY_GB=16
MAX_CONCURRENT_MODELS=3

# Cost Management
MAX_MONTHLY_SPEND_USD=100
COST_ALERT_THRESHOLD_USD=50
```

### Agent Configuration

Configure models per agent type:
```python
# config/agent_models.py
AGENT_MODEL_CONFIG = {
    "simple_agent": {
        "provider": "ollama",
        "model": "tinyllama",
        "temperature": 0.7
    },
    "merchant_agent": {
        "provider": "ollama",
        "model": "mistral:7b",
        "temperature": 0.5
    },
    "leader_agent": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.8
    },
    "creative_agent": {
        "provider": "openai",
        "model": "gpt-4-turbo-preview",
        "temperature": 0.9
    }
}
```

---

## Testing Your Setup

### Quick Test
```bash
# Test all configured providers
python services/agent-scheduler/test_llm_system.py

# Test specific provider
python -c "
from llm_manager import MultiProviderLLM
import asyncio

async def test():
    llm = MultiProviderLLM()
    await llm.initialize()
    response = await llm.generate('Hello!', provider='ollama')
    print(response)

asyncio.run(test())
"
```

### Benchmark Script
```python
# benchmark_llms.py
import asyncio
from llm_manager import MultiProviderLLM

async def benchmark():
    llm = MultiProviderLLM()
    await llm.initialize()

    results = await llm.benchmark_models(
        prompt="Explain quantum computing in one sentence."
    )

    for model, metrics in results.items():
        if metrics['success']:
            print(f"{model}: {metrics['response_time']:.2f}s")

asyncio.run(benchmark())
```

### Expected Output
```
System Resources Detected:
- Platform: darwin (macOS)
- RAM: 16.0 GB available
- GPUs: 1 (Apple M1)
- Metal: Available

Providers Initialized:
- ollama: 3 models (tinyllama, phi, mistral)
- openai: 2 models (gpt-4, gpt-3.5-turbo)
- anthropic: 3 models (claude-3-opus, sonnet, haiku)

Test Results:
✅ ollama/tinyllama: 0.3s (150 tokens/sec)
✅ ollama/mistral: 1.2s (40 tokens/sec)
✅ openai/gpt-3.5-turbo: 0.8s
✅ anthropic/claude-3-haiku: 0.6s

Recommendations:
- Best local model: ollama/mistral
- Best cloud model: anthropic/claude-3-haiku
- Fallback chain: ollama -> anthropic -> openai
```

---

## Performance Optimization

### GPU Optimization

#### NVIDIA GPUs
```bash
# Check CUDA version
nvidia-smi

# Set GPU memory fraction
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

# For Ollama
export OLLAMA_NUM_GPU=-1  # Use all GPU layers
```

#### AMD GPUs
```bash
# Check ROCm version
rocm-smi

# Set GPU for Ollama
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For specific GPU
```

#### Apple Silicon
```bash
# MLX automatically uses Metal
# No configuration needed
```

### Model Selection by Task

| Task | Recommended Model | Reason |
|------|------------------|---------|
| Simple decisions | tinyllama, phi | Fast, low resource |
| Conversations | mistral:7b-instruct | Good dialogue |
| Planning | llama2:13b | Better reasoning |
| Creative writing | gpt-4-turbo | Best quality |
| Code generation | claude-3-opus | Excellent coding |
| Vision tasks | gemini-pro-vision | Multimodal |

### Batching Strategies

```python
# Batch multiple agent decisions
async def batch_decisions(agents, observations):
    tasks = []
    for agent, obs in zip(agents, observations):
        tasks.append(
            llm.generate(
                prompt=create_prompt(agent, obs),
                model="tinyllama"  # Fast model for batch
            )
        )

    responses = await asyncio.gather(*tasks)
    return responses
```

---

## Cost Management

### Estimated Costs (per 1M tokens)

| Provider | Model | Input Cost | Output Cost |
|----------|-------|------------|-------------|
| OpenAI | GPT-4-Turbo | $10 | $30 |
| OpenAI | GPT-3.5-Turbo | $0.50 | $1.50 |
| Anthropic | Claude 3 Opus | $15 | $75 |
| Anthropic | Claude 3 Sonnet | $3 | $15 |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 |
| Google | Gemini Pro | $0.50 | $1.50 |
| OpenRouter | Varies | $0.10-20 | $0.50-100 |
| Ollama | All models | Free | Free |

### Cost Optimization Tips

1. **Use local models when possible**
   - Ollama models are free and private
   - Good for 90% of agent decisions

2. **Implement smart routing**
   ```python
   # Route by complexity
   if is_simple_task(prompt):
       model = "tinyllama"  # Free, fast
   elif is_medium_task(prompt):
       model = "claude-3-haiku"  # Cheap, capable
   else:
       model = "gpt-4-turbo"  # Expensive, powerful
   ```

3. **Cache responses**
   ```python
   # Cache common queries
   cache_key = hash(prompt)
   if cache_key in response_cache:
       return response_cache[cache_key]
   ```

4. **Monitor usage**
   ```python
   # Track costs
   costs = {
       "openai": {"tokens": 0, "cost": 0},
       "anthropic": {"tokens": 0, "cost": 0}
   }
   ```

---

## Troubleshooting

### Common Issues

#### Ollama Not Detected
```bash
# Check if Ollama is installed
which ollama

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually
ollama serve

# Check logs
journalctl -u ollama  # Linux
tail -f ~/.ollama/logs/server.log  # macOS
```

#### GPU Not Detected
```bash
# NVIDIA
nvidia-smi  # Should show GPU info
nvcc --version  # Check CUDA

# AMD
rocm-smi  # Should show GPU info

# Apple Silicon
system_profiler SPDisplaysDataType | grep Metal
```

#### Out of Memory
```python
# Reduce GPU layers
export OLLAMA_NUM_GPU=10  # Use only 10 layers on GPU

# Use quantized models
ollama pull llama2:7b-q4_0  # 4-bit quantization

# Clear GPU memory
ollama stop  # Stop all models
```

#### Slow Performance
```python
# Check resource usage
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"RAM: {psutil.virtual_memory().percent}%")

# Use smaller models
model = "tinyllama" if low_resources else "mistral"

# Reduce context length
max_tokens = 100 if low_resources else 500
```

#### API Errors
```python
# Check API keys
import os
assert os.getenv("OPENAI_API_KEY"), "OpenAI key not set"

# Test connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Use fallback
try:
    response = await llm.generate(prompt, provider="openai")
except:
    response = await llm.generate(prompt, provider="ollama")
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with verbose output
python test_llm_system.py --verbose
```

---

## Advanced Configuration

### Custom Model Deployment

```python
# Deploy custom GGUF model with Ollama
# 1. Create Modelfile
cat > Modelfile << EOF
FROM ./my-custom-model.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a helpful assistant."
EOF

# 2. Create model
ollama create my-custom-model -f Modelfile

# 3. Use in platform
export OLLAMA_AUTO_MODELS="my-custom-model"
```

### Fine-tuning Integration

```python
# Use fine-tuned OpenAI model
AGENT_CONFIG = {
    "specialized_agent": {
        "provider": "openai",
        "model": "ft:gpt-3.5-turbo-0613:my-org::8abc123",
        "temperature": 0.7
    }
}
```

### Load Balancing

```python
# Distribute load across multiple Ollama instances
OLLAMA_SERVERS = [
    "http://server1:11434",
    "http://server2:11434",
    "http://server3:11434"
]

# Round-robin selection
server = OLLAMA_SERVERS[agent_id % len(OLLAMA_SERVERS)]
```

---

## Best Practices

1. **Start with Ollama** for development and testing
2. **Use quantized models** to save memory
3. **Implement fallback chains** for reliability
4. **Cache responses** for common queries
5. **Monitor costs** for cloud providers
6. **Profile performance** regularly
7. **Update models** periodically for improvements
8. **Test thoroughly** before production deployment

---

## Support

- **Ollama Documentation**: https://github.com/ollama/ollama
- **OpenAI API**: https://platform.openai.com/docs
- **Anthropic API**: https://docs.anthropic.com
- **Google AI**: https://ai.google.dev
- **MLX Documentation**: https://ml-explore.github.io/mlx/

For platform-specific issues, see our [GitHub Issues](https://github.com/your-org/multi-agent-city/issues).