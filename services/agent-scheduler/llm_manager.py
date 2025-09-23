"""Comprehensive LLM Manager with multi-provider support and resource management"""

import asyncio
import logging
import os
import sys
import platform
import subprocess
import psutil
import json
import httpx
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import GPUtil
import aiohttp

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROK = "grok"
    OPENROUTER = "openrouter"
    MLX = "mlx"  # Mac Metal
    LOCAL_LLAMACPP = "llamacpp"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    context_window: int = 4096
    cost_per_1k_tokens: float = 0.0
    supports_streaming: bool = True
    requires_gpu: bool = False
    min_vram_gb: float = 0.0
    port: Optional[int] = None


@dataclass
class SystemResources:
    """System resource information"""
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    available_vram_gb: float
    cpu_cores: int
    cpu_usage_percent: float
    gpu_count: int
    gpu_info: List[Dict]
    available_ports: List[int]
    platform: str
    has_metal: bool
    has_cuda: bool
    has_rocm: bool


class LLMResourceManager:
    """Manages system resources for LLM deployment"""

    def __init__(self):
        self.allocated_resources = {}
        self.running_models = {}
        self.port_range = (11434, 11534)  # 100 ports for local models

    async def detect_system_resources(self) -> SystemResources:
        """Detect available system resources"""
        try:
            # CPU and RAM
            ram = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)

            # GPU detection
            gpu_info = []
            total_vram = 0.0
            available_vram = 0.0

            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_free": gpu.memoryFree,
                        "memory_used": gpu.memoryUsed,
                        "utilization": gpu.memoryUtil * 100
                    })
                    total_vram += gpu.memoryTotal / 1024  # Convert to GB
                    available_vram += gpu.memoryFree / 1024
            except:
                logger.info("No NVIDIA GPUs detected")

            # Platform detection
            system_platform = platform.system().lower()

            # Check for Metal (macOS)
            has_metal = False
            if system_platform == "darwin":
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType"],
                        capture_output=True,
                        text=True
                    )
                    has_metal = "Metal" in result.stdout
                except:
                    pass

            # Check for CUDA
            has_cuda = self._check_cuda()

            # Check for ROCm (AMD)
            has_rocm = self._check_rocm()

            # Find available ports
            available_ports = await self._find_available_ports()

            return SystemResources(
                total_ram_gb=ram.total / (1024**3),
                available_ram_gb=ram.available / (1024**3),
                total_vram_gb=total_vram,
                available_vram_gb=available_vram,
                cpu_cores=psutil.cpu_count(),
                cpu_usage_percent=cpu_percent,
                gpu_count=len(gpu_info),
                gpu_info=gpu_info,
                available_ports=available_ports,
                platform=system_platform,
                has_metal=has_metal,
                has_cuda=has_cuda,
                has_rocm=has_rocm
            )

        except Exception as e:
            logger.error(f"Error detecting system resources: {e}")
            return SystemResources(
                total_ram_gb=8.0,
                available_ram_gb=4.0,
                total_vram_gb=0.0,
                available_vram_gb=0.0,
                cpu_cores=4,
                cpu_usage_percent=50.0,
                gpu_count=0,
                gpu_info=[],
                available_ports=[11434],
                platform=platform.system().lower(),
                has_metal=False,
                has_cuda=False,
                has_rocm=False
            )

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    def _check_rocm(self) -> bool:
        """Check if ROCm is available"""
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False

    async def _find_available_ports(self) -> List[int]:
        """Find available ports in the range"""
        import socket
        available = []

        for port in range(self.port_range[0], self.port_range[1]):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
                available.append(port)
                sock.close()
            except:
                continue

            if len(available) >= 10:  # Find at most 10 available ports
                break

        return available

    async def allocate_resources_for_model(
        self,
        model_name: str,
        provider: LLMProvider,
        resources: SystemResources
    ) -> Optional[Dict]:
        """Allocate resources for a model"""

        # Estimate resource requirements
        requirements = self._estimate_model_requirements(model_name, provider)

        # Check if resources are available
        if requirements["ram_gb"] > resources.available_ram_gb:
            logger.warning(f"Not enough RAM for {model_name}: need {requirements['ram_gb']}GB, have {resources.available_ram_gb}GB")
            return None

        if requirements["vram_gb"] > 0 and requirements["vram_gb"] > resources.available_vram_gb:
            logger.warning(f"Not enough VRAM for {model_name}: need {requirements['vram_gb']}GB, have {resources.available_vram_gb}GB")
            # Try CPU fallback
            requirements["use_gpu"] = False
            requirements["vram_gb"] = 0
            requirements["ram_gb"] *= 1.5  # Increase RAM requirement for CPU

        # Allocate port if needed
        port = None
        if provider in [LLMProvider.OLLAMA, LLMProvider.LOCAL_LLAMACPP, LLMProvider.MLX]:
            if resources.available_ports:
                port = resources.available_ports.pop(0)
            else:
                logger.error("No available ports for local model")
                return None

        allocation = {
            "model": model_name,
            "provider": provider.value,
            "ram_gb": requirements["ram_gb"],
            "vram_gb": requirements["vram_gb"],
            "use_gpu": requirements["use_gpu"],
            "port": port,
            "threads": min(4, resources.cpu_cores // 2)
        }

        self.allocated_resources[model_name] = allocation
        return allocation

    def _estimate_model_requirements(self, model_name: str, provider: LLMProvider) -> Dict:
        """Estimate resource requirements for a model"""

        model_size = model_name.lower()

        # Base requirements by model size
        if "70b" in model_size or "mixtral" in model_size:
            base_ram = 40.0
            base_vram = 40.0
        elif "34b" in model_size or "30b" in model_size:
            base_ram = 20.0
            base_vram = 20.0
        elif "13b" in model_size:
            base_ram = 10.0
            base_vram = 10.0
        elif "7b" in model_size:
            base_ram = 6.0
            base_vram = 6.0
        elif "3b" in model_size or "phi" in model_size:
            base_ram = 3.0
            base_vram = 3.0
        elif "1b" in model_size or "tiny" in model_size:
            base_ram = 1.5
            base_vram = 1.5
        else:
            # Default for unknown models
            base_ram = 4.0
            base_vram = 4.0

        # Adjust for quantization
        if "q4" in model_size or "gguf" in model_size:
            base_ram *= 0.5
            base_vram *= 0.5
        elif "q8" in model_size:
            base_ram *= 0.75
            base_vram *= 0.75

        # Provider-specific adjustments
        use_gpu = provider not in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI]

        return {
            "ram_gb": base_ram,
            "vram_gb": base_vram if use_gpu else 0,
            "use_gpu": use_gpu
        }


class OllamaManager:
    """Manages Ollama installation and models"""

    def __init__(self):
        self.base_url = None
        self.is_running = False
        self.available_models = []
        self.running_models = {}

    async def detect_ollama(self) -> Tuple[bool, Optional[str]]:
        """Detect if Ollama is installed and running"""
        try:
            # Check common Ollama ports
            for port in [11434, 11435, 11436]:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"http://localhost:{port}/api/tags", timeout=2.0)
                        if response.status_code == 200:
                            self.base_url = f"http://localhost:{port}"
                            self.is_running = True
                            await self._fetch_available_models()
                            logger.info(f"Ollama detected at {self.base_url}")
                            return True, self.base_url
                except:
                    continue

            # Check if Ollama is installed but not running
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Ollama is installed but not running")
                return False, None

            logger.info("Ollama not found")
            return False, None

        except Exception as e:
            logger.error(f"Error detecting Ollama: {e}")
            return False, None

    async def start_ollama(self, port: int = 11434) -> bool:
        """Start Ollama server"""
        try:
            # Start Ollama serve
            process = subprocess.Popen(
                ["ollama", "serve"],
                env={**os.environ, "OLLAMA_HOST": f"0.0.0.0:{port}"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for server to start
            await asyncio.sleep(2)

            # Check if running
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    self.base_url = f"http://localhost:{port}"
                    self.is_running = True
                    await self._fetch_available_models()
                    logger.info(f"Ollama started on port {port}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            return False

    async def _fetch_available_models(self):
        """Fetch list of available Ollama models"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                    logger.info(f"Available Ollama models: {self.available_models}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")

    async def pull_model(self, model_name: str) -> bool:
        """Pull an Ollama model"""
        try:
            logger.info(f"Pulling Ollama model: {model_name}")

            # Use subprocess for pulling (streaming)
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for completion
            stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout

            if process.returncode == 0:
                logger.info(f"Successfully pulled {model_name}")
                self.available_models.append(model_name)
                return True
            else:
                logger.error(f"Failed to pull {model_name}: {stderr}")
                return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def load_model(self, model_name: str, num_gpu: int = -1) -> bool:
        """Load a model into memory"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "options": {
                            "num_gpu": num_gpu  # -1 for all layers on GPU
                        }
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    self.running_models[model_name] = True
                    logger.info(f"Loaded model {model_name}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False


class MLXManager:
    """Manages MLX models for Mac Metal acceleration"""

    def __init__(self):
        self.is_available = False
        self.models = {}

    async def detect_mlx(self) -> bool:
        """Detect if MLX is available (Mac with Metal)"""
        try:
            if platform.system().lower() != "darwin":
                return False

            # Try to import mlx
            try:
                import mlx
                import mlx.core as mx
                self.is_available = True
                logger.info("MLX detected for Metal acceleration")
                return True
            except ImportError:
                logger.info("MLX not installed")
                return False

        except Exception as e:
            logger.error(f"Error detecting MLX: {e}")
            return False

    async def install_mlx(self) -> bool:
        """Install MLX for Metal acceleration"""
        try:
            result = subprocess.run(
                ["pip", "install", "mlx", "mlx-lm"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error installing MLX: {e}")
            return False

    async def load_mlx_model(self, model_name: str) -> Optional[Any]:
        """Load an MLX model"""
        try:
            from mlx_lm import load, generate

            # Load model and tokenizer
            model, tokenizer = load(model_name)
            self.models[model_name] = (model, tokenizer)
            logger.info(f"Loaded MLX model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Error loading MLX model: {e}")
            return None


class MultiProviderLLM:
    """Unified interface for multiple LLM providers"""

    def __init__(self):
        self.providers = {}
        self.configs = {}
        self.resource_manager = LLMResourceManager()
        self.ollama_manager = OllamaManager()
        self.mlx_manager = MLXManager()
        self.default_provider = None

    async def initialize(self):
        """Initialize all available LLM providers"""
        logger.info("Initializing Multi-Provider LLM System...")

        # Detect system resources
        resources = await self.resource_manager.detect_system_resources()
        logger.info(f"System: {resources.platform}, RAM: {resources.available_ram_gb:.1f}GB, VRAM: {resources.available_vram_gb:.1f}GB")

        # Initialize providers based on environment and resources
        await self._init_ollama(resources)
        await self._init_openai()
        await self._init_anthropic()
        await self._init_gemini()
        await self._init_grok()
        await self._init_openrouter()

        if resources.platform == "darwin" and resources.has_metal:
            await self._init_mlx(resources)

        # Select default provider
        self._select_default_provider()

        logger.info(f"Initialized {len(self.providers)} LLM providers")
        logger.info(f"Default provider: {self.default_provider}")

    async def _init_ollama(self, resources: SystemResources):
        """Initialize Ollama if available"""
        is_running, base_url = await self.ollama_manager.detect_ollama()

        if not is_running and os.getenv("AUTO_START_OLLAMA", "true").lower() == "true":
            # Try to start Ollama
            if resources.available_ports:
                port = resources.available_ports[0]
                is_running = await self.ollama_manager.start_ollama(port)
                base_url = f"http://localhost:{port}" if is_running else None

        if is_running:
            # Auto-pull recommended models based on resources
            recommended_models = self._get_recommended_models(resources)

            for model in recommended_models:
                if model not in self.ollama_manager.available_models:
                    logger.info(f"Auto-pulling recommended model: {model}")
                    await self.ollama_manager.pull_model(model)

            self.providers[LLMProvider.OLLAMA] = {
                "manager": self.ollama_manager,
                "base_url": base_url,
                "models": self.ollama_manager.available_models
            }

            # Create configs for each model
            for model in self.ollama_manager.available_models:
                self.configs[f"ollama/{model}"] = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model=model,
                    base_url=base_url,
                    requires_gpu=resources.gpu_count > 0,
                    context_window=4096 if "7b" in model.lower() else 2048
                )

    async def _init_openai(self):
        """Initialize OpenAI if API key is available"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.providers[LLMProvider.OPENAI] = {
                "api_key": api_key,
                "models": ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            }

            # Create configs for each model
            model_configs = {
                "gpt-4-turbo-preview": {"context": 128000, "cost": 0.01},
                "gpt-4": {"context": 8192, "cost": 0.03},
                "gpt-3.5-turbo": {"context": 4096, "cost": 0.001},
                "gpt-3.5-turbo-16k": {"context": 16384, "cost": 0.003}
            }

            for model, config in model_configs.items():
                self.configs[f"openai/{model}"] = LLMConfig(
                    provider=LLMProvider.OPENAI,
                    model=model,
                    api_key=api_key,
                    context_window=config["context"],
                    cost_per_1k_tokens=config["cost"]
                )

    async def _init_anthropic(self):
        """Initialize Anthropic if API key is available"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.providers[LLMProvider.ANTHROPIC] = {
                "api_key": api_key,
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            }

            model_configs = {
                "claude-3-opus-20240229": {"context": 200000, "cost": 0.015},
                "claude-3-sonnet-20240229": {"context": 200000, "cost": 0.003},
                "claude-3-haiku-20240307": {"context": 200000, "cost": 0.00025}
            }

            for model, config in model_configs.items():
                self.configs[f"anthropic/{model}"] = LLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model=model,
                    api_key=api_key,
                    context_window=config["context"],
                    cost_per_1k_tokens=config["cost"]
                )

    async def _init_gemini(self):
        """Initialize Google Gemini if API key is available"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.providers[LLMProvider.GEMINI] = {
                "api_key": api_key,
                "models": ["gemini-pro", "gemini-pro-vision"]
            }

            for model in ["gemini-pro", "gemini-pro-vision"]:
                self.configs[f"gemini/{model}"] = LLMConfig(
                    provider=LLMProvider.GEMINI,
                    model=model,
                    api_key=api_key,
                    context_window=32768,
                    cost_per_1k_tokens=0.00025
                )

    async def _init_grok(self):
        """Initialize xAI Grok if API key is available"""
        api_key = os.getenv("XAI_API_KEY")
        if api_key:
            self.providers[LLMProvider.GROK] = {
                "api_key": api_key,
                "base_url": "https://api.x.ai/v1",
                "models": ["grok-beta"]
            }

            self.configs["grok/grok-beta"] = LLMConfig(
                provider=LLMProvider.GROK,
                model="grok-beta",
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                context_window=8192
            )

    async def _init_openrouter(self):
        """Initialize OpenRouter for access to multiple models"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self.providers[LLMProvider.OPENROUTER] = {
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "models": [
                    "anthropic/claude-3-opus",
                    "openai/gpt-4-turbo",
                    "google/gemini-pro",
                    "meta-llama/llama-2-70b-chat",
                    "mistralai/mixtral-8x7b"
                ]
            }

            for model in self.providers[LLMProvider.OPENROUTER]["models"]:
                self.configs[f"openrouter/{model}"] = LLMConfig(
                    provider=LLMProvider.OPENROUTER,
                    model=model,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    context_window=4096
                )

    async def _init_mlx(self, resources: SystemResources):
        """Initialize MLX for Mac Metal acceleration"""
        if await self.mlx_manager.detect_mlx():
            self.providers[LLMProvider.MLX] = {
                "manager": self.mlx_manager,
                "models": ["mlx-community/Llama-3-8B-4bit", "mlx-community/phi-3-mini-4bit"]
            }

            for model in self.providers[LLMProvider.MLX]["models"]:
                self.configs[f"mlx/{model}"] = LLMConfig(
                    provider=LLMProvider.MLX,
                    model=model,
                    requires_gpu=True,
                    context_window=4096
                )

    def _get_recommended_models(self, resources: SystemResources) -> List[str]:
        """Get recommended models based on available resources"""
        models = []

        if resources.available_vram_gb >= 24:
            models.extend(["mixtral:8x7b", "llama2:70b"])
        elif resources.available_vram_gb >= 12:
            models.extend(["llama2:13b", "mistral:7b"])
        elif resources.available_vram_gb >= 6:
            models.extend(["llama2:7b", "mistral:7b-instruct"])
        elif resources.available_ram_gb >= 8:
            models.extend(["phi", "tinyllama", "orca-mini:3b"])
        else:
            models.append("tinyllama")

        return models

    def _select_default_provider(self):
        """Select the best available provider as default"""
        priority_order = [
            LLMProvider.OLLAMA,  # Prefer local
            LLMProvider.MLX,
            LLMProvider.ANTHROPIC,
            LLMProvider.OPENAI,
            LLMProvider.GEMINI,
            LLMProvider.OPENROUTER,
            LLMProvider.GROK
        ]

        for provider in priority_order:
            if provider in self.providers:
                self.default_provider = provider
                break

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """Generate text using specified or default provider"""

        # Select provider and model
        if provider is None:
            provider = self.default_provider

        if model is None:
            # Use first available model from provider
            if provider in self.providers:
                models = self.providers[provider].get("models", [])
                if models:
                    model = models[0]

        if not provider or not model:
            raise ValueError("No provider or model available")

        # Route to appropriate handler
        if provider == LLMProvider.OLLAMA:
            return await self._generate_ollama(prompt, model, **kwargs)
        elif provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, model, **kwargs)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(prompt, model, **kwargs)
        elif provider == LLMProvider.GEMINI:
            return await self._generate_gemini(prompt, model, **kwargs)
        elif provider == LLMProvider.MLX:
            return await self._generate_mlx(prompt, model, **kwargs)
        elif provider == LLMProvider.OPENROUTER:
            return await self._generate_openrouter(prompt, model, **kwargs)
        elif provider == LLMProvider.GROK:
            return await self._generate_grok(prompt, model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _generate_ollama(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_manager.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", 0.7),
                            "top_p": kwargs.get("top_p", 0.9),
                            "max_tokens": kwargs.get("max_tokens", 1000)
                        }
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    raise Exception(f"Ollama error: {response.text}")

        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    async def _generate_openai(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using OpenAI"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.providers[LLMProvider.OPENAI]["api_key"])

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise

    async def _generate_anthropic(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using Anthropic"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=self.providers[LLMProvider.ANTHROPIC]["api_key"]
            )

            response = await client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            raise

    async def _generate_gemini(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using Google Gemini"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.providers[LLMProvider.GEMINI]["api_key"])

            model = genai.GenerativeModel(model)
            response = model.generate_content(prompt)

            return response.text

        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            raise

    async def _generate_mlx(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using MLX (Mac Metal)"""
        try:
            from mlx_lm import generate

            if model not in self.mlx_manager.models:
                await self.mlx_manager.load_mlx_model(model)

            model_obj, tokenizer = self.mlx_manager.models[model]

            response = generate(
                model_obj,
                tokenizer,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )

            return response

        except Exception as e:
            logger.error(f"Error generating with MLX: {e}")
            raise

    async def _generate_openrouter(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using OpenRouter"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.providers[LLMProvider.OPENROUTER]['api_key']}",
                        "HTTP-Referer": "https://multiagentcity.ai",
                        "X-Title": "Multi-Agent City"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 1000)
                    }
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"OpenRouter error: {response.text}")

        except Exception as e:
            logger.error(f"Error generating with OpenRouter: {e}")
            raise

    async def _generate_grok(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using xAI Grok"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.providers[LLMProvider.GROK]['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.providers[LLMProvider.GROK]['api_key']}"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 1000)
                    }
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Grok error: {response.text}")

        except Exception as e:
            logger.error(f"Error generating with Grok: {e}")
            raise

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models by provider"""
        available = {}
        for provider, info in self.providers.items():
            available[provider.value] = info.get("models", [])
        return available

    def get_model_info(self, model_key: str) -> Optional[LLMConfig]:
        """Get configuration for a specific model"""
        return self.configs.get(model_key)

    async def benchmark_models(self, prompt: str = "Hello, how are you?") -> Dict:
        """Benchmark all available models"""
        results = {}

        for model_key, config in self.configs.items():
            try:
                import time
                start = time.time()

                response = await self.generate(
                    prompt=prompt,
                    model=config.model,
                    provider=config.provider,
                    max_tokens=50
                )

                elapsed = time.time() - start

                results[model_key] = {
                    "success": True,
                    "response_time": elapsed,
                    "response_length": len(response),
                    "cost_estimate": config.cost_per_1k_tokens * 0.05  # ~50 tokens
                }

            except Exception as e:
                results[model_key] = {
                    "success": False,
                    "error": str(e)
                }

        return results