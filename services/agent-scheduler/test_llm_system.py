"""Comprehensive testing system for multi-provider LLM support"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import statistics

from llm_manager import MultiProviderLLM, LLMProvider, OllamaManager, LLMResourceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test"""
    provider: str
    model: str
    test_name: str
    success: bool
    response_time: float
    tokens_per_second: Optional[float] = None
    memory_used: Optional[float] = None
    error: Optional[str] = None
    response: Optional[str] = None


class LLMTestSuite:
    """Comprehensive test suite for LLM systems"""

    def __init__(self):
        self.llm_manager = MultiProviderLLM()
        self.results = []
        self.test_prompts = {
            "simple": "What is 2+2?",
            "reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
            "coding": "Write a Python function that finds the factorial of a number.",
            "creative": "Write a haiku about artificial intelligence.",
            "conversation": "Hello! I'm testing different AI models. Can you tell me something interesting about yourself?",
            "agent_persona": "You are a helpful shopkeeper in a virtual city. A customer asks you about the best deals today. How do you respond?",
            "memory_recall": "Remember these items: apple, bicycle, cloud, dragon, elephant. Now, what was the third item?",
            "planning": "You need to travel from New York to Los Angeles. List 3 different ways to make this journey and their pros/cons."
        }

    async def run_full_test_suite(self) -> Dict:
        """Run complete test suite on all available models"""
        logger.info("=" * 60)
        logger.info("Starting Comprehensive LLM System Test")
        logger.info("=" * 60)

        # Initialize LLM manager
        await self.llm_manager.initialize()

        # Get available models
        available_models = self.llm_manager.get_available_models()
        logger.info(f"\nFound {sum(len(m) for m in available_models.values())} models across {len(available_models)} providers")

        # Test system resources
        await self.test_system_resources()

        # Test each provider
        for provider_name, models in available_models.items():
            await self.test_provider(provider_name, models)

        # Run comparative tests
        await self.run_comparative_tests()

        # Generate report
        report = self.generate_report()

        return report

    async def test_system_resources(self):
        """Test system resource detection"""
        logger.info("\n" + "=" * 40)
        logger.info("Testing System Resources")
        logger.info("=" * 40)

        resource_manager = LLMResourceManager()
        resources = await resource_manager.detect_system_resources()

        logger.info(f"Platform: {resources.platform}")
        logger.info(f"CPU Cores: {resources.cpu_cores}")
        logger.info(f"RAM: {resources.available_ram_gb:.1f}/{resources.total_ram_gb:.1f} GB available")
        logger.info(f"GPUs: {resources.gpu_count}")

        if resources.gpu_count > 0:
            for gpu in resources.gpu_info:
                logger.info(f"  - {gpu['name']}: {gpu['memory_free']:.1f}/{gpu['memory_total']:.1f} MB free")

        logger.info(f"CUDA available: {resources.has_cuda}")
        logger.info(f"Metal available: {resources.has_metal}")
        logger.info(f"ROCm available: {resources.has_rocm}")
        logger.info(f"Available ports: {resources.available_ports[:5]}")

    async def test_provider(self, provider_name: str, models: List[str]):
        """Test all models from a specific provider"""
        logger.info(f"\n" + "=" * 40)
        logger.info(f"Testing Provider: {provider_name}")
        logger.info("=" * 40)

        # Test only first model for each provider to save time
        # In production, you might want to test all models
        models_to_test = models[:1] if len(models) > 0 else []

        for model in models_to_test:
            logger.info(f"\nTesting model: {model}")

            # Run basic functionality test
            result = await self.test_model_basic(provider_name, model)
            self.results.append(result)

            if result.success:
                # Run performance test
                perf_result = await self.test_model_performance(provider_name, model)
                self.results.append(perf_result)

                # Run agent simulation test
                agent_result = await self.test_agent_simulation(provider_name, model)
                self.results.append(agent_result)

    async def test_model_basic(self, provider: str, model: str) -> TestResult:
        """Test basic functionality of a model"""
        try:
            start_time = time.time()

            provider_enum = LLMProvider[provider.upper()] if provider.upper() in LLMProvider.__members__ else None

            response = await self.llm_manager.generate(
                prompt=self.test_prompts["simple"],
                model=model,
                provider=provider_enum,
                max_tokens=50
            )

            elapsed = time.time() - start_time

            # Validate response
            success = response and len(response) > 0 and ("4" in response or "four" in response.lower())

            return TestResult(
                provider=provider,
                model=model,
                test_name="basic_functionality",
                success=success,
                response_time=elapsed,
                response=response[:100] if response else None
            )

        except Exception as e:
            return TestResult(
                provider=provider,
                model=model,
                test_name="basic_functionality",
                success=False,
                response_time=0,
                error=str(e)
            )

    async def test_model_performance(self, provider: str, model: str) -> TestResult:
        """Test performance characteristics of a model"""
        try:
            prompts = [self.test_prompts["simple"], self.test_prompts["reasoning"], self.test_prompts["coding"]]
            response_times = []
            token_counts = []

            for prompt in prompts:
                start_time = time.time()

                provider_enum = LLMProvider[provider.upper()] if provider.upper() in LLMProvider.__members__ else None

                response = await self.llm_manager.generate(
                    prompt=prompt,
                    model=model,
                    provider=provider_enum,
                    max_tokens=100
                )

                elapsed = time.time() - start_time
                response_times.append(elapsed)

                # Estimate tokens (rough approximation)
                token_counts.append(len(response.split()) if response else 0)

            avg_response_time = statistics.mean(response_times)
            avg_tokens = statistics.mean(token_counts)
            tokens_per_second = avg_tokens / avg_response_time if avg_response_time > 0 else 0

            return TestResult(
                provider=provider,
                model=model,
                test_name="performance",
                success=True,
                response_time=avg_response_time,
                tokens_per_second=tokens_per_second
            )

        except Exception as e:
            return TestResult(
                provider=provider,
                model=model,
                test_name="performance",
                success=False,
                response_time=0,
                error=str(e)
            )

    async def test_agent_simulation(self, provider: str, model: str) -> TestResult:
        """Test model in agent simulation context"""
        try:
            # Simulate an agent decision-making scenario
            agent_prompt = """You are an autonomous agent in a virtual city.
Current situation:
- Location: Market Square
- Time: Afternoon
- Nearby: Coffee shop, Bookstore, Park
- Current goal: Socialize and explore
- Energy: 80%
- Money: 100 credits

What action do you take next? Respond with:
1. ACTION: (the action you take)
2. REASON: (why you chose this action)
Keep your response under 100 words."""

            start_time = time.time()

            provider_enum = LLMProvider[provider.upper()] if provider.upper() in LLMProvider.__members__ else None

            response = await self.llm_manager.generate(
                prompt=agent_prompt,
                model=model,
                provider=provider_enum,
                max_tokens=150,
                temperature=0.7
            )

            elapsed = time.time() - start_time

            # Check if response contains expected structure
            success = response and "ACTION:" in response and "REASON:" in response

            return TestResult(
                provider=provider,
                model=model,
                test_name="agent_simulation",
                success=success,
                response_time=elapsed,
                response=response[:200] if response else None
            )

        except Exception as e:
            return TestResult(
                provider=provider,
                model=model,
                test_name="agent_simulation",
                success=False,
                response_time=0,
                error=str(e)
            )

    async def run_comparative_tests(self):
        """Run tests comparing different models"""
        logger.info("\n" + "=" * 40)
        logger.info("Running Comparative Tests")
        logger.info("=" * 40)

        # Get one model from each provider for comparison
        test_models = []
        available = self.llm_manager.get_available_models()

        for provider, models in available.items():
            if models:
                test_models.append((provider, models[0]))

        if len(test_models) < 2:
            logger.info("Not enough models for comparative testing")
            return

        # Test prompt consistency
        test_prompt = self.test_prompts["reasoning"]
        logger.info(f"\nTesting prompt: {test_prompt[:50]}...")

        for provider, model in test_models[:3]:  # Test up to 3 models
            try:
                provider_enum = LLMProvider[provider.upper()] if provider.upper() in LLMProvider.__members__ else None

                response = await self.llm_manager.generate(
                    prompt=test_prompt,
                    model=model,
                    provider=provider_enum,
                    max_tokens=100,
                    temperature=0.3  # Lower temperature for more consistent results
                )

                logger.info(f"\n{provider}/{model}:")
                logger.info(f"Response: {response[:150]}...")

            except Exception as e:
                logger.error(f"Error testing {provider}/{model}: {e}")

    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "providers_tested": len(set(r.provider for r in self.results)),
                "models_tested": len(set(f"{r.provider}/{r.model}" for r in self.results))
            },
            "providers": {},
            "recommendations": []
        }

        # Group results by provider
        for result in self.results:
            if result.provider not in report["providers"]:
                report["providers"][result.provider] = {
                    "models": {},
                    "success_rate": 0,
                    "avg_response_time": 0
                }

            if result.model not in report["providers"][result.provider]["models"]:
                report["providers"][result.provider]["models"][result.model] = []

            report["providers"][result.provider]["models"][result.model].append({
                "test": result.test_name,
                "success": result.success,
                "response_time": result.response_time,
                "tokens_per_second": result.tokens_per_second,
                "error": result.error
            })

        # Calculate provider statistics
        for provider in report["providers"]:
            provider_results = [r for r in self.results if r.provider == provider]
            if provider_results:
                report["providers"][provider]["success_rate"] = (
                    sum(1 for r in provider_results if r.success) / len(provider_results) * 100
                )
                report["providers"][provider]["avg_response_time"] = (
                    statistics.mean([r.response_time for r in provider_results if r.response_time > 0])
                    if any(r.response_time > 0 for r in provider_results) else 0
                )

        # Generate recommendations
        report["recommendations"] = self.generate_recommendations(report)

        return report

    def generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Find best performer
        best_provider = None
        best_success_rate = 0

        for provider, data in report["providers"].items():
            if data["success_rate"] > best_success_rate:
                best_success_rate = data["success_rate"]
                best_provider = provider

        if best_provider:
            recommendations.append(f"Best overall provider: {best_provider} with {best_success_rate:.1f}% success rate")

        # Find fastest provider
        fastest_provider = None
        fastest_time = float('inf')

        for provider, data in report["providers"].items():
            if 0 < data["avg_response_time"] < fastest_time:
                fastest_time = data["avg_response_time"]
                fastest_provider = provider

        if fastest_provider:
            recommendations.append(f"Fastest provider: {fastest_provider} with {fastest_time:.2f}s average response time")

        # Check for local models
        if "ollama" in report["providers"] and report["providers"]["ollama"]["success_rate"] > 80:
            recommendations.append("Ollama is working well - recommended for privacy and offline use")

        # Check for cloud providers
        cloud_providers = ["openai", "anthropic", "gemini"]
        working_cloud = [p for p in cloud_providers if p in report["providers"] and report["providers"][p]["success_rate"] > 80]

        if working_cloud:
            recommendations.append(f"Cloud providers available: {', '.join(working_cloud)} - recommended for advanced capabilities")

        return recommendations

    def print_report(self, report: Dict):
        """Print formatted test report"""
        print("\n" + "=" * 60)
        print("LLM SYSTEM TEST REPORT")
        print("=" * 60)

        print("\nSUMMARY:")
        print(f"  Total Tests: {report['summary']['total_tests']}")
        print(f"  Successful: {report['summary']['successful_tests']}")
        print(f"  Failed: {report['summary']['failed_tests']}")
        print(f"  Providers Tested: {report['summary']['providers_tested']}")
        print(f"  Models Tested: {report['summary']['models_tested']}")

        print("\nPROVIDER RESULTS:")
        for provider, data in report["providers"].items():
            print(f"\n  {provider.upper()}:")
            print(f"    Success Rate: {data['success_rate']:.1f}%")
            print(f"    Avg Response Time: {data['avg_response_time']:.2f}s")
            print(f"    Models Tested: {len(data['models'])}")

        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

        print("\n" + "=" * 60)


async def test_ollama_specific():
    """Test Ollama-specific functionality"""
    logger.info("Testing Ollama Installation and Models")

    ollama_manager = OllamaManager()

    # Detect Ollama
    is_running, base_url = await ollama_manager.detect_ollama()
    logger.info(f"Ollama detected: {is_running}, URL: {base_url}")

    if not is_running:
        logger.info("Attempting to start Ollama...")
        success = await ollama_manager.start_ollama()
        logger.info(f"Ollama started: {success}")

    if ollama_manager.is_running:
        logger.info(f"Available models: {ollama_manager.available_models}")

        # Try to pull a small model if none available
        if not ollama_manager.available_models:
            logger.info("No models found, pulling tinyllama...")
            success = await ollama_manager.pull_model("tinyllama")
            logger.info(f"Model pull success: {success}")


async def main():
    """Main test entry point"""
    # Run full test suite
    test_suite = LLMTestSuite()
    report = await test_suite.run_full_test_suite()

    # Print report
    test_suite.print_report(report)

    # Save report to file
    with open("llm_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("\nTest report saved to llm_test_report.json")


if __name__ == "__main__":
    # Test specific components
    # asyncio.run(test_ollama_specific())

    # Run full test suite
    asyncio.run(main())