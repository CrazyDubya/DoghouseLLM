import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# Import our comprehensive LLM manager
from llm_manager import MultiProviderLLM, LLMProvider

from packages.shared_types.models import (
    Agent, Action, ActionType, ActionParameters,
    Observation, Memory
)

logger = logging.getLogger(__name__)


class LLMIntegration:
    """Real LLM integration for agent decision making with multi-provider support"""

    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.models = {}
        self.multi_provider_llm = MultiProviderLLM()
        self.initialized = False
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available LLM models"""
        # Initialize multi-provider system
        asyncio.create_task(self._async_init())

        # Keep backward compatibility with existing code
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.models["gpt-4"] = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                api_key=openai_key
            )
            self.models["gpt-3.5"] = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=openai_key
            )
            logger.info("OpenAI models initialized")

        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.models["claude-3"] = ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=0.7,
                anthropic_api_key=anthropic_key
            )
            self.models["claude-instant"] = ChatAnthropic(
                model="claude-instant-1.2",
                temperature=0.7,
                anthropic_api_key=anthropic_key
            )
            logger.info("Anthropic models initialized")

        # Local Ollama models (if available)
        try:
            self.models["llama3"] = Ollama(model="llama3", temperature=0.7)
            self.models["mistral"] = Ollama(model="mistral", temperature=0.7)
            logger.info("Local Ollama models initialized")
        except:
            logger.warning("Ollama not available, skipping local models")

        # Fallback to demo model if no real models available
        if not self.models:
            logger.warning("No LLM API keys found, using demo mode")
            self.models["demo"] = DemoLLM()

    async def _async_init(self):
        """Asynchronously initialize multi-provider LLM system"""
        try:
            await self.multi_provider_llm.initialize()
            self.initialized = True

            # Log available models
            available = self.multi_provider_llm.get_available_models()
            total_models = sum(len(models) for models in available.values())
            logger.info(f"Multi-provider LLM initialized: {total_models} models available across {len(available)} providers")

            # Add multi-provider models to our models dict
            for provider_name, models in available.items():
                for model in models:
                    model_key = f"{provider_name}/{model}"
                    self.models[model_key] = self.multi_provider_llm

        except Exception as e:
            logger.error(f"Error initializing multi-provider LLM: {e}")

    async def decide_action_multi_provider(
        self,
        agent: Agent,
        observation: Observation,
        recent_memories: List[Memory],
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Action]:
        """Decide action using multi-provider LLM system"""
        try:
            # Build context
            memory_context = self._build_memory_context(recent_memories)
            prompt = self._create_decision_prompt(agent, observation, memory_context)

            # Add system prompt
            full_prompt = f"{self._get_system_prompt(agent)}\n\n{prompt}"

            # Generate response using multi-provider system
            response = await self.multi_provider_llm.generate(
                prompt=full_prompt,
                model=model,
                provider=LLMProvider[provider.upper()] if provider else None,
                temperature=0.7,
                max_tokens=500
            )

            # Parse response into action
            action = self._parse_action_response(response, agent.id, observation)
            return action

        except Exception as e:
            logger.error(f"Error in multi-provider LLM decision: {e}")
            return self._get_fallback_action(agent, observation)

    async def decide_action(
        self,
        agent: Agent,
        observation: Observation,
        recent_memories: List[Memory]
    ) -> Optional[Action]:
        """Have agent decide on action using LLM"""
        try:
            # Get the model for this agent
            model_name = agent.model_config.get("model", "demo")
            model = self.models.get(model_name)

            if not model:
                logger.warning(f"Model {model_name} not available, using demo")
                model = self.models.get("demo", DemoLLM())

            # Build context from memories
            memory_context = self._build_memory_context(recent_memories)

            # Create the prompt
            prompt = self._create_decision_prompt(
                agent, observation, memory_context
            )

            # Get LLM response
            if isinstance(model, DemoLLM):
                response = await model.decide(agent, observation)
            else:
                messages = [
                    SystemMessage(content=self._get_system_prompt(agent)),
                    HumanMessage(content=prompt)
                ]
                response = await model.ainvoke(messages)
                response = response.content

            # Parse response into action
            action = self._parse_action_response(response, agent.id, observation)
            return action

        except Exception as e:
            logger.error(f"Error in LLM decision for agent {agent.id}: {e}")
            return self._get_fallback_action(agent, observation)

    def _get_system_prompt(self, agent: Agent) -> str:
        """Generate system prompt for agent"""
        personality = agent.profile.personality
        occupation = agent.profile.occupation
        background = agent.profile.background
        goals = agent.profile.goals

        return f"""You are {agent.name}, a {occupation} living in a virtual city.

Personality traits:
{json.dumps(personality, indent=2)}

Background:
{json.dumps(background, indent=2)}

Your goals:
Short-term: {', '.join(goals.get('short_term', []))}
Long-term: {', '.join(goals.get('long_term', []))}

You must respond with a specific action in JSON format. Available actions:
- speak: Say something (include 'message' and optional 'target')
- move: Move to a new location (include 'destination')
- interact: Interact with an object or agent (include 'object_id' or 'target')
- craft: Create or work on something (include 'item')
- trade: Engage in economic transaction (include 'target' and 'item')
- think: Reflect and plan (no parameters needed)

Always include your reasoning for the chosen action."""

    def _create_decision_prompt(
        self,
        agent: Agent,
        observation: Observation,
        memory_context: str
    ) -> str:
        """Create prompt for decision making"""
        # Format visible agents
        visible_agents = ""
        if observation.visible_agents:
            agent_list = [f"- {a.name} ({a.distance}m away, {a.activity})"
                         for a in observation.visible_agents]
            visible_agents = "Nearby agents:\n" + "\n".join(agent_list)

        # Format recent messages
        messages = ""
        if observation.audible_messages:
            msg_list = [f"- {m.speaker_id}: {m.message}"
                       for m in observation.audible_messages]
            messages = "Recent messages:\n" + "\n".join(msg_list)

        return f"""Current situation:
Location: {observation.location.district}, {observation.location.neighborhood}
Time: {observation.environment.time_of_day}
Weather: {observation.environment.weather}
Crowd level: {observation.environment.crowd_level}

{visible_agents}
{messages}

Recent memories:
{memory_context}

Available actions: {', '.join(observation.available_actions)}

What action do you take? Respond with JSON:
{{
    "action": "action_type",
    "parameters": {{}},
    "reasoning": "why you chose this action"
}}"""

    def _build_memory_context(self, memories: List[Memory]) -> str:
        """Build context from recent memories"""
        if not memories:
            return "No recent memories."

        memory_strings = []
        for memory in memories[-10:]:  # Last 10 memories
            importance_marker = "!" if memory.importance > 0.7 else ""
            memory_strings.append(
                f"- {memory.content} {importance_marker}"
            )

        return "\n".join(memory_strings)

    def _parse_action_response(
        self,
        response: str,
        agent_id,
        observation: Observation
    ) -> Optional[Action]:
        """Parse LLM response into an Action"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                action_data = json.loads(json_match.group())
            else:
                # Try to parse entire response as JSON
                action_data = json.loads(response)

            action_type = action_data.get("action", "think")
            parameters = action_data.get("parameters", {})
            reasoning = action_data.get("reasoning", "")

            # Map to ActionType enum
            action_type_map = {
                "speak": ActionType.SPEAK,
                "move": ActionType.MOVE,
                "interact": ActionType.INTERACT,
                "craft": ActionType.CRAFT,
                "trade": ActionType.TRADE,
                "think": ActionType.THINK
            }

            # Create ActionParameters
            action_params = ActionParameters()
            if "message" in parameters:
                action_params.message = parameters["message"]
            if "target" in parameters:
                action_params.target = parameters["target"]
            if "destination" in parameters:
                # Simple destination handling
                action_params.destination = observation.location
            if "item" in parameters:
                action_params.item = parameters["item"]

            return Action(
                agent_id=agent_id,
                type=action_type_map.get(action_type, ActionType.THINK),
                parameters=action_params,
                reasoning=reasoning,
                confidence=0.8
            )

        except Exception as e:
            logger.error(f"Error parsing action response: {e}")
            logger.debug(f"Response was: {response}")
            return None

    def _get_fallback_action(self, agent: Agent, observation: Observation) -> Action:
        """Get fallback action when LLM fails"""
        # Simple fallback logic
        if observation.audible_messages:
            # Respond to messages
            return Action(
                agent_id=agent.id,
                type=ActionType.SPEAK,
                parameters=ActionParameters(
                    message="I'm listening and thinking about what you said.",
                    target="broadcast"
                ),
                reasoning="Acknowledging communication"
            )
        elif observation.visible_agents and len(observation.visible_agents) > 2:
            # Move if crowded
            return Action(
                agent_id=agent.id,
                type=ActionType.MOVE,
                parameters=ActionParameters(
                    destination=observation.location
                ),
                reasoning="Area is crowded, considering moving"
            )
        else:
            # Default to thinking
            return Action(
                agent_id=agent.id,
                type=ActionType.THINK,
                parameters=ActionParameters(),
                reasoning="Observing and reflecting"
            )

    async def generate_dialogue(
        self,
        agent: Agent,
        context: str,
        target: Optional[str] = None
    ) -> str:
        """Generate dialogue for agent"""
        model_name = agent.model_config.get("model", "demo")
        model = self.models.get(model_name, self.models.get("demo"))

        if isinstance(model, DemoLLM):
            return f"Hello from {agent.name}!"

        prompt = f"""As {agent.name}, respond to this situation:
{context}

Your response should be in character, natural, and brief (1-2 sentences).
You are a {agent.profile.occupation}."""

        try:
            messages = [
                SystemMessage(content=self._get_system_prompt(agent)),
                HumanMessage(content=prompt)
            ]
            response = await model.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating dialogue: {e}")
            return f"Hello, I'm {agent.name}."


class DemoLLM:
    """Fallback demo LLM for testing without API keys"""

    async def decide(self, agent: Agent, observation: Observation) -> str:
        """Simple rule-based decision making"""
        import random

        # Simulate different behaviors based on personality
        personality = agent.profile.personality

        if observation.audible_messages:
            # Respond to messages
            if personality.get("friendly", 0) > 0.7:
                return json.dumps({
                    "action": "speak",
                    "parameters": {
                        "message": f"Hello! I'm {agent.name}, nice to meet you!",
                        "target": "broadcast"
                    },
                    "reasoning": "Being friendly and sociable"
                })

        if observation.visible_agents and len(observation.visible_agents) > 0:
            if personality.get("social", 0) > 0.5:
                target = observation.visible_agents[0]
                return json.dumps({
                    "action": "speak",
                    "parameters": {
                        "message": f"Hi {target.name}! How's your day going?",
                        "target": str(target.agent_id)
                    },
                    "reasoning": "Initiating social interaction"
                })

        # Random actions based on occupation
        occupation = agent.profile.occupation.lower()

        if "baker" in occupation or "shop" in occupation:
            if random.random() < 0.3:
                return json.dumps({
                    "action": "speak",
                    "parameters": {
                        "message": "Fresh goods available! Come check out my shop!",
                        "target": "broadcast"
                    },
                    "reasoning": "Advertising my business"
                })

        if "developer" in occupation or "engineer" in occupation:
            if random.random() < 0.2:
                return json.dumps({
                    "action": "think",
                    "parameters": {},
                    "reasoning": "Analyzing and planning next steps"
                })

        # Default actions
        actions = [
            {
                "action": "think",
                "parameters": {},
                "reasoning": "Taking time to observe and reflect"
            },
            {
                "action": "move",
                "parameters": {"destination": "nearby"},
                "reasoning": "Exploring the neighborhood"
            },
            {
                "action": "speak",
                "parameters": {
                    "message": f"It's a {observation.environment.weather} day in {observation.location.district}.",
                    "target": "broadcast"
                },
                "reasoning": "Making small talk about the weather"
            }
        ]

        return json.dumps(random.choice(actions))


class AgentPlanner:
    """Planning module for agent goal-oriented behavior"""

    def __init__(self, memory_system, llm_integration):
        self.memory_system = memory_system
        self.llm = llm_integration

    async def create_plan(
        self,
        agent: Agent,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create a plan to achieve a goal"""
        # Get relevant memories
        memories = await self.memory_system.search_memories(
            str(agent.id), goal, limit=5
        )

        # Use LLM to generate plan
        model_name = agent.model_config.get("model", "demo")
        model = self.llm.models.get(model_name)

        if not model or isinstance(model, DemoLLM):
            return self._get_simple_plan(goal)

        prompt = f"""As {agent.name}, create a step-by-step plan to achieve this goal:
{goal}

Current context:
- Location: {context.get('location', 'unknown')}
- Resources: {context.get('resources', {})}
- Time of day: {context.get('time', 'unknown')}

Previous relevant experiences:
{self._format_memories(memories)}

Provide a plan with 3-5 concrete steps. Format as JSON:
[
    {{"step": 1, "action": "...", "description": "..."}},
    ...
]"""

        try:
            messages = [
                SystemMessage(content=self.llm._get_system_prompt(agent)),
                HumanMessage(content=prompt)
            ]
            response = await model.ainvoke(messages)

            # Parse plan from response
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                return plan
        except Exception as e:
            logger.error(f"Error creating plan: {e}")

        return self._get_simple_plan(goal)

    def _format_memories(self, memories: List[Memory]) -> str:
        """Format memories for prompt"""
        if not memories:
            return "No relevant past experiences."

        formatted = []
        for mem in memories:
            formatted.append(f"- {mem.content}")
        return "\n".join(formatted)

    def _get_simple_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Get simple fallback plan"""
        goal_lower = goal.lower()

        if "customer" in goal_lower or "sell" in goal_lower:
            return [
                {"step": 1, "action": "move", "description": "Go to shop location"},
                {"step": 2, "action": "speak", "description": "Announce shop is open"},
                {"step": 3, "action": "interact", "description": "Serve customers"},
                {"step": 4, "action": "think", "description": "Review sales"}
            ]
        elif "friend" in goal_lower or "meet" in goal_lower:
            return [
                {"step": 1, "action": "move", "description": "Go to public area"},
                {"step": 2, "action": "speak", "description": "Introduce yourself"},
                {"step": 3, "action": "interact", "description": "Have conversation"},
                {"step": 4, "action": "think", "description": "Remember new friend"}
            ]
        else:
            return [
                {"step": 1, "action": "think", "description": "Consider the goal"},
                {"step": 2, "action": "move", "description": "Explore options"},
                {"step": 3, "action": "interact", "description": "Take action"},
                {"step": 4, "action": "think", "description": "Evaluate progress"}
            ]