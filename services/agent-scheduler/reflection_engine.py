import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from packages.shared_types.models import Memory, MemoryType

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Agent reflection and insight generation system"""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    async def generate_reflection(self, agent_id: str) -> Optional[Memory]:
        """Generate a reflection for an agent based on recent experiences"""
        try:
            # Get recent memories for analysis
            recent_memories = await self.memory_system.get_recent_memories(
                agent_id, limit=50, hours=24
            )

            if len(recent_memories) < 3:
                logger.debug(f"Not enough memories for reflection for agent {agent_id}")
                return None

            # Analyze patterns and generate insights
            insights = await self._analyze_memories(recent_memories)

            if not insights:
                return None

            # Create reflection memory
            reflection_content = self._format_reflection(insights)

            reflection = Memory(
                agent_id=agent_id,
                type=MemoryType.REFLECTION,
                content=reflection_content,
                importance=0.8,  # Reflections are important
                tags=["reflection", "insight"]
            )

            # Store the reflection
            await self.memory_system.store_memory(reflection)

            logger.info(f"Generated reflection for agent {agent_id}")
            return reflection

        except Exception as e:
            logger.error(f"Error generating reflection for agent {agent_id}: {e}")
            return None

    async def _analyze_memories(self, memories: List[Memory]) -> Dict[str, any]:
        """Analyze memories to extract patterns and insights"""
        insights = {
            "interaction_patterns": [],
            "location_preferences": [],
            "activity_trends": [],
            "emotional_patterns": [],
            "goal_progress": []
        }

        try:
            # Analyze interaction patterns
            interactions = [m for m in memories if m.type == MemoryType.INTERACTION]
            if interactions:
                insights["interaction_patterns"] = self._analyze_interactions(interactions)

            # Analyze location patterns
            insights["location_preferences"] = self._analyze_locations(memories)

            # Analyze activities
            insights["activity_trends"] = self._analyze_activities(memories)

            # Analyze emotional patterns
            emotions_mentioned = []
            for memory in memories:
                emotions_mentioned.extend(memory.emotions)

            if emotions_mentioned:
                insights["emotional_patterns"] = self._analyze_emotions(emotions_mentioned)

            # Analyze goal-related activities
            goal_memories = [m for m in memories if "goal" in m.tags or "achievement" in m.tags]
            if goal_memories:
                insights["goal_progress"] = self._analyze_goal_progress(goal_memories)

            return insights

        except Exception as e:
            logger.error(f"Error analyzing memories: {e}")
            return {}

    def _analyze_interactions(self, interactions: List[Memory]) -> List[str]:
        """Analyze interaction patterns"""
        patterns = []

        # Count interactions by participant
        participant_counts = {}
        for interaction in interactions:
            for participant in interaction.participants:
                participant_id = str(participant)
                participant_counts[participant_id] = participant_counts.get(participant_id, 0) + 1

        # Identify frequent interaction partners
        if participant_counts:
            most_frequent = max(participant_counts.items(), key=lambda x: x[1])
            patterns.append(f"Most frequent interaction partner: {most_frequent[0]} ({most_frequent[1]} interactions)")

        # Analyze interaction frequency
        if len(interactions) > 5:
            patterns.append("High social activity - engaging frequently with others")
        elif len(interactions) > 2:
            patterns.append("Moderate social activity")
        else:
            patterns.append("Low social activity - may prefer solitude")

        return patterns

    def _analyze_locations(self, memories: List[Memory]) -> List[str]:
        """Analyze location preferences"""
        patterns = []

        # Extract location mentions from memory content
        location_keywords = ["downtown", "market", "park", "home", "office", "shop", "cafe", "street"]
        location_mentions = {}

        for memory in memories:
            content_lower = memory.content.lower()
            for keyword in location_keywords:
                if keyword in content_lower:
                    location_mentions[keyword] = location_mentions.get(keyword, 0) + 1

        if location_mentions:
            most_mentioned = max(location_mentions.items(), key=lambda x: x[1])
            patterns.append(f"Frequently visits: {most_mentioned[0]}")

        return patterns

    def _analyze_activities(self, memories: List[Memory]) -> List[str]:
        """Analyze activity trends"""
        patterns = []

        # Extract activity keywords from memory content
        activity_keywords = [
            "work", "business", "shop", "buy", "sell", "talk", "meet",
            "walk", "explore", "rest", "eat", "drink", "learn", "help"
        ]

        activity_counts = {}
        for memory in memories:
            content_lower = memory.content.lower()
            for keyword in activity_keywords:
                if keyword in content_lower:
                    activity_counts[keyword] = activity_counts.get(keyword, 0) + 1

        if activity_counts:
            # Find most common activities
            sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
            top_activities = sorted_activities[:3]

            for activity, count in top_activities:
                patterns.append(f"Frequently engaged in: {activity} ({count} times)")

        return patterns

    def _analyze_emotions(self, emotions: List[str]) -> List[str]:
        """Analyze emotional patterns"""
        patterns = []

        if not emotions:
            return patterns

        # Count emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Identify dominant emotions
        if emotion_counts:
            most_common = max(emotion_counts.items(), key=lambda x: x[1])
            patterns.append(f"Dominant emotion: {most_common[0]}")

            # Analyze emotional stability
            unique_emotions = len(emotion_counts)
            if unique_emotions > 5:
                patterns.append("Emotionally dynamic - experiences wide range of feelings")
            elif unique_emotions > 2:
                patterns.append("Moderate emotional range")
            else:
                patterns.append("Emotionally stable - consistent mood")

        return patterns

    def _analyze_goal_progress(self, goal_memories: List[Memory]) -> List[str]:
        """Analyze goal-related progress"""
        patterns = []

        if not goal_memories:
            return patterns

        # Look for achievement keywords
        achievement_keywords = ["completed", "achieved", "success", "finished", "accomplished"]
        challenge_keywords = ["difficult", "hard", "struggle", "challenge", "problem"]

        achievements = 0
        challenges = 0

        for memory in goal_memories:
            content_lower = memory.content.lower()

            for keyword in achievement_keywords:
                if keyword in content_lower:
                    achievements += 1
                    break

            for keyword in challenge_keywords:
                if keyword in content_lower:
                    challenges += 1
                    break

        if achievements > challenges:
            patterns.append("Making good progress toward goals")
        elif challenges > achievements:
            patterns.append("Facing challenges in goal achievement")
        else:
            patterns.append("Balanced progress with some successes and challenges")

        return patterns

    def _format_reflection(self, insights: Dict[str, any]) -> str:
        """Format insights into a readable reflection"""
        reflection_parts = ["Reflecting on recent experiences:"]

        for category, patterns in insights.items():
            if patterns:
                category_name = category.replace("_", " ").title()
                reflection_parts.append(f"\n{category_name}:")

                for pattern in patterns:
                    reflection_parts.append(f"- {pattern}")

        # Add forward-looking statements
        reflection_parts.append("\nLooking ahead:")
        reflection_parts.append("- Continue building meaningful relationships")
        reflection_parts.append("- Explore new areas and opportunities")
        reflection_parts.append("- Maintain focus on personal goals")

        return "\n".join(reflection_parts)

    async def generate_daily_summary(self, agent_id: str) -> Optional[Memory]:
        """Generate a daily summary for an agent"""
        try:
            # Get memories from the last 24 hours
            recent_memories = await self.memory_system.get_recent_memories(
                agent_id, limit=100, hours=24
            )

            if not recent_memories:
                return None

            # Count different types of activities
            interaction_count = len([m for m in recent_memories if m.type == MemoryType.INTERACTION])
            observation_count = len([m for m in recent_memories if m.type == MemoryType.OBSERVATION])

            # Create summary
            summary_content = f"""Daily Summary:
- Total memories: {len(recent_memories)}
- Interactions: {interaction_count}
- Observations: {observation_count}
- Most important event: {recent_memories[0].content if recent_memories else 'None'}
"""

            summary = Memory(
                agent_id=agent_id,
                type=MemoryType.REFLECTION,
                content=summary_content,
                importance=0.6,
                tags=["daily_summary", "reflection"]
            )

            await self.memory_system.store_memory(summary)
            return summary

        except Exception as e:
            logger.error(f"Error generating daily summary for agent {agent_id}: {e}")
            return None

    async def should_reflect(self, agent_id: str) -> bool:
        """Determine if an agent should generate a reflection"""
        try:
            # Check when last reflection was generated
            memories = await self.memory_system.search_memories(
                agent_id, "reflection", limit=1, memory_type=MemoryType.REFLECTION
            )

            if not memories:
                # No previous reflections, should reflect
                return True

            last_reflection = memories[0]
            hours_since_reflection = (datetime.utcnow() - last_reflection.timestamp).total_seconds() / 3600

            # Reflect at least once per day, or if many new memories
            if hours_since_reflection > 24:
                return True

            # Check if significant new experiences
            memory_count = await self.memory_system.get_memory_count(agent_id)
            if memory_count > 0 and memory_count % 50 == 0:  # Every 50 memories
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking reflection need for agent {agent_id}: {e}")
            return False