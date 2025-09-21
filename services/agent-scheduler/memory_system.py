import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

from packages.shared_types.models import Memory, MemoryType

logger = logging.getLogger(__name__)


class MemorySystem:
    """Agent memory storage and retrieval system"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.embedding_model = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the memory system"""
        try:
            # Initialize sentence transformer for embeddings
            # Using a lightweight model for demo
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_initialized = True
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            raise

    def is_healthy(self) -> bool:
        """Check if memory system is healthy"""
        return self.is_initialized and self.embedding_model is not None

    async def store_memory(self, memory: Memory) -> str:
        """Store a memory for an agent"""
        try:
            # Generate embedding for semantic search
            embedding = self.embedding_model.encode(memory.content).tolist()

            # Store memory data
            memory_data = {
                "id": str(memory.id),
                "agent_id": str(memory.agent_id),
                "type": memory.type,
                "content": memory.content,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
                "participants": [str(p) for p in memory.participants],
                "emotions": memory.emotions,
                "tags": memory.tags,
                "embedding": json.dumps(embedding)
            }

            # Store in Redis hash
            await self.redis.hset(
                f"memory:{memory.id}",
                mapping=memory_data
            )

            # Add to agent's memory list (sorted by timestamp)
            await self.redis.zadd(
                f"agent_memories:{memory.agent_id}",
                {str(memory.id): memory.timestamp.timestamp()}
            )

            # Add to importance index
            await self.redis.zadd(
                f"agent_memories_importance:{memory.agent_id}",
                {str(memory.id): memory.importance}
            )

            # Add to type index
            await self.redis.sadd(
                f"agent_memories_type:{memory.agent_id}:{memory.type}",
                str(memory.id)
            )

            logger.debug(f"Stored memory {memory.id} for agent {memory.agent_id}")
            return str(memory.id)

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID"""
        try:
            memory_data = await self.redis.hgetall(f"memory:{memory_id}")

            if not memory_data:
                return None

            return self._dict_to_memory(memory_data)

        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """Search memories using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Get all memory IDs for the agent
            if memory_type:
                memory_ids = await self.redis.smembers(
                    f"agent_memories_type:{agent_id}:{memory_type}"
                )
            else:
                memory_ids = await self.redis.zrange(
                    f"agent_memories:{agent_id}",
                    0, -1
                )

            # Calculate similarities
            similarities = []

            for memory_id in memory_ids:
                memory_data = await self.redis.hgetall(f"memory:{memory_id}")

                if not memory_data:
                    continue

                importance = float(memory_data.get("importance", 0))
                if importance < min_importance:
                    continue

                # Get stored embedding
                embedding_str = memory_data.get("embedding", "[]")
                embedding = json.loads(embedding_str)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)

                # Combine similarity with importance and recency
                recency_score = self._calculate_recency_score(
                    memory_data.get("timestamp", "")
                )
                final_score = (
                    0.6 * similarity +
                    0.3 * importance +
                    0.1 * recency_score
                )

                similarities.append((memory_id, final_score, memory_data))

            # Sort by score and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)

            results = []
            for memory_id, score, memory_data in similarities[:limit]:
                memory = self._dict_to_memory(memory_data)
                if memory:
                    results.append(memory)

            return results

        except Exception as e:
            logger.error(f"Error searching memories for agent {agent_id}: {e}")
            return []

    async def get_recent_memories(
        self,
        agent_id: str,
        limit: int = 10,
        hours: int = 24
    ) -> List[Memory]:
        """Get recent memories for an agent"""
        try:
            # Calculate timestamp threshold
            threshold = datetime.utcnow() - timedelta(hours=hours)
            timestamp_score = threshold.timestamp()

            # Get recent memory IDs
            memory_ids = await self.redis.zrangebyscore(
                f"agent_memories:{agent_id}",
                timestamp_score,
                "+inf",
                start=0,
                num=limit,
                withscores=False
            )

            # Fetch memory data
            memories = []
            for memory_id in memory_ids:
                memory = await self.get_memory(memory_id.decode() if isinstance(memory_id, bytes) else memory_id)
                if memory:
                    memories.append(memory)

            # Sort by timestamp (most recent first)
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories

        except Exception as e:
            logger.error(f"Error getting recent memories for agent {agent_id}: {e}")
            return []

    async def get_important_memories(
        self,
        agent_id: str,
        limit: int = 10,
        min_importance: float = 0.7
    ) -> List[Memory]:
        """Get important memories for an agent"""
        try:
            # Get memory IDs by importance
            memory_ids = await self.redis.zrangebyscore(
                f"agent_memories_importance:{agent_id}",
                min_importance,
                "+inf",
                start=0,
                num=limit,
                withscores=False
            )

            # Fetch memory data
            memories = []
            for memory_id in memory_ids:
                memory = await self.get_memory(memory_id.decode() if isinstance(memory_id, bytes) else memory_id)
                if memory:
                    memories.append(memory)

            # Sort by importance (highest first)
            memories.sort(key=lambda m: m.importance, reverse=True)
            return memories

        except Exception as e:
            logger.error(f"Error getting important memories for agent {agent_id}: {e}")
            return []

    async def get_memory_count(self, agent_id: str) -> int:
        """Get total memory count for an agent"""
        try:
            count = await self.redis.zcard(f"agent_memories:{agent_id}")
            return count
        except Exception as e:
            logger.error(f"Error getting memory count for agent {agent_id}: {e}")
            return 0

    async def delete_memory(self, memory_id: str, agent_id: str):
        """Delete a memory"""
        try:
            # Remove from all indexes
            await self.redis.delete(f"memory:{memory_id}")
            await self.redis.zrem(f"agent_memories:{agent_id}", memory_id)
            await self.redis.zrem(f"agent_memories_importance:{agent_id}", memory_id)

            # Remove from type indexes (we don't know the type, so check all)
            for memory_type in MemoryType:
                await self.redis.srem(
                    f"agent_memories_type:{agent_id}:{memory_type}",
                    memory_id
                )

            logger.debug(f"Deleted memory {memory_id} for agent {agent_id}")

        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")

    async def prune_memories(
        self,
        agent_id: str,
        max_memories: int = 1000,
        min_importance: float = 0.3
    ):
        """Prune old or unimportant memories"""
        try:
            # Get total memory count
            total_count = await self.get_memory_count(agent_id)

            if total_count <= max_memories:
                return

            # Get memories to delete (oldest and least important)
            memories_to_delete = total_count - max_memories

            # Get least important memories
            low_importance_ids = await self.redis.zrangebyscore(
                f"agent_memories_importance:{agent_id}",
                0,
                min_importance,
                start=0,
                num=memories_to_delete
            )

            # Delete them
            for memory_id in low_importance_ids:
                await self.delete_memory(memory_id.decode() if isinstance(memory_id, bytes) else memory_id, agent_id)

            logger.info(f"Pruned {len(low_importance_ids)} memories for agent {agent_id}")

        except Exception as e:
            logger.error(f"Error pruning memories for agent {agent_id}: {e}")

    def _dict_to_memory(self, memory_data: dict) -> Optional[Memory]:
        """Convert Redis hash data to Memory object"""
        try:
            # Handle bytes keys from Redis
            data = {}
            for key, value in memory_data.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                data[key_str] = value_str

            return Memory(
                id=UUID(data["id"]),
                agent_id=UUID(data["agent_id"]),
                type=MemoryType(data["type"]),
                content=data["content"],
                importance=float(data["importance"]),
                timestamp=datetime.fromisoformat(data["timestamp"]),
                participants=[UUID(p) for p in json.loads(data.get("participants", "[]"))],
                emotions=json.loads(data.get("emotions", "[]")),
                tags=json.loads(data.get("tags", "[]"))
            )

        except Exception as e:
            logger.error(f"Error converting dict to memory: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np

            v1 = np.array(vec1)
            v2 = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return dot_product / (norm_v1 * norm_v2)

        except Exception:
            return 0.0

    def _calculate_recency_score(self, timestamp_str: str) -> float:
        """Calculate recency score (0-1, higher = more recent)"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            now = datetime.utcnow()
            hours_ago = (now - timestamp).total_seconds() / 3600

            # Exponential decay: score decreases as time passes
            # After 24 hours, score is ~0.5; after 168 hours (1 week), score is ~0.01
            return max(0.0, 2.0 ** (-hours_ago / 24))

        except Exception:
            return 0.0