import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, SearchParams
)
from sentence_transformers import SentenceTransformer
import numpy as np

from packages.shared_types.models import Memory, MemoryType

logger = logging.getLogger(__name__)


class VectorMemoryStore:
    """Qdrant-based vector memory storage for agents"""

    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.collection_name = "agent_memories"
        self.embedding_dim = 384  # For all-MiniLM-L6-v2

    async def initialize(self):
        """Initialize Qdrant client and embedding model"""
        try:
            # Connect to Qdrant
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.client = QdrantClient(url=qdrant_url)

            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Create or update collection
            await self._ensure_collection()

            logger.info("Vector memory store initialized with Qdrant")

        except Exception as e:
            logger.error(f"Error initializing vector memory store: {e}")
            # Fall back to basic memory system if Qdrant not available
            logger.warning("Falling back to Redis-only memory storage")

    async def _ensure_collection(self):
        """Ensure the memories collection exists"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")

    async def store_memory(self, memory: Memory) -> bool:
        """Store a memory with its embedding in Qdrant"""
        if not self.client:
            return False

        try:
            # Generate embedding
            embedding = self.embedding_model.encode(memory.content).tolist()

            # Prepare point data
            point = PointStruct(
                id=str(memory.id),
                vector=embedding,
                payload={
                    "agent_id": str(memory.agent_id),
                    "type": memory.type,
                    "content": memory.content,
                    "importance": memory.importance,
                    "timestamp": memory.timestamp.isoformat(),
                    "participants": [str(p) for p in memory.participants],
                    "emotions": memory.emotions,
                    "tags": memory.tags
                }
            )

            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Stored memory {memory.id} in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error storing memory in Qdrant: {e}")
            return False

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """Search memories using semantic similarity"""
        if not self.client:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Build filter conditions
            conditions = [
                FieldCondition(
                    key="agent_id",
                    match=MatchValue(value=str(agent_id))
                )
            ]

            if memory_type:
                conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=memory_type)
                    )
                )

            if min_importance > 0:
                conditions.append(
                    FieldCondition(
                        key="importance",
                        range={"gte": min_importance}
                    )
                )

            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=conditions),
                limit=limit
            )

            # Convert results to Memory objects
            memories = []
            for hit in search_result:
                memory = self._payload_to_memory(hit.id, hit.payload)
                if memory:
                    memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Error searching memories in Qdrant: {e}")
            return []

    async def get_recent_memories(
        self,
        agent_id: str,
        limit: int = 10,
        hours: int = 24
    ) -> List[Memory]:
        """Get recent memories for an agent"""
        if not self.client:
            return []

        try:
            # Calculate timestamp threshold
            from datetime import timedelta
            threshold = datetime.utcnow() - timedelta(hours=hours)

            # Search with filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=str(agent_id))
                        ),
                        FieldCondition(
                            key="timestamp",
                            range={"gte": threshold.isoformat()}
                        )
                    ]
                ),
                limit=limit
            )

            # Convert and sort by timestamp
            memories = []
            for point in search_result[0]:
                memory = self._payload_to_memory(point.id, point.payload)
                if memory:
                    memories.append(memory)

            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Error getting recent memories from Qdrant: {e}")
            return []

    async def get_important_memories(
        self,
        agent_id: str,
        limit: int = 10,
        min_importance: float = 0.7
    ) -> List[Memory]:
        """Get important memories for an agent"""
        if not self.client:
            return []

        try:
            # Search with importance filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=str(agent_id))
                        ),
                        FieldCondition(
                            key="importance",
                            range={"gte": min_importance}
                        )
                    ]
                ),
                limit=limit
            )

            # Convert results
            memories = []
            for point in search_result[0]:
                memory = self._payload_to_memory(point.id, point.payload)
                if memory:
                    memories.append(memory)

            # Sort by importance
            memories.sort(key=lambda m: m.importance, reverse=True)
            return memories

        except Exception as e:
            logger.error(f"Error getting important memories from Qdrant: {e}")
            return []

    async def delete_memory(self, memory_id: str):
        """Delete a memory from Qdrant"""
        if not self.client:
            return

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            logger.debug(f"Deleted memory {memory_id} from Qdrant")

        except Exception as e:
            logger.error(f"Error deleting memory from Qdrant: {e}")

    async def get_memory_count(self, agent_id: str) -> int:
        """Get total memory count for an agent"""
        if not self.client:
            return 0

        try:
            # Count with filter
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=str(agent_id))
                        )
                    ]
                )
            )

            return count_result.count

        except Exception as e:
            logger.error(f"Error counting memories in Qdrant: {e}")
            return 0

    async def find_similar_memories(
        self,
        memory: Memory,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Memory]:
        """Find memories similar to a given memory"""
        if not self.client:
            return []

        try:
            # Generate embedding for the memory
            embedding = self.embedding_model.encode(memory.content).tolist()

            # Search for similar memories (excluding the same memory)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=str(memory.agent_id))
                        )
                    ],
                    must_not=[
                        FieldCondition(
                            key="id",
                            match=MatchValue(value=str(memory.id))
                        )
                    ]
                ),
                limit=limit,
                score_threshold=min_similarity
            )

            # Convert results
            similar_memories = []
            for hit in search_result:
                similar_memory = self._payload_to_memory(hit.id, hit.payload)
                if similar_memory:
                    similar_memories.append(similar_memory)

            return similar_memories

        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def _payload_to_memory(self, memory_id: str, payload: dict) -> Optional[Memory]:
        """Convert Qdrant payload to Memory object"""
        try:
            return Memory(
                id=UUID(memory_id),
                agent_id=UUID(payload["agent_id"]),
                type=MemoryType(payload["type"]),
                content=payload["content"],
                importance=float(payload["importance"]),
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                participants=[UUID(p) for p in payload.get("participants", [])],
                emotions=payload.get("emotions", []),
                tags=payload.get("tags", [])
            )

        except Exception as e:
            logger.error(f"Error converting payload to memory: {e}")
            return None

    async def consolidate_memories(
        self,
        agent_id: str,
        max_memories: int = 1000
    ):
        """Consolidate memories when limit is exceeded"""
        try:
            # Get current count
            count = await self.get_memory_count(agent_id)

            if count <= max_memories:
                return

            # Get all memories for the agent
            all_memories = await self.get_recent_memories(
                agent_id, limit=count, hours=24*365  # Get all
            )

            # Group similar memories using clustering
            clusters = await self._cluster_memories(all_memories)

            # Create summary memories for each cluster
            for cluster in clusters:
                if len(cluster) > 1:
                    summary = await self._summarize_cluster(cluster)
                    if summary:
                        await self.store_memory(summary)

                        # Delete original memories (keep most important)
                        cluster.sort(key=lambda m: m.importance, reverse=True)
                        for memory in cluster[1:]:  # Keep the most important
                            await self.delete_memory(str(memory.id))

            logger.info(f"Consolidated memories for agent {agent_id}")

        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")

    async def _cluster_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """Cluster similar memories together"""
        if not memories:
            return []

        try:
            # Generate embeddings for all memories
            contents = [m.content for m in memories]
            embeddings = self.embedding_model.encode(contents)

            # Simple clustering using cosine similarity
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_similarity

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Convert to distance matrix for DBSCAN
            distance_matrix = 1 - similarity_matrix

            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
            labels = clustering.fit_predict(distance_matrix)

            # Group memories by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Noise point (not in any cluster)
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(memories[i])

            return list(clusters.values())

        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            return [[m] for m in memories]  # Each memory as its own cluster

    async def _summarize_cluster(self, cluster: List[Memory]) -> Optional[Memory]:
        """Create a summary memory for a cluster"""
        if not cluster:
            return None

        try:
            # Simple summarization: combine content and use highest importance
            combined_content = "Summary of related experiences:\n"
            combined_content += "\n".join([f"- {m.content}" for m in cluster[:5]])

            max_importance = max(m.importance for m in cluster)
            avg_importance = sum(m.importance for m in cluster) / len(cluster)

            # Create summary memory
            summary = Memory(
                agent_id=cluster[0].agent_id,
                type=MemoryType.REFLECTION,
                content=combined_content,
                importance=min(1.0, max_importance * 1.1),  # Slightly higher
                tags=["summary", "consolidated"],
                emotions=list(set(e for m in cluster for e in m.emotions))[:5]
            )

            return summary

        except Exception as e:
            logger.error(f"Error summarizing cluster: {e}")
            return None