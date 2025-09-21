"""Prometheus metrics for agent scheduler"""

import logging
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Agent metrics
total_agents_gauge = Gauge(
    'scheduler_total_agents',
    'Total number of registered agents',
    registry=registry
)

active_agents_gauge = Gauge(
    'scheduler_active_agents',
    'Number of currently active agents',
    registry=registry
)

idle_agents_gauge = Gauge(
    'scheduler_idle_agents',
    'Number of idle agents',
    registry=registry
)

agents_by_type = Gauge(
    'scheduler_agents_by_type',
    'Number of agents by type',
    ['agent_type'],
    registry=registry
)

# Task metrics
tasks_scheduled_counter = Counter(
    'scheduler_tasks_scheduled_total',
    'Total number of tasks scheduled',
    ['task_type'],
    registry=registry
)

tasks_completed_counter = Counter(
    'scheduler_tasks_completed_total',
    'Total number of tasks completed',
    ['task_type'],
    registry=registry
)

tasks_failed_counter = Counter(
    'scheduler_tasks_failed_total',
    'Total number of tasks failed',
    ['task_type', 'error_type'],
    registry=registry
)

task_queue_size = Gauge(
    'scheduler_task_queue_size',
    'Current size of task queue',
    registry=registry
)

task_execution_time = Histogram(
    'scheduler_task_execution_seconds',
    'Task execution time',
    ['task_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=registry
)

# LLM metrics
llm_requests_counter = Counter(
    'scheduler_llm_requests_total',
    'Total LLM requests',
    ['model', 'request_type'],
    registry=registry
)

llm_errors_counter = Counter(
    'scheduler_llm_errors_total',
    'Total LLM errors',
    ['model', 'error_type'],
    registry=registry
)

llm_response_time = Histogram(
    'scheduler_llm_response_seconds',
    'LLM response time',
    ['model', 'request_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

llm_token_usage = Counter(
    'scheduler_llm_tokens_total',
    'Total tokens used',
    ['model', 'token_type'],  # token_type: prompt, completion
    registry=registry
)

# Memory metrics
memories_stored_counter = Counter(
    'scheduler_memories_stored_total',
    'Total memories stored',
    ['memory_type'],
    registry=registry
)

memories_retrieved_counter = Counter(
    'scheduler_memories_retrieved_total',
    'Total memories retrieved',
    registry=registry
)

memory_search_time = Histogram(
    'scheduler_memory_search_seconds',
    'Memory search time',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    registry=registry
)

total_memories_gauge = Gauge(
    'scheduler_total_memories',
    'Total number of memories in storage',
    registry=registry
)

vector_db_operations = Counter(
    'scheduler_vector_db_operations_total',
    'Vector database operations',
    ['operation'],  # store, search, delete
    registry=registry
)

# Decision metrics
decisions_made_counter = Counter(
    'scheduler_decisions_made_total',
    'Total decisions made by agents',
    ['decision_type'],
    registry=registry
)

action_distribution = Counter(
    'scheduler_actions_taken_total',
    'Distribution of actions taken',
    ['action_type'],
    registry=registry
)

planning_time = Histogram(
    'scheduler_planning_time_seconds',
    'Time spent on planning',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

# Resource metrics
agent_cpu_usage = Gauge(
    'scheduler_agent_cpu_usage_percent',
    'CPU usage by agent processes',
    registry=registry
)

agent_memory_usage = Gauge(
    'scheduler_agent_memory_usage_bytes',
    'Memory usage by agent processes',
    registry=registry
)

redis_connections = Gauge(
    'scheduler_redis_connections',
    'Number of Redis connections',
    registry=registry
)

# Queue metrics
mqtt_messages_received = Counter(
    'scheduler_mqtt_messages_received_total',
    'Total MQTT messages received',
    ['topic'],
    registry=registry
)

mqtt_messages_sent = Counter(
    'scheduler_mqtt_messages_sent_total',
    'Total MQTT messages sent',
    ['topic'],
    registry=registry
)

mqtt_processing_time = Histogram(
    'scheduler_mqtt_processing_seconds',
    'MQTT message processing time',
    ['topic'],
    registry=registry
)


class MetricsCollector:
    """Collect and expose metrics for Prometheus"""

    def __init__(self):
        self.registry = registry

    def update_agent_metrics(self, agent_stats: Dict[str, Any]):
        """Update agent-related metrics"""
        try:
            if 'total_agents' in agent_stats:
                total_agents_gauge.set(agent_stats['total_agents'])

            if 'active_agents' in agent_stats:
                active_agents_gauge.set(agent_stats['active_agents'])

            if 'idle_agents' in agent_stats:
                idle_agents_gauge.set(agent_stats['idle_agents'])

            if 'agents_by_type' in agent_stats:
                for agent_type, count in agent_stats['agents_by_type'].items():
                    agents_by_type.labels(agent_type=agent_type).set(count)

        except Exception as e:
            logger.error(f"Error updating agent metrics: {e}")

    def record_task(self, task_type: str, status: str, duration: float = None):
        """Record task execution"""
        if status == 'scheduled':
            tasks_scheduled_counter.labels(task_type=task_type).inc()
        elif status == 'completed':
            tasks_completed_counter.labels(task_type=task_type).inc()
            if duration:
                task_execution_time.labels(task_type=task_type).observe(duration)
        elif status == 'failed':
            tasks_failed_counter.labels(task_type=task_type, error_type='unknown').inc()

    def record_llm_request(self, model: str, request_type: str, duration: float, tokens: Dict[str, int] = None):
        """Record LLM request"""
        llm_requests_counter.labels(model=model, request_type=request_type).inc()
        llm_response_time.labels(model=model, request_type=request_type).observe(duration)

        if tokens:
            if 'prompt_tokens' in tokens:
                llm_token_usage.labels(model=model, token_type='prompt').inc(tokens['prompt_tokens'])
            if 'completion_tokens' in tokens:
                llm_token_usage.labels(model=model, token_type='completion').inc(tokens['completion_tokens'])

    def record_llm_error(self, model: str, error_type: str):
        """Record LLM error"""
        llm_errors_counter.labels(model=model, error_type=error_type).inc()

    def record_memory_operation(self, operation: str, memory_type: str = None, duration: float = None):
        """Record memory operation"""
        if operation == 'store' and memory_type:
            memories_stored_counter.labels(memory_type=memory_type).inc()
        elif operation == 'retrieve':
            memories_retrieved_counter.inc()

        if operation == 'search' and duration:
            memory_search_time.observe(duration)

        vector_db_operations.labels(operation=operation).inc()

    def record_decision(self, decision_type: str, action: str = None):
        """Record agent decision"""
        decisions_made_counter.labels(decision_type=decision_type).inc()
        if action:
            action_distribution.labels(action_type=action).inc()

    def record_planning(self, duration: float):
        """Record planning duration"""
        planning_time.observe(duration)

    def update_memory_metrics(self, memory_stats: Dict[str, Any]):
        """Update memory storage metrics"""
        try:
            if 'total_memories' in memory_stats:
                total_memories_gauge.set(memory_stats['total_memories'])

        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")

    def record_mqtt_message(self, topic: str, direction: str, duration: float = None):
        """Record MQTT message"""
        if direction == 'received':
            mqtt_messages_received.labels(topic=topic).inc()
        elif direction == 'sent':
            mqtt_messages_sent.labels(topic=topic).inc()

        if duration:
            mqtt_processing_time.labels(topic=topic).observe(duration)

    def update_resource_metrics(self, cpu_percent: float = None, memory_bytes: int = None, redis_conns: int = None):
        """Update resource usage metrics"""
        try:
            if cpu_percent is not None:
                agent_cpu_usage.set(cpu_percent)

            if memory_bytes is not None:
                agent_memory_usage.set(memory_bytes)

            if redis_conns is not None:
                redis_connections.set(redis_conns)

        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")

    def update_queue_metrics(self, queue_size: int):
        """Update task queue metrics"""
        task_queue_size.set(queue_size)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics collector instance
metrics_collector = MetricsCollector()