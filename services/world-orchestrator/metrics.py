"""Prometheus metrics for world orchestrator"""

import logging
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# World metrics
world_tick_counter = Counter(
    'world_simulation_ticks_total',
    'Total number of simulation ticks',
    registry=registry
)

active_agents_gauge = Gauge(
    'world_active_agents',
    'Number of active agents in the world',
    registry=registry
)

world_time_gauge = Gauge(
    'world_simulation_time_hours',
    'Current simulation time in hours',
    registry=registry
)

# Event metrics
events_processed_counter = Counter(
    'world_events_processed_total',
    'Total number of events processed',
    ['event_type'],
    registry=registry
)

event_processing_time = Histogram(
    'world_event_processing_seconds',
    'Time spent processing events',
    ['event_type'],
    registry=registry
)

# Agent metrics
agent_spawns_counter = Counter(
    'world_agent_spawns_total',
    'Total number of agent spawns',
    registry=registry
)

agent_moves_counter = Counter(
    'world_agent_moves_total',
    'Total number of agent movements',
    registry=registry
)

agents_by_district = Gauge(
    'world_agents_by_district',
    'Number of agents in each district',
    ['district'],
    registry=registry
)

agents_by_status = Gauge(
    'world_agents_by_status',
    'Number of agents by status',
    ['status'],
    registry=registry
)

# Economy metrics
transactions_counter = Counter(
    'economy_transactions_total',
    'Total number of economic transactions',
    ['transaction_type'],
    registry=registry
)

transaction_volume_counter = Counter(
    'economy_transaction_volume_total',
    'Total transaction volume in credits',
    ['transaction_type'],
    registry=registry
)

total_money_supply = Gauge(
    'economy_total_supply',
    'Total money supply in circulation',
    registry=registry
)

average_agent_balance = Gauge(
    'economy_average_balance',
    'Average agent balance',
    registry=registry
)

gdp_gauge = Gauge(
    'economy_gdp',
    'Economic GDP',
    registry=registry
)

# Property metrics
properties_total = Gauge(
    'property_total',
    'Total number of properties',
    registry=registry
)

properties_occupied = Gauge(
    'property_occupied',
    'Number of occupied properties',
    registry=registry
)

properties_available = Gauge(
    'property_available',
    'Number of available properties',
    registry=registry
)

property_claims_counter = Counter(
    'property_claims_total',
    'Total number of property claims',
    registry=registry
)

property_releases_counter = Counter(
    'property_releases_total',
    'Total number of property releases',
    registry=registry
)

property_tax_collected = Counter(
    'property_tax_collected_total',
    'Total property tax collected',
    registry=registry
)

# Performance metrics
tick_duration = Histogram(
    'world_tick_duration_seconds',
    'Duration of each simulation tick',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

database_query_time = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    registry=registry
)

redis_operation_time = Histogram(
    'redis_operation_duration_seconds',
    'Redis operation duration',
    ['operation'],
    registry=registry
)

mqtt_message_counter = Counter(
    'mqtt_messages_total',
    'Total MQTT messages sent',
    ['topic'],
    registry=registry
)

# System metrics
memory_usage_gauge = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

cpu_usage_gauge = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)


class MetricsCollector:
    """Collect and expose metrics for Prometheus"""

    def __init__(self):
        self.registry = registry

    def update_world_metrics(self, world_state: Dict[str, Any]):
        """Update world-related metrics"""
        try:
            if 'active_agents' in world_state:
                active_agents_gauge.set(world_state['active_agents'])

            if 'current_tick' in world_state:
                world_tick_counter.inc()

            if 'simulation_time' in world_state:
                world_time_gauge.set(world_state['simulation_time'])

            # Update district metrics
            if 'agents_by_district' in world_state:
                for district, count in world_state['agents_by_district'].items():
                    agents_by_district.labels(district=district).set(count)

            # Update status metrics
            if 'agents_by_status' in world_state:
                for status, count in world_state['agents_by_status'].items():
                    agents_by_status.labels(status=status).set(count)

        except Exception as e:
            logger.error(f"Error updating world metrics: {e}")

    def update_economy_metrics(self, economy_metrics: Dict[str, Any]):
        """Update economy-related metrics"""
        try:
            if 'total_supply' in economy_metrics:
                total_money_supply.set(economy_metrics['total_supply'])

            if 'average_balance' in economy_metrics:
                average_agent_balance.set(economy_metrics['average_balance'])

            if 'gdp' in economy_metrics:
                gdp_gauge.set(economy_metrics['gdp'])

        except Exception as e:
            logger.error(f"Error updating economy metrics: {e}")

    def update_property_metrics(self, property_metrics: Dict[str, Any]):
        """Update property-related metrics"""
        try:
            if 'total_properties' in property_metrics:
                properties_total.set(property_metrics['total_properties'])

            if 'occupied_properties' in property_metrics:
                properties_occupied.set(property_metrics['occupied_properties'])

            if 'available_properties' in property_metrics:
                properties_available.set(property_metrics['available_properties'])

        except Exception as e:
            logger.error(f"Error updating property metrics: {e}")

    def record_event(self, event_type: str, duration: float = None):
        """Record an event"""
        events_processed_counter.labels(event_type=event_type).inc()
        if duration:
            event_processing_time.labels(event_type=event_type).observe(duration)

    def record_transaction(self, transaction_type: str, amount: float):
        """Record a transaction"""
        transactions_counter.labels(transaction_type=transaction_type).inc()
        transaction_volume_counter.labels(transaction_type=transaction_type).inc(amount)

    def record_agent_spawn(self):
        """Record an agent spawn"""
        agent_spawns_counter.inc()

    def record_agent_move(self):
        """Record an agent movement"""
        agent_moves_counter.inc()

    def record_property_claim(self):
        """Record a property claim"""
        property_claims_counter.inc()

    def record_property_release(self):
        """Record a property release"""
        property_releases_counter.inc()

    def record_property_tax(self, amount: float):
        """Record property tax collection"""
        property_tax_collected.inc(amount)

    def record_tick_duration(self, duration: float):
        """Record simulation tick duration"""
        tick_duration.observe(duration)

    def record_database_query(self, query_type: str, duration: float):
        """Record database query duration"""
        database_query_time.labels(query_type=query_type).observe(duration)

    def record_redis_operation(self, operation: str, duration: float):
        """Record Redis operation duration"""
        redis_operation_time.labels(operation=operation).observe(duration)

    def record_mqtt_message(self, topic: str):
        """Record MQTT message"""
        mqtt_message_counter.labels(topic=topic).inc()

    def update_system_metrics(self, memory_bytes: int = None, cpu_percent: float = None):
        """Update system resource metrics"""
        try:
            if memory_bytes is not None:
                memory_usage_gauge.set(memory_bytes)

            if cpu_percent is not None:
                cpu_usage_gauge.set(cpu_percent)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics collector instance
metrics_collector = MetricsCollector()