import asyncio
import json
import logging
from typing import Callable, Dict, Optional

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:
    """Async MQTT client wrapper for world communication"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None
        self.connected = False
        self.message_handlers: Dict[str, Callable] = {}
        self.publish_queue = asyncio.Queue()

    async def connect(self):
        """Connect to MQTT broker"""
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Connect to broker
        try:
            self.client.connect(self.host, self.port, 60)
            self.client.loop_start()

            # Wait for connection
            for _ in range(50):  # 5 second timeout
                if self.connected:
                    break
                await asyncio.sleep(0.1)

            if not self.connected:
                raise ConnectionError("Failed to connect to MQTT broker")

            logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")

            # Start publish worker
            asyncio.create_task(self._publish_worker())

        except Exception as e:
            logger.error(f"Error connecting to MQTT broker: {e}")
            raise

    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response"""
        if rc == 0:
            self.connected = True
            logger.info("MQTT client connected successfully")

            # Subscribe to essential topics
            self.client.subscribe("world/+")
            self.client.subscribe("district/+/+")
            self.client.subscribe("agent/+/+")
        else:
            logger.error(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects"""
        self.connected = False
        logger.warning(f"MQTT client disconnected with code {rc}")

    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')

            # Handle message based on topic pattern
            if topic in self.message_handlers:
                self.message_handlers[topic](topic, payload)
            else:
                # Try pattern matching
                for pattern, handler in self.message_handlers.items():
                    if self._match_topic(pattern, topic):
                        handler(topic, payload)
                        break

        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")

    def _match_topic(self, pattern: str, topic: str) -> bool:
        """Simple topic pattern matching"""
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')

        if len(pattern_parts) != len(topic_parts):
            return False

        for p, t in zip(pattern_parts, topic_parts):
            if p != '+' and p != '#' and p != t:
                return False

        return True

    async def publish(self, topic: str, payload: str, qos: int = 0):
        """Publish a message"""
        await self.publish_queue.put((topic, payload, qos))

    async def _publish_worker(self):
        """Background worker to publish messages"""
        while True:
            try:
                topic, payload, qos = await self.publish_queue.get()

                if self.connected:
                    result = self.client.publish(topic, payload, qos)
                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        logger.error(f"Failed to publish to {topic}: {result.rc}")
                else:
                    logger.warning(f"Cannot publish to {topic}: MQTT not connected")

            except Exception as e:
                logger.error(f"Error in publish worker: {e}")

    def subscribe(self, topic: str, handler: Callable[[str, str], None]):
        """Subscribe to a topic with a handler"""
        self.message_handlers[topic] = handler
        if self.connected:
            self.client.subscribe(topic)

    def unsubscribe(self, topic: str):
        """Unsubscribe from a topic"""
        if topic in self.message_handlers:
            del self.message_handlers[topic]
        if self.connected:
            self.client.unsubscribe(topic)

    async def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info("MQTT client disconnected")

    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.connected

    async def broadcast_to_district(self, district: str, message_type: str, data: dict):
        """Broadcast a message to all agents in a district"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.publish(f"district/{district}/broadcast", json.dumps(message))

    async def send_to_agent(self, agent_id: str, message_type: str, data: dict):
        """Send a direct message to an agent"""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.publish(f"agent/{agent_id}/direct", json.dumps(message))

    async def publish_world_event(self, event_type: str, data: dict):
        """Publish a world-wide event"""
        message = {
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        await self.publish("world/events", json.dumps(message))


class MQTTMessageHandler:
    """Handler for processing MQTT messages in the world orchestrator"""

    def __init__(self, world_engine):
        self.world_engine = world_engine

    def handle_agent_message(self, topic: str, payload: str):
        """Handle messages from agents"""
        try:
            data = json.loads(payload)
            message_type = data.get("type")

            if message_type == "action":
                asyncio.create_task(self._handle_agent_action(data))
            elif message_type == "heartbeat":
                asyncio.create_task(self._handle_agent_heartbeat(data))

        except Exception as e:
            logger.error(f"Error handling agent message: {e}")

    async def _handle_agent_action(self, data: dict):
        """Handle agent action message"""
        from packages.shared_types.models import Event, EventType

        agent_id = data.get("agent_id")
        action_data = data.get("action")

        event = Event(
            type=EventType.AGENT_ACTION,
            agent_id=agent_id,
            data=action_data
        )

        await self.world_engine.process_event(event)

    async def _handle_agent_heartbeat(self, data: dict):
        """Handle agent heartbeat message"""
        agent_id = data.get("agent_id")
        status = data.get("status", "active")

        # Update agent status in world engine
        if agent_id and agent_id in self.world_engine.agents:
            self.world_engine.agents[agent_id].state.status = status

    def handle_system_message(self, topic: str, payload: str):
        """Handle system messages"""
        try:
            data = json.loads(payload)
            message_type = data.get("type")

            if message_type == "shutdown":
                logger.info("Received shutdown signal")
                self.world_engine.stop()

        except Exception as e:
            logger.error(f"Error handling system message: {e}")