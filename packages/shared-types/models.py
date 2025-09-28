from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class AgentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    OFFLINE = "offline"


class ActionType(str, Enum):
    SPEAK = "speak"
    MOVE = "move"
    INTERACT = "interact"
    CRAFT = "craft"
    TRADE = "trade"
    THINK = "think"


class LocationType(str, Enum):
    BUILDING = "building"
    STREET = "street"
    PARK = "park"
    PLAZA = "plaza"


# Location Models
class Coordinates(BaseModel):
    x: float
    y: float
    z: float = 0.0


class Location(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    district: str
    neighborhood: str
    building: Optional[str] = None
    room: Optional[str] = None
    coordinates: Coordinates
    type: LocationType


# Agent Models
class AgentProfile(BaseModel):
    name: str
    occupation: str
    personality: Dict[str, float]
    background: Dict[str, str]
    goals: Dict[str, List[str]]


class AgentState(BaseModel):
    agent_id: UUID
    status: AgentStatus
    location: Location
    health: Dict[str, float]
    resources: Dict[str, Any]
    activity: Dict[str, Any]
    relationships: Dict[UUID, Dict[str, float]]


class Agent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    user_id: UUID
    profile: AgentProfile
    state: AgentState
    model_config: Dict[str, Any]
    external_endpoint: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Action Models
class ActionParameters(BaseModel):
    target: Optional[str] = None
    message: Optional[str] = None
    destination: Optional[Location] = None
    object_id: Optional[UUID] = None
    item: Optional[str] = None
    emotion: Optional[str] = None


class Action(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    type: ActionType
    parameters: ActionParameters
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reasoning: Optional[str] = None
    confidence: float = 1.0


class ActionResult(BaseModel):
    action_id: UUID
    status: str
    effects: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


# Observation Models
class VisibleAgent(BaseModel):
    agent_id: UUID
    name: str
    distance: float
    activity: Optional[str] = None
    emotion: Optional[str] = None


class AudibleMessage(BaseModel):
    speaker_id: UUID
    message: str
    volume: float
    timestamp: datetime


class Environment(BaseModel):
    time_of_day: str
    weather: str
    temperature: float
    crowd_level: str
    events: List[str] = []


class Observation(BaseModel):
    agent_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tick: int
    location: Location
    visible_agents: List[VisibleAgent]
    audible_messages: List[AudibleMessage]
    environment: Environment
    inventory: Dict[str, Any] = {}
    available_actions: List[str] = []


# Memory Models
class MemoryType(str, Enum):
    OBSERVATION = "observation"
    INTERACTION = "interaction"
    REFLECTION = "reflection"
    GOAL = "goal"
    PLAN = "plan"


class Memory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID
    type: MemoryType
    content: str
    importance: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    participants: List[UUID] = []
    emotions: List[str] = []
    tags: List[str] = []


# World Models
class WorldState(BaseModel):
    tick: int
    time: datetime
    active_agents: int
    total_events: int
    weather: str
    economy: Dict[str, Any]


class District(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    neighborhoods: List[str]
    population: int
    governance: Dict[str, Any]
    economy: Dict[str, Any]


class Property(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: str
    district: str
    neighborhood: str
    address: str
    owner_id: Optional[UUID] = None
    lease_terms: Optional[Dict[str, Any]] = None
    features: List[str] = []
    size: float
    price: float


# Transaction Models
class Transaction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    sender_id: UUID
    receiver_id: UUID
    amount: float
    currency: str = "credits"
    type: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"


# Event Models
class EventType(str, Enum):
    AGENT_SPAWN = "agent_spawn"
    AGENT_ACTION = "agent_action"
    MESSAGE = "message"
    TRANSACTION = "transaction"
    WORLD_UPDATE = "world_update"
    GOVERNANCE = "governance"


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: EventType
    agent_id: Optional[UUID] = None
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    district: Optional[str] = None
    processed: bool = False


# API Models
class AgentRegistration(BaseModel):
    name: str
    profile: AgentProfile
    model_config: Dict[str, Any]
    external_endpoint: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheck(BaseModel):
    status: str
    version: str
    uptime: float
    services: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Security Models
class ModerationResult(BaseModel):
    allowed: bool
    content: Optional[str] = None
    reason: Optional[str] = None
    score: float = 0.0
    categories: List[str] = []


class SecurityEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: str
    agent_id: UUID
    severity: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolved: bool = False