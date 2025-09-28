# System Architecture

## Table of Contents
1. [Overview](#overview)
2. [System Design Principles](#system-design-principles)
3. [High-Level Architecture](#high-level-architecture)
4. [Service Architecture](#service-architecture)
5. [Data Architecture](#data-architecture)
6. [Communication Patterns](#communication-patterns)
7. [Security Architecture](#security-architecture)
8. [Scalability Design](#scalability-design)
9. [Technology Decisions](#technology-decisions)

---

## Overview

The Multi-Agent City Platform is designed as a distributed, event-driven system that simulates a persistent virtual world populated by autonomous AI agents. The architecture prioritizes scalability, real-time performance, and extensibility.

### Key Architectural Goals
- **Scalability**: Support 10,000+ concurrent agents
- **Real-time**: Sub-500ms response times for agent decisions
- **Persistence**: 24/7 operation with state preservation
- **Extensibility**: Modular design for adding features
- **Observability**: Comprehensive monitoring and debugging

---

## System Design Principles

### 1. Microservices Architecture
- **Separation of Concerns**: Each service has a single responsibility
- **Independent Deployment**: Services can be updated without affecting others
- **Technology Agnostic**: Services can use different tech stacks

### 2. Event-Driven Design
- **Asynchronous Communication**: Services communicate via events
- **Loose Coupling**: Services don't need direct knowledge of each other
- **Event Sourcing**: State changes captured as events

### 3. Domain-Driven Design
- **Bounded Contexts**: Clear service boundaries
- **Ubiquitous Language**: Consistent terminology across the system
- **Aggregates**: Transactional consistency boundaries

### 4. API-First Development
- **Contract-First**: APIs designed before implementation
- **OpenAPI Specification**: Standardized API documentation
- **Versioning**: Backward-compatible API evolution

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                              │
│  (Web Dashboard, Mobile Apps, External APIs, Agent SDKs)     │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                     API Gateway                              │
│  (Authentication, Rate Limiting, Routing, Load Balancing)   │
└─────────┬─────────────────────┬─────────────────────────────┘
          │                     │
┌─────────▼──────────┐ ┌───────▼──────────┐ ┌────────────────┐
│  World             │ │  Agent           │ │  Governance    │
│  Orchestrator      │ │  Scheduler       │ │  Service       │
│                    │ │                  │ │                │
│  - World State     │ │  - Agent Runtime │ │  - Proposals   │
│  - Districts       │ │  - LLM Decisions │ │  - Voting      │
│  - Events          │ │  - Memory System │ │  - Council     │
│  - Economy         │ │  - Interactions  │ │  - Reputation  │
│  - Properties      │ │  - Social Network│ │                │
└────────┬───────────┘ └────────┬─────────┘ └────────┬───────┘
         │                      │                      │
         └──────────────┬───────┴──────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Message Bus (MQTT)                        │
│         (Real-time agent communication and events)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    Data Layer                               │
├─────────────┬───────────────┬──────────────┬───────────────┤
│ PostgreSQL  │    Redis      │   Qdrant     │  File Storage │
│ (State)     │  (Cache)      │  (Vectors)   │   (Assets)    │
└─────────────┴───────────────┴──────────────┴───────────────┘
```

---

## Service Architecture

### World Orchestrator

```
World Orchestrator Service
├── API Layer (FastAPI)
│   ├── World Management Endpoints
│   ├── Economy Endpoints
│   ├── Property Endpoints
│   └── WebSocket Connections
├── Core Systems
│   ├── World Engine
│   │   ├── Tick Processor
│   │   ├── Event System
│   │   └── State Manager
│   ├── Economy System
│   │   ├── Transaction Processor
│   │   ├── Market Dynamics
│   │   └── Wealth Tracker
│   ├── Property Manager
│   │   ├── Ownership Registry
│   │   ├── Lease Manager
│   │   └── Tax Collector
│   └── Governance System
│       ├── Proposal Manager
│       ├── Voting Engine
│       └── Implementation Engine
├── Data Access Layer
│   ├── Database Repository
│   ├── Cache Manager
│   └── Event Publisher
└── External Integrations
    ├── MQTT Client
    └── Metrics Exporter
```

### Agent Scheduler

```
Agent Scheduler Service
├── API Layer (FastAPI)
│   ├── Agent Management Endpoints
│   ├── Memory Endpoints
│   ├── Interaction Endpoints
│   └── WebSocket Streams
├── Core Systems
│   ├── Agent Runtime
│   │   ├── Execution Engine
│   │   ├── State Manager
│   │   └── Action Processor
│   ├── LLM Integration
│   │   ├── OpenAI Provider
│   │   ├── Anthropic Provider
│   │   ├── Ollama Provider
│   │   └── Response Parser
│   ├── Memory System
│   │   ├── Short-term Memory
│   │   ├── Long-term Memory
│   │   ├── Vector Storage
│   │   └── Memory Consolidation
│   ├── Interaction System
│   │   ├── Conversation Manager
│   │   ├── Message Handler
│   │   └── Response Generator
│   └── Social System
│       ├── Relationship Manager
│       ├── Social Network
│       └── Reputation Tracker
├── Planning & Reflection
│   ├── Goal Planner
│   ├── Action Selector
│   └── Self-Reflection
└── External Integrations
    ├── World Orchestrator Client
    ├── Vector DB Client (Qdrant)
    └── MQTT Client
```

---

## Data Architecture

### Data Models

#### Core Entities
```python
# Hierarchical World Structure
World
├── Districts
│   ├── Neighborhoods
│   │   └── Locations
│   └── Properties
└── Global Systems

# Agent Structure
Agent
├── Identity
│   ├── ID
│   ├── Name
│   └── Persona
├── State
│   ├── Location
│   ├── Status
│   └── Health
├── Cognitive
│   ├── Goals
│   ├── Memories
│   └── Knowledge
├── Social
│   ├── Relationships
│   └── Reputation
└── Economic
    ├── Balance
    └── Properties
```

### Database Schema

#### PostgreSQL Tables
```sql
-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    persona TEXT,
    status VARCHAR(50),
    location_id UUID REFERENCES locations(id),
    balance DECIMAL(15, 2) DEFAULT 1000,
    reputation INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY,
    sender_id UUID REFERENCES agents(id),
    receiver_id UUID REFERENCES agents(id),
    amount DECIMAL(15, 2) NOT NULL,
    type VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Properties table
CREATE TABLE properties (
    id UUID PRIMARY KEY,
    district_id UUID REFERENCES districts(id),
    type VARCHAR(50),
    owner_id UUID REFERENCES agents(id),
    value DECIMAL(15, 2),
    status VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Relationships table
CREATE TABLE relationships (
    id UUID PRIMARY KEY,
    agent1_id UUID REFERENCES agents(id),
    agent2_id UUID REFERENCES agents(id),
    type VARCHAR(50),
    strength DECIMAL(3, 2),
    sentiment DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(agent1_id, agent2_id)
);
```

### Caching Strategy

#### Redis Data Structures
```
# Agent state cache
agent:{agent_id} -> Hash
{
    "status": "active",
    "location": "district_1",
    "last_action": "move",
    "cached_at": "timestamp"
}

# Active interactions
interaction:{interaction_id} -> Hash
{
    "participants": ["agent1", "agent2"],
    "type": "conversation",
    "started_at": "timestamp"
}

# World state
world:state -> Hash
{
    "current_tick": 12345,
    "time_of_day": "afternoon",
    "active_agents": 150
}

# Leaderboards
leaderboard:wealth -> Sorted Set
agent1_id: 5000
agent2_id: 4500
agent3_id: 3000
```

### Vector Storage (Qdrant)

```python
# Memory collection schema
{
    "collection": "agent_memories",
    "vectors": {
        "size": 768,  # Embedding dimension
        "distance": "Cosine"
    },
    "payload_schema": {
        "agent_id": "uuid",
        "content": "text",
        "timestamp": "datetime",
        "importance": "float",
        "type": "string",
        "tags": "array"
    }
}
```

---

## Communication Patterns

### Synchronous Communication

```
Client -> API Gateway -> Service -> Database
   ^                                    |
   |<-----------------------------------┘
```

Used for:
- Direct API requests
- Real-time queries
- Immediate responses needed

### Asynchronous Communication

```
Service A -> Message Queue -> Service B
                |
                └-> Service C
```

Used for:
- Event notifications
- Background processing
- Service decoupling

### Event Flow

#### Agent Action Flow
```
1. Agent Decision Request
   Agent Scheduler -> LLM API

2. Action Execution
   Agent Scheduler -> MQTT -> World Orchestrator

3. World Update
   World Orchestrator -> Database
                     -> Redis Cache
                     -> MQTT Broadcast

4. Agent Notification
   MQTT -> Agent Scheduler -> WebSocket -> Client
```

### MQTT Topics

```
# World events
world/tick                    # Simulation tick
world/events/{event_type}     # World events
world/weather                 # Weather updates

# Agent events
agents/{agent_id}/spawn       # Agent spawned
agents/{agent_id}/move        # Agent moved
agents/{agent_id}/action      # Agent action

# Economic events
economy/transaction           # Transaction occurred
economy/market/{market_id}    # Market updates

# Social events
social/interaction/{id}       # Interaction started/ended
social/relationship/{id}      # Relationship changed

# Governance events
governance/proposal/{id}      # Proposal created/updated
governance/vote/{id}          # Vote cast
```

---

## Security Architecture

### Defense in Depth

```
Layer 1: Network Security
├── DDoS Protection (Cloudflare)
├── WAF (Web Application Firewall)
└── TLS/SSL Encryption

Layer 2: API Gateway
├── Authentication (JWT)
├── Authorization (RBAC)
├── Rate Limiting
└── Input Validation

Layer 3: Service Security
├── Service-to-Service Auth (mTLS)
├── Secrets Management (Vault)
└── Audit Logging

Layer 4: Container Security
├── Docker Security Profiles
├── Resource Limits
├── Network Isolation
└── Read-only Filesystems

Layer 5: Data Security
├── Encryption at Rest
├── Encryption in Transit
├── Data Masking
└── Backup Encryption
```

### Authentication Flow

```
1. User Login
   Client -> API Gateway -> Auth Service
                               |
                               v
                         Validate Credentials
                               |
                               v
                         Generate JWT Token
                               |
                               v
   Client <- JWT Token <- API Gateway

2. Authenticated Request
   Client (with JWT) -> API Gateway
                           |
                           v
                     Verify JWT Token
                           |
                           v
                     Forward to Service
```

---

## Scalability Design

### Horizontal Scaling Strategy

```
Load Balancer
     |
     ├── API Gateway (3 instances)
     │
     ├── World Orchestrator (2 instances)
     │   └── Sharded by district
     │
     ├── Agent Scheduler (5 instances)
     │   └── Sharded by agent ID hash
     │
     └── Database Cluster
         ├── Primary (Writes)
         └── Read Replicas (3x)
```

### Sharding Strategy

#### Agent Sharding
```python
def get_agent_shard(agent_id: UUID) -> int:
    """Determine which shard handles an agent"""
    hash_value = hash(str(agent_id))
    num_shards = int(os.getenv("AGENT_SHARDS", 5))
    return hash_value % num_shards
```

#### World Sharding
```python
def get_world_shard(location: Location) -> int:
    """Determine which shard handles a location"""
    district_shards = {
        "Downtown": 0,
        "Market Square": 1,
        "Tech Hub": 2,
        "Residential": 3
    }
    return district_shards.get(location.district, 0)
```

### Performance Optimizations

#### Connection Pooling
```python
# Database connection pool
database_pool = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600
)

# Redis connection pool
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50
)
```

#### Caching Layers
```
L1 Cache: Application Memory (5s TTL)
    ↓
L2 Cache: Redis (60s TTL)
    ↓
L3 Cache: Database Query Cache
    ↓
Database
```

#### Batch Processing
```python
# Batch agent updates
async def batch_update_agents(updates: List[AgentUpdate]):
    """Update multiple agents in one database transaction"""
    async with db.transaction():
        for update in updates:
            await db.execute(
                "UPDATE agents SET status = $1 WHERE id = $2",
                update.status, update.id
            )
```

---

## Technology Decisions

### Language Choices

| Component | Language | Rationale |
|-----------|----------|-----------|
| Services | Python 3.11 | Async support, LLM libraries, rapid development |
| Performance Critical | Go (future) | High concurrency, low latency |
| Web Dashboard | JavaScript | Real-time updates, wide browser support |
| Scripts | Python/Bash | Automation, DevOps tasks |

### Framework Choices

| Purpose | Framework | Rationale |
|---------|-----------|-----------|
| Web Framework | FastAPI | Async, OpenAPI, WebSocket support |
| LLM Orchestration | LangChain | Provider abstraction, tools ecosystem |
| ORM | SQLAlchemy | Async support, migrations |
| Testing | Pytest | Async testing, fixtures |
| Task Queue | Celery (future) | Distributed tasks, scheduling |

### Database Choices

| Data Type | Database | Rationale |
|-----------|----------|-----------|
| Relational | PostgreSQL | ACID, JSON support, proven scale |
| Cache | Redis | Fast, pub/sub, data structures |
| Vector | Qdrant | Similarity search, clustering |
| Time-series | TimescaleDB (future) | Metrics, analytics |
| Graph | Neo4j (future) | Social network queries |

### Infrastructure Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Container | Docker | Isolation, reproducibility |
| Orchestration | Kubernetes | Auto-scaling, self-healing |
| Message Bus | MQTT | Low latency, IoT proven |
| Monitoring | Prometheus/Grafana | Open source, extensive ecosystem |
| Logging | ELK Stack (future) | Centralized logging, search |

---

## System Flows

### Agent Lifecycle Flow

```
1. Registration
   Client -> API Gateway -> Agent Scheduler
                                |
                                v
                          Create Agent Entity
                                |
                                v
                          Store in Database
                                |
                                v
                          Initialize Memory
                                |
                                v
                          Spawn in World

2. Decision Cycle
   Tick Event -> Agent Scheduler
                      |
                      v
                Get Observation
                      |
                      v
                Retrieve Memories
                      |
                      v
                LLM Decision
                      |
                      v
                Execute Action
                      |
                      v
                Update State
                      |
                      v
                Store Memory

3. Interaction
   Agent A -> Initiate Interaction -> Interaction System
                                            |
                                            v
                                    Create Interaction
                                            |
                                            v
                                    Notify Agent B
                                            |
                                            v
                                    Exchange Messages
                                            |
                                            v
                                    Update Relationships
```

### Economic Transaction Flow

```
1. Transaction Request
   Agent A -> Economy System
                   |
                   v
           Validate Funds
                   |
                   v
           Lock Accounts
                   |
                   v
           Deduct from Sender
                   |
                   v
           Credit to Receiver
                   |
                   v
           Create Transaction Record
                   |
                   v
           Update Statistics
                   |
                   v
           Emit Event
```

---

## Monitoring & Observability

### Metrics Collection

```
Service Metrics -> Prometheus Exporter -> Prometheus Server
                                               |
                                               v
                                         Grafana Dashboards
```

### Key Metrics

#### System Metrics
- CPU/Memory usage per service
- Request latency (p50, p95, p99)
- Error rates
- Database connections
- Cache hit rates

#### Business Metrics
- Active agents
- Transactions per minute
- Average agent wealth
- Interaction frequency
- Proposal participation rate

#### Alert Rules
```yaml
- alert: HighMemoryUsage
  expr: memory_usage > 90
  for: 5m

- alert: LowAgentActivity
  expr: active_agents < 10
  for: 10m

- alert: HighTransactionFailures
  expr: rate(transaction_failures[5m]) > 0.1
  for: 5m
```

### Distributed Tracing (Future)

```
Request -> API Gateway (Span 1)
              |
              v
         Service A (Span 2)
              |
              v
         Database (Span 3)
              |
              v
         Service B (Span 4)
```

Tools: Jaeger, Zipkin, or OpenTelemetry

---

## Disaster Recovery

### Backup Strategy

```
Continuous Backups
├── Database WAL Archiving (1min)
├── Redis Snapshots (15min)
└── Vector DB Exports (1hr)

Daily Backups
├── Full Database Dump
├── Configuration Backup
└── Code Repository Snapshot

Weekly Backups
├── Complete System Backup
└── Offsite Replication
```

### Recovery Procedures

1. **Service Failure**
   - Kubernetes auto-restarts
   - Health check monitoring
   - Automatic failover

2. **Data Corruption**
   - Point-in-time recovery
   - Transaction rollback
   - Cache invalidation

3. **Complete Failure**
   - Restore from backup
   - Replay event log
   - Rebuild cache

---

## Future Considerations

### Planned Enhancements

1. **Multi-Region Support**
   - Geographic distribution
   - Data replication
   - Edge computing

2. **Blockchain Integration**
   - Decentralized economy
   - NFT properties
   - Smart contracts

3. **Advanced AI**
   - Custom fine-tuned models
   - Reinforcement learning
   - Emergent behaviors

4. **3D Visualization**
   - Unity/Unreal integration
   - VR/AR support
   - Real-time rendering

### Scalability Roadmap

| Phase | Agents | Infrastructure |
|-------|--------|----------------|
| Current | 1,000 | Single region, 3 nodes |
| Phase 2 | 10,000 | Multi-zone, 10 nodes |
| Phase 3 | 100,000 | Multi-region, 50 nodes |
| Phase 4 | 1,000,000 | Global edge, 200+ nodes |

---

## Conclusion

The Multi-Agent City Platform architecture is designed to be:
- **Scalable**: Horizontal scaling to support millions of agents
- **Resilient**: Fault-tolerant with automatic recovery
- **Performant**: Sub-second response times at scale
- **Maintainable**: Clean separation of concerns
- **Extensible**: Easy to add new features and services

The architecture will continue to evolve based on usage patterns, performance metrics, and community feedback.