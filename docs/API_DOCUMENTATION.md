# API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [World Management APIs](#world-management-apis)
4. [Agent APIs](#agent-apis)
5. [Economy APIs](#economy-apis)
6. [Governance APIs](#governance-apis)
7. [Social & Interaction APIs](#social--interaction-apis)
8. [Property APIs](#property-apis)
9. [Monitoring APIs](#monitoring-apis)
10. [WebSocket APIs](#websocket-apis)

---

## Overview

The Multi-Agent City Platform provides RESTful APIs for managing agents, world state, economy, governance, and social interactions.

### Base URLs
- **API Gateway**: `http://localhost:8000`
- **World Orchestrator**: `http://localhost:8001`
- **Agent Scheduler**: `http://localhost:8002`
- **Web Dashboard**: `http://localhost:8080`

### Response Format
All API responses follow this format:
```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Codes
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

---

## Authentication

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}

Response:
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

### Headers
Include the JWT token in requests:
```http
Authorization: Bearer <token>
```

---

## World Management APIs

### Get World State
```http
GET /world/state

Response:
{
  "current_tick": 12345,
  "current_time": "2024-01-01T12:00:00Z",
  "time_of_day": "afternoon",
  "districts": [...],
  "total_agents": 150,
  "active_agents": 142,
  "events_today": 23
}
```

### List Districts
```http
GET /world/districts

Response:
[
  {
    "id": "uuid",
    "name": "Downtown",
    "neighborhoods": ["Financial District", "Arts Quarter"],
    "population": 45,
    "governance": {
      "council_size": 5,
      "voting_threshold": 0.6
    },
    "economy": {
      "base_rent": 100,
      "business_tax": 0.1
    }
  }
]
```

### Create World Event
```http
POST /world/events
Content-Type: application/json

{
  "type": "festival",
  "title": "Summer Festival",
  "description": "Annual summer celebration",
  "location": {
    "district": "Downtown",
    "coordinates": {"x": 100, "y": 200}
  },
  "duration_hours": 24
}
```

### Get Agent Observation
```http
GET /world/agents/{agent_id}/observation

Response:
{
  "location": {...},
  "visible_agents": [...],
  "environment": {...},
  "audible_messages": [...]
}
```

---

## Agent APIs

### Register Agent
```http
POST /agents/{agent_id}/register
Content-Type: application/json

{
  "name": "Agent Smith",
  "persona": "Curious explorer who loves meeting new people",
  "goals": ["explore the city", "make friends", "start a business"],
  "traits": ["friendly", "ambitious", "creative"],
  "initial_location": {
    "district": "Downtown",
    "coordinates": {"x": 0, "y": 0}
  }
}
```

### Start Agent
```http
POST /agents/{agent_id}/start
```

### Stop Agent
```http
POST /agents/{agent_id}/stop
```

### Execute Agent Action
```http
POST /agents/{agent_id}/action
Content-Type: application/json

{
  "type": "move",
  "parameters": {
    "destination": {
      "district": "Market Square",
      "coordinates": {"x": 150, "y": 300}
    }
  }
}

Response:
{
  "success": true,
  "result": "Moved to Market Square",
  "new_state": {...}
}
```

### List All Agents
```http
GET /agents

Response:
[
  {
    "id": "uuid",
    "name": "Agent Smith",
    "status": "active",
    "location": {...},
    "balance": 1000,
    "reputation": 75
  }
]
```

### Search Agent Memories
```http
GET /agents/{agent_id}/memory/search?query=market&limit=10

Response:
{
  "memories": [
    {
      "id": "uuid",
      "content": "Visited the market and bought fresh produce",
      "timestamp": "2024-01-01T10:00:00Z",
      "importance": 0.7,
      "type": "observation"
    }
  ]
}
```

### Store Memory
```http
POST /agents/{agent_id}/memory/store
Content-Type: application/json

{
  "content": "Met Alice at the coffee shop",
  "type": "interaction",
  "importance": 0.8,
  "participants": ["agent_id_1", "agent_id_2"],
  "tags": ["social", "coffee", "friendship"]
}
```

### Trigger Reflection
```http
POST /agents/{agent_id}/reflect

Response:
{
  "reflection": "Today I focused on social interactions and made progress on my goal of making friends.",
  "insights": ["Need to explore more districts", "Should attend more events"]
}
```

---

## Economy APIs

### Process Transaction
```http
POST /economy/transaction
Content-Type: application/json

{
  "sender_id": "uuid",
  "receiver_id": "uuid",
  "amount": 100,
  "transaction_type": "payment",
  "metadata": {
    "description": "Coffee purchase",
    "item": "latte"
  }
}

Response:
{
  "transaction_id": "uuid",
  "status": "completed",
  "timestamp": "2024-01-01T10:00:00Z",
  "new_sender_balance": 900,
  "new_receiver_balance": 1100
}
```

### Get Agent Balance
```http
GET /economy/balance/{agent_id}

Response:
{
  "agent_id": "uuid",
  "balance": 1000,
  "wealth": 1500,
  "currency": "SimCoins"
}
```

### Get Economy Metrics
```http
GET /economy/metrics

Response:
{
  "total_supply": 1000000,
  "gdp": 50000,
  "average_balance": 1000,
  "gini_coefficient": 0.35,
  "total_transactions": 15234,
  "daily_velocity": 0.23,
  "wealth_distribution": {
    "top_1_percent": 0.15,
    "top_10_percent": 0.40,
    "bottom_50_percent": 0.20
  }
}
```

---

## Governance APIs

### Create Proposal
```http
POST /governance/proposals
Content-Type: application/json

{
  "proposer_id": "uuid",
  "proposal_type": "policy_change",
  "title": "Reduce Property Tax",
  "description": "Proposal to reduce property tax from 10% to 8%",
  "metadata": {
    "new_tax_rate": 0.08,
    "affected_districts": ["Downtown", "Tech Hub"]
  },
  "voting_duration_hours": 48
}

Response:
{
  "proposal_id": "uuid",
  "status": "draft",
  "created_at": "2024-01-01T10:00:00Z"
}
```

### Start Voting
```http
POST /governance/proposals/{proposal_id}/start-voting
```

### Cast Vote
```http
POST /governance/proposals/{proposal_id}/vote
Content-Type: application/json

{
  "voter_id": "uuid",
  "vote": "yes",
  "comment": "This will help small businesses"
}
```

### Get Active Proposals
```http
GET /governance/proposals

Response:
{
  "proposals": [
    {
      "id": "uuid",
      "title": "Reduce Property Tax",
      "type": "policy_change",
      "status": "active",
      "yes_votes": 45,
      "no_votes": 23,
      "abstain_votes": 5,
      "voting_end": "2024-01-03T10:00:00Z"
    }
  ]
}
```

### Get Voting Results
```http
GET /governance/proposals/{proposal_id}/results

Response:
{
  "yes_votes": 120,
  "no_votes": 80,
  "abstain_votes": 20,
  "total_votes": 220,
  "yes_percentage": 54.5,
  "passed": true,
  "quorum_met": true
}
```

### Get Agent Reputation
```http
GET /governance/agents/{agent_id}/reputation

Response:
{
  "agent_id": "uuid",
  "reputation": 150,
  "can_propose": true,
  "can_vote": true,
  "is_council_member": false
}
```

---

## Social & Interaction APIs

### Initiate Interaction
```http
POST /interactions/initiate
Content-Type: application/json

{
  "initiator_id": "uuid",
  "target_ids": ["uuid1", "uuid2"],
  "interaction_type": "conversation",
  "initial_message": "Hello! How are you today?",
  "location": {
    "district": "Downtown",
    "venue": "Coffee Shop"
  }
}

Response:
{
  "interaction_id": "uuid",
  "status": "active",
  "participants": ["uuid", "uuid1", "uuid2"]
}
```

### Send Message
```http
POST /interactions/{interaction_id}/message
Content-Type: application/json

{
  "sender_id": "uuid",
  "message": "I'm doing great, thanks for asking!",
  "message_type": "text"
}
```

### End Interaction
```http
POST /interactions/{interaction_id}/end
Content-Type: application/json

{
  "ender_id": "uuid",
  "reason": "conversation_complete",
  "outcomes": {
    "mood_change": 0.2,
    "relationship_formed": true
  }
}
```

### Create Relationship
```http
POST /relationships/create
Content-Type: application/json

{
  "agent1_id": "uuid",
  "agent2_id": "uuid",
  "relationship_type": "friend",
  "initial_sentiment": 0.5
}
```

### Get Agent Relationships
```http
GET /agents/{agent_id}/relationships

Response:
{
  "relationships": [
    {
      "id": "uuid",
      "other_agent_id": "uuid",
      "type": "friend",
      "strength": 0.7,
      "sentiment": 0.8,
      "trust": 0.6,
      "familiarity": 0.9,
      "last_interaction": "2024-01-01T10:00:00Z"
    }
  ]
}
```

### Get Social Network
```http
GET /agents/{agent_id}/social-network?max_distance=2

Response:
{
  "agent_id": "uuid",
  "direct_connections": 15,
  "second_degree_connections": 45,
  "social_circle": ["uuid1", "uuid2", ...],
  "recommended_connections": ["uuid3", "uuid4"]
}
```

---

## Property APIs

### Claim Property
```http
POST /properties/{property_id}/claim
Content-Type: application/json

{
  "agent_id": "uuid",
  "lease_type": "long_term"
}
```

### Transfer Property
```http
POST /properties/{property_id}/transfer
Content-Type: application/json

{
  "from_agent_id": "uuid",
  "to_agent_id": "uuid",
  "sale_price": 5000
}
```

### List Available Properties
```http
GET /properties/available?district=Downtown&type=residential

Response:
[
  {
    "id": "uuid",
    "type": "residential",
    "district": "Downtown",
    "size": 100,
    "base_value": 10000,
    "monthly_rent": 500
  }
]
```

### Get Agent Properties
```http
GET /agents/{agent_id}/properties

Response:
{
  "owned_properties": [...],
  "leased_properties": [...],
  "total_value": 50000,
  "monthly_income": 2000
}
```

---

## Monitoring APIs

### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": "operational",
    "redis": "operational",
    "mqtt": "operational",
    "world_engine": "operational"
  }
}
```

### Get Metrics (JSON)
```http
GET /metrics

Response:
{
  "world": {
    "current_tick": 12345,
    "active_agents": 150
  },
  "economy": {
    "total_transactions": 5000,
    "gdp": 100000
  },
  "performance": {
    "avg_response_time": 45,
    "requests_per_second": 120
  }
}
```

### Get Prometheus Metrics
```http
GET /metrics/prometheus

Response: (text/plain)
# HELP world_simulation_ticks_total Total number of simulation ticks
# TYPE world_simulation_ticks_total counter
world_simulation_ticks_total 12345

# HELP world_active_agents Number of active agents
# TYPE world_active_agents gauge
world_active_agents 150
```

---

## WebSocket APIs

### Real-time Updates
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

// Subscribe to updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update received:', data);
};

// Data format:
{
  "timestamp": "2024-01-01T12:00:00Z",
  "world": {
    "current_tick": 12345,
    "active_agents": 150
  },
  "economy": {
    "recent_transactions": [...]
  },
  "events": [
    {
      "type": "agent_spawned",
      "agent_id": "uuid",
      "location": {...}
    }
  ]
}
```

### Agent Stream
```javascript
// Connect to agent-specific stream
const ws = new WebSocket('ws://localhost:8002/agents/{agent_id}/stream');

// Receive observations
ws.onmessage = (event) => {
  const observation = JSON.parse(event.data);
  console.log('Agent observation:', observation);
};

// Send actions
ws.send(JSON.stringify({
  type: 'action',
  action: {
    type: 'move',
    destination: {...}
  }
}));
```

---

## Rate Limiting

All endpoints are rate-limited:
- **Default**: 100 requests per minute per IP
- **Auth endpoints**: 10 requests per minute per IP
- **Transaction endpoints**: 50 requests per minute per user
- **WebSocket connections**: 5 concurrent per user

Headers returned:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704110400
```

---

## Examples

### Complete Agent Lifecycle
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Register an agent
agent_data = {
    "name": "Explorer Bot",
    "persona": "Curious and friendly",
    "goals": ["explore", "socialize"]
}
response = requests.post(f"{BASE_URL}/agents/my-agent-id/register", json=agent_data)

# 2. Start the agent
requests.post(f"{BASE_URL}/agents/my-agent-id/start")

# 3. Get observation
observation = requests.get(f"{BASE_URL}/agents/my-agent-id/observation").json()

# 4. Execute action
action = {
    "type": "move",
    "parameters": {"destination": {"district": "Market Square"}}
}
requests.post(f"{BASE_URL}/agents/my-agent-id/action", json=action)

# 5. Interact with another agent
interaction = {
    "initiator_id": "my-agent-id",
    "target_ids": ["other-agent-id"],
    "interaction_type": "conversation",
    "initial_message": "Hello!"
}
requests.post(f"{BASE_URL}/interactions/initiate", json=interaction)
```

### Economic Transaction
```python
# Transfer money between agents
transaction = {
    "sender_id": "agent1",
    "receiver_id": "agent2",
    "amount": 100,
    "transaction_type": "payment",
    "metadata": {"reason": "goods purchase"}
}
response = requests.post(f"{BASE_URL}/economy/transaction", json=transaction)

# Check new balance
balance = requests.get(f"{BASE_URL}/economy/balance/agent1").json()
print(f"New balance: {balance['balance']} SimCoins")
```

### Governance Participation
```python
# Create a proposal
proposal = {
    "proposer_id": "agent1",
    "proposal_type": "policy_change",
    "title": "New Market Rules",
    "description": "Implement fair trading practices",
    "voting_duration_hours": 24
}
proposal_response = requests.post(f"{BASE_URL}/governance/proposals", json=proposal).json()
proposal_id = proposal_response['data']['proposal']['id']

# Start voting
requests.post(f"{BASE_URL}/governance/proposals/{proposal_id}/start-voting")

# Cast vote
vote = {
    "voter_id": "agent2",
    "vote": "yes",
    "comment": "This will improve the market"
}
requests.post(f"{BASE_URL}/governance/proposals/{proposal_id}/vote", json=vote)
```

---

## SDK Support

### Python SDK
```python
from multiagent_city import Client

client = Client(api_key="your-api-key")
agent = client.create_agent(name="My Agent", persona="Helpful assistant")
agent.move_to("Market Square")
agent.interact_with("other-agent-id", message="Hello!")
```

### JavaScript SDK
```javascript
import { MultiAgentCity } from 'multiagent-city-sdk';

const client = new MultiAgentCity({ apiKey: 'your-api-key' });
const agent = await client.createAgent({
  name: 'My Agent',
  persona: 'Helpful assistant'
});

await agent.moveTo('Market Square');
await agent.interactWith('other-agent-id', 'Hello!');
```

---

## Support

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **GitHub Issues**: https://github.com/your-org/multi-agent-city/issues
- **Discord**: https://discord.gg/multiagentcity