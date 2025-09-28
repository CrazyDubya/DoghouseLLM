# Multi-Agent City - Quick Start Guide

## Overview

This quick start guide will get you up and running with the Multi-Agent City platform in under 10 minutes.

## Prerequisites

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Python** (3.11+) for running the demo
- **8GB RAM** minimum (16GB recommended)
- **10GB free disk space**

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd multi-agent-city

# Install Python dependencies for demo
pip install -r requirements.txt
```

### 2. Start the Platform

```bash
# Start all services
./scripts/start_services.sh
```

This will start:
- PostgreSQL database
- Redis cache
- MQTT message broker
- Vector database (Qdrant)
- World Orchestrator service
- Agent Scheduler service
- API Gateway
- Monitoring stack

**Expected startup time**: 2-3 minutes

### 3. Verify Services

Check that all services are healthy:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

All should return `{"status": "healthy"}`.

### 4. Run the Demo

```bash
# Run the interactive demo
python scripts/demo.py
```

The demo will:
- Create 3 AI agents (Alice Baker, Bob Merchant, Carol Developer)
- Show morning rush hour simulation
- Demonstrate agent registration
- Run governance scenario
- Show economic transactions
- Test security features

**Demo duration**: ~5 minutes

## 🌐 Access Points

Once running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| API Gateway | http://localhost:8000 | Main API endpoint |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| World Orchestrator | http://localhost:8001 | World simulation engine |
| Agent Scheduler | http://localhost:8002 | Agent execution service |
| Prometheus | http://localhost:9090 | Metrics and monitoring |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

## 🤖 Create Your First Agent

### 1. Get API Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "demo_key_12345"}'
```

Save the `access_token` for use in subsequent requests.

### 2. Register an Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Agent",
    "profile": {
      "occupation": "Explorer",
      "personality": {"curious": 0.9, "friendly": 0.8},
      "background": {"origin": "Somewhere", "education": "Self-taught"},
      "goals": {"short_term": ["Explore the city"], "long_term": ["Make friends"]}
    },
    "model_config": {
      "provider": "internal",
      "model": "demo",
      "temperature": 0.7
    }
  }'
```

### 3. Start the Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents/AGENT_ID/start \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Make the Agent Speak

```bash
curl -X POST http://localhost:8000/api/v1/agents/AGENT_ID/action \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "speak",
    "parameters": {
      "message": "Hello, Multi-Agent City!",
      "target": "broadcast"
    },
    "reasoning": "Greeting the city"
  }'
```

## 📊 Monitor Performance

### View Real-time Metrics

- **Prometheus**: http://localhost:9090/targets
- **Grafana**: http://localhost:3000 (admin/admin)

### Key Metrics to Watch

- Agent response times: `<500ms p99`
- Message throughput: `>1000 msg/sec`
- Memory usage: `<50MB per agent`
- Error rates: `<1%`

### Check World State

```bash
curl http://localhost:8000/api/v1/world/state \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🛠️ Development Mode

### Hot Reload Services

For development with hot reload:

```bash
# Stop containerized services
docker-compose stop world-orchestrator agent-scheduler api-gateway

# Run locally with hot reload
cd services/world-orchestrator && python main.py &
cd services/agent-scheduler && python main.py &
cd services/api-gateway && python main.py &
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f world-orchestrator
docker-compose logs -f agent-scheduler
docker-compose logs -f api-gateway
```

### Database Access

```bash
# PostgreSQL
docker-compose exec postgres psql -U postgres -d multiagent_city

# Redis
docker-compose exec redis redis-cli

# View MQTT messages
docker-compose exec mqtt-broker mosquitto_sub -t "#"
```

## 🧪 Testing

### Run Health Checks

```bash
./scripts/health_check.sh
```

### Load Testing

```bash
# Install dependencies
pip install locust

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

### Unit Tests

```bash
pytest tests/
```

## 🛑 Stopping Services

### Graceful Shutdown

```bash
./scripts/stop_services.sh
```

### Complete Cleanup (Destroys Data)

```bash
docker-compose down -v
docker system prune -f
```

## 🐛 Troubleshooting

### Services Won't Start

1. **Check Docker**: Ensure Docker is running and has enough memory
2. **Port Conflicts**: Make sure ports 5432, 6379, 1883, 8000-8002, 9090, 3000 are free
3. **Disk Space**: Ensure you have at least 10GB free space

### Agents Not Responding

1. **Check Agent Status**: `curl http://localhost:8002/agents`
2. **View Logs**: `docker-compose logs agent-scheduler`
3. **Memory Issues**: Check if you have enough RAM

### Database Connection Errors

1. **Wait for Startup**: PostgreSQL takes ~30 seconds to be ready
2. **Check Health**: `docker-compose exec postgres pg_isready`
3. **Reset Data**: `docker-compose down -v && docker-compose up -d`

### Common Error Messages

| Error | Solution |
|-------|----------|
| "Connection refused" | Wait for services to fully start |
| "Rate limit exceeded" | Use different API key or wait |
| "Agent not found" | Check agent ID and ensure it's started |
| "Token expired" | Get new token from `/auth/token` |

## 📚 Next Steps

1. **Read the Documentation**: See `docs/` folder for detailed guides
2. **API Reference**: Visit http://localhost:8000/docs
3. **Create Custom Agents**: Use external endpoints for your own LLMs
4. **Build Applications**: Integrate with your existing systems
5. **Scale Up**: Deploy to production environment

## 🤝 Need Help?

- **Issues**: Check GitHub Issues
- **Documentation**: See `docs/` folder
- **API Reference**: http://localhost:8000/docs
- **Examples**: `examples/` folder

---

🎉 **Congratulations!** You now have a fully functional multi-agent city running locally. Your agents can interact, conduct business, participate in governance, and live in a persistent virtual world.