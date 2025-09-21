#!/bin/bash

# Multi-Agent City - Service Startup Script

set -e

echo "🚀 Starting Multi-Agent City Platform..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    exit 1
fi

# Function to wait for a service to be healthy
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=0

    echo "⏳ Waiting for $service_name to be healthy..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi

        sleep 2
        attempt=$((attempt + 1))
        echo "   Attempt $attempt/$max_attempts..."
    done

    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Start infrastructure services first
echo "🔧 Starting infrastructure services..."
docker-compose up -d postgres redis mqtt-broker qdrant

# Wait for infrastructure
echo "⏳ Waiting for infrastructure to be ready..."
sleep 10

# Check PostgreSQL
echo "🐘 Checking PostgreSQL..."
for i in {1..30}; do
    if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start"
        exit 1
    fi
    sleep 1
done

# Check Redis
echo "🔴 Checking Redis..."
for i in {1..30}; do
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Redis failed to start"
        exit 1
    fi
    sleep 1
done

# Check MQTT
echo "📡 Checking MQTT..."
for i in {1..30}; do
    if docker-compose logs mqtt-broker 2>&1 | grep -q "mosquitto version.*starting"; then
        echo "✅ MQTT broker is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ MQTT broker failed to start"
        exit 1
    fi
    sleep 1
done

# Start core services
echo "🏗️ Starting core services..."
docker-compose up -d world-orchestrator

# Wait for world orchestrator
wait_for_service "World Orchestrator" "http://localhost:8001/health"

# Start agent scheduler
echo "🤖 Starting agent scheduler..."
docker-compose up -d agent-scheduler

# Wait for agent scheduler
wait_for_service "Agent Scheduler" "http://localhost:8002/health"

# Start API gateway
echo "🌐 Starting API gateway..."
docker-compose up -d api-gateway

# Wait for API gateway
wait_for_service "API Gateway" "http://localhost:8000/health"

# Start monitoring (optional)
echo "📊 Starting monitoring services..."
docker-compose up -d prometheus grafana

echo ""
echo "🎉 Multi-Agent City Platform is now running!"
echo ""
echo "📊 Service Status:"
echo "  🌐 API Gateway:        http://localhost:8000"
echo "  🌍 World Orchestrator: http://localhost:8001"
echo "  🤖 Agent Scheduler:    http://localhost:8002"
echo "  📊 Prometheus:         http://localhost:9090"
echo "  📈 Grafana:            http://localhost:3000"
echo ""
echo "🔧 Database Access:"
echo "  🐘 PostgreSQL:         localhost:5432"
echo "  🔴 Redis:              localhost:6379"
echo "  📡 MQTT:               localhost:1883"
echo "  🧠 Qdrant:             http://localhost:6333"
echo ""
echo "🚀 Ready for agents! Run the demo with:"
echo "  python scripts/demo.py"
echo ""
echo "📚 API Documentation:"
echo "  http://localhost:8000/docs"
echo ""
echo "🛑 To stop all services:"
echo "  ./scripts/stop_services.sh"