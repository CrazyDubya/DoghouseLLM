#!/bin/bash

# Multi-Agent City - Service Shutdown Script

set -e

echo "🛑 Stopping Multi-Agent City Platform..."

# Stop all services gracefully
echo "📊 Stopping monitoring services..."
docker-compose stop grafana prometheus

echo "🌐 Stopping API gateway..."
docker-compose stop api-gateway

echo "🤖 Stopping agent scheduler..."
docker-compose stop agent-scheduler

echo "🌍 Stopping world orchestrator..."
docker-compose stop world-orchestrator

echo "🔧 Stopping infrastructure services..."
docker-compose stop mqtt-broker qdrant redis postgres

echo "🧹 Cleaning up containers..."
docker-compose down

echo ""
echo "✅ All services stopped successfully!"
echo ""
echo "💾 Data persisted in Docker volumes:"
echo "  🐘 PostgreSQL data"
echo "  🔴 Redis data"
echo "  📡 MQTT data"
echo "  🧠 Qdrant data"
echo ""
echo "🔄 To start again:"
echo "  ./scripts/start_services.sh"
echo ""
echo "🗑️ To remove all data (DESTRUCTIVE):"
echo "  docker-compose down -v"