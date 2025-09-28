# Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Monitoring Setup](#monitoring-setup)
8. [Security Hardening](#security-hardening)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

#### Minimum (Development)
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Storage**: 20 GB SSD
- **OS**: Linux, macOS, or Windows with WSL2

#### Recommended (Production)
- **CPU**: 8 cores
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **OS**: Ubuntu 22.04 LTS or RHEL 9

#### High Scale (10,000+ agents)
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **Storage**: 500 GB SSD (NVMe preferred)
- **Network**: 1 Gbps

### Software Requirements
```bash
# Required
docker >= 24.0.0
docker-compose >= 2.20.0
python >= 3.11
node >= 18.0.0

# Optional
kubectl >= 1.28 (for Kubernetes)
helm >= 3.12 (for Kubernetes)
terraform >= 1.5 (for cloud deployment)
```

---

## Local Development

### 1. Clone Repository
```bash
git clone https://github.com/your-org/multi-agent-city.git
cd multi-agent-city
```

### 2. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

Required environment variables:
```env
# Database
POSTGRES_DB=multiagent_city
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<secure-password>

# Redis
REDIS_PASSWORD=<secure-password>

# JWT
JWT_SECRET=<secure-random-string>

# LLM API Keys (optional but recommended)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_HOST=http://localhost:11434

# MQTT
MQTT_USER=admin
MQTT_PASSWORD=<secure-password>
```

### 3. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# Node dependencies (for dashboard)
cd services/web-dashboard
npm install
cd ../..
```

### 4. Initialize Database
```bash
# Start only PostgreSQL
docker-compose up -d postgres

# Run migrations
python scripts/init_db.py

# Verify database
psql -h localhost -U postgres -d multiagent_city -c "\dt"
```

### 5. Start Services
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 6. Verify Deployment
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Access dashboard
open http://localhost:8080
```

---

## Docker Deployment

### Production Docker Compose

Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    restart: always
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 2G
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 1G

  # Add other services with production settings...

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Deploy with Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml multiagent-city

# Check services
docker service ls

# Scale services
docker service scale multiagent-city_agent-scheduler=3
```

---

## Kubernetes Deployment

### 1. Create Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: multiagent-city
```

### 2. ConfigMaps and Secrets
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: multiagent-city
data:
  WORLD_ORCHESTRATOR_URL: "http://world-orchestrator:8000"
  AGENT_SCHEDULER_URL: "http://agent-scheduler:8000"
  REDIS_URL: "redis://redis:6379"
---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: multiagent-city
type: Opaque
stringData:
  POSTGRES_PASSWORD: "your-password"
  JWT_SECRET: "your-jwt-secret"
  OPENAI_API_KEY: "sk-..."
```

### 3. Deployments
```yaml
# k8s/world-orchestrator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: world-orchestrator
  namespace: multiagent-city
spec:
  replicas: 2
  selector:
    matchLabels:
      app: world-orchestrator
  template:
    metadata:
      labels:
        app: world-orchestrator
    spec:
      containers:
      - name: world-orchestrator
        image: multiagent-city/world-orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DATABASE_URL
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### 4. Services
```yaml
# k8s/services.yaml
apiVersion: v1
kind: Service
metadata:
  name: world-orchestrator
  namespace: multiagent-city
spec:
  selector:
    app: world-orchestrator
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 5. Ingress
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multiagent-city
  namespace: multiagent-city
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.multiagentcity.com
    secretName: multiagent-city-tls
  rules:
  - host: api.multiagentcity.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8000
```

### 6. Deploy with Helm
```bash
# Install Helm chart
helm install multiagent-city ./helm-chart \
  --namespace multiagent-city \
  --values ./helm-chart/values.production.yaml

# Upgrade deployment
helm upgrade multiagent-city ./helm-chart \
  --namespace multiagent-city \
  --values ./helm-chart/values.production.yaml

# Check status
kubectl get pods -n multiagent-city
kubectl get services -n multiagent-city
```

---

## Cloud Deployment

### AWS Deployment

#### Using ECS
```bash
# Build and push images to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -t world-orchestrator services/world-orchestrator
docker tag world-orchestrator:latest $ECR_REGISTRY/world-orchestrator:latest
docker push $ECR_REGISTRY/world-orchestrator:latest

# Deploy with Terraform
cd terraform/aws
terraform init
terraform plan -var-file=production.tfvars
terraform apply -var-file=production.tfvars
```

#### Using EKS
```bash
# Create EKS cluster
eksctl create cluster \
  --name multiagent-city \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5

# Deploy application
kubectl apply -f k8s/
```

### Google Cloud Deployment

#### Using GKE
```bash
# Create GKE cluster
gcloud container clusters create multiagent-city \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n2-standard-2 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials multiagent-city --zone us-central1-a

# Deploy
kubectl apply -f k8s/
```

### Azure Deployment

#### Using AKS
```bash
# Create AKS cluster
az aks create \
  --resource-group multiagent-city-rg \
  --name multiagent-city-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group multiagent-city-rg --name multiagent-city-aks

# Deploy
kubectl apply -f k8s/
```

---

## Configuration

### Environment Variables

#### Core Services
```env
# World Orchestrator
WORLD_ENGINE_TICK_RATE=60  # Seconds per tick
WORLD_MAX_AGENTS=10000
WORLD_DISTRICTS=4

# Agent Scheduler
AGENT_MAX_CONCURRENT=100
AGENT_MEMORY_LIMIT=1000  # Memories per agent
AGENT_DECISION_TIMEOUT=5000  # ms

# Economy
ECONOMY_INITIAL_SUPPLY=1000000
ECONOMY_STARTING_BALANCE=1000
ECONOMY_TAX_RATE=0.1

# Governance
GOVERNANCE_PROPOSAL_COST=100
GOVERNANCE_MIN_REPUTATION_VOTE=10
GOVERNANCE_MIN_REPUTATION_PROPOSE=100
```

### Performance Tuning

#### PostgreSQL
```sql
-- postgresql.conf
max_connections = 200
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 10485kB
min_wal_size = 2GB
max_wal_size = 8GB
```

#### Redis
```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

#### MQTT
```conf
# mosquitto.conf
max_connections 10000
max_queued_messages 1000
message_size_limit 256000
persistence true
persistence_location /mosquitto/data/
autosave_interval 300
```

---

## Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'world-orchestrator'
    static_configs:
    - targets: ['world-orchestrator:8000']
    metrics_path: '/metrics/prometheus'

  - job_name: 'agent-scheduler'
    static_configs:
    - targets: ['agent-scheduler:8000']
    metrics_path: '/metrics/prometheus'

  - job_name: 'node-exporter'
    static_configs:
    - targets: ['node-exporter:9100']
```

### Grafana Dashboards

Import dashboards:
```bash
# Download dashboard JSONs
wget https://raw.githubusercontent.com/your-org/multi-agent-city/main/grafana/dashboards/overview.json
wget https://raw.githubusercontent.com/your-org/multi-agent-city/main/grafana/dashboards/agents.json
wget https://raw.githubusercontent.com/your-org/multi-agent-city/main/grafana/dashboards/economy.json

# Import via API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @overview.json
```

### Alerts
```yaml
# alerts.yml
groups:
  - name: multiagent_city
    rules:
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"world-orchestrator.*"} > 1e+9
      for: 5m
      annotations:
        summary: "High memory usage detected"

    - alert: AgentSchedulerDown
      expr: up{job="agent-scheduler"} == 0
      for: 1m
      annotations:
        summary: "Agent scheduler is down"

    - alert: HighTransactionFailureRate
      expr: rate(economy_transaction_failures[5m]) > 0.1
      for: 5m
      annotations:
        summary: "High transaction failure rate"
```

---

## Security Hardening

### 1. Network Security
```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: multiagent-city
spec:
  podSelector: {}
  policyTypes:
  - Ingress
```

### 2. RBAC
```yaml
# RBAC configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: multiagent-city-reader
  namespace: multiagent-city
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

### 3. Secrets Management
```bash
# Use external secrets manager
kubectl create secret generic multiagent-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  --from-literal=db-password=$(openssl rand -hex 16)

# Rotate secrets
kubectl create secret generic multiagent-secrets-new \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 4. TLS Configuration
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=api.multiagentcity.com"

# Create TLS secret
kubectl create secret tls multiagent-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n multiagent-city
```

---

## Backup & Recovery

### Database Backup

#### Automated Backups
```bash
# backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# PostgreSQL backup
pg_dump -h postgres -U $POSTGRES_USER -d $POSTGRES_DB | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# Redis backup
redis-cli -h redis --rdb $BACKUP_DIR/redis_$DATE.rdb

# Upload to S3
aws s3 cp $BACKUP_DIR/postgres_$DATE.sql.gz s3://multiagent-backups/postgres/
aws s3 cp $BACKUP_DIR/redis_$DATE.rdb s3://multiagent-backups/redis/

# Clean old backups
find $BACKUP_DIR -mtime +7 -delete
```

#### Restore Process
```bash
# Restore PostgreSQL
gunzip < postgres_20240101_120000.sql.gz | psql -h postgres -U $POSTGRES_USER -d $POSTGRES_DB

# Restore Redis
redis-cli -h redis --pipe < redis_20240101_120000.rdb
```

### Disaster Recovery

#### Backup Strategy
- **Daily**: Full database backups
- **Hourly**: Incremental backups
- **Real-time**: WAL archiving for PostgreSQL
- **Retention**: 30 days local, 90 days cloud

#### Recovery Time Objectives
- **RTO**: 2 hours
- **RPO**: 1 hour

---

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check logs
docker-compose logs world-orchestrator
kubectl logs -n multiagent-city deployment/world-orchestrator

# Check resources
docker stats
kubectl top pods -n multiagent-city

# Restart services
docker-compose restart world-orchestrator
kubectl rollout restart deployment/world-orchestrator -n multiagent-city
```

#### Database Connection Issues
```bash
# Test connection
psql -h localhost -U postgres -d multiagent_city -c "SELECT 1"

# Check network
docker network ls
docker network inspect multiagent-city_default

# Reset connections
docker-compose restart postgres
```

#### High Memory Usage
```bash
# Identify memory consumers
docker stats --no-stream
kubectl top pods -n multiagent-city --sort-by=memory

# Increase limits
docker-compose up -d --scale agent-scheduler=2
kubectl scale deployment agent-scheduler --replicas=3
```

#### Performance Issues
```bash
# Check metrics
curl http://localhost:8001/metrics | jq '.performance'

# Profile services
python -m cProfile services/world-orchestrator/main.py

# Optimize queries
EXPLAIN ANALYZE SELECT * FROM agents WHERE status = 'active';
```

### Health Checks
```bash
# Service health
for port in 8000 8001 8002 8080; do
  echo "Checking port $port:"
  curl -s http://localhost:$port/health | jq '.status'
done

# Database health
docker exec postgres pg_isready

# Redis health
docker exec redis redis-cli ping

# MQTT health
docker exec mqtt-broker mosquitto_sub -t '$SYS/#' -C 1
```

### Log Analysis
```bash
# Aggregate logs
docker-compose logs --tail=100 --follow

# Search for errors
docker-compose logs | grep ERROR

# Export logs
docker-compose logs > multiagent-city.log

# Analyze with tools
cat multiagent-city.log | grep -E "ERROR|WARNING" | sort | uniq -c
```

---

## Performance Optimization

### Scaling Guidelines

| Agents | CPU Cores | RAM | Postgres | Redis | MQTT Connections |
|--------|-----------|-----|----------|-------|------------------|
| 100    | 2         | 4GB | 50 conn  | 1GB   | 200             |
| 1,000  | 4         | 8GB | 100 conn | 2GB   | 2,000           |
| 10,000 | 16        | 32GB| 200 conn | 4GB   | 20,000          |

### Optimization Tips

1. **Database**
   - Add indexes for frequently queried columns
   - Use connection pooling
   - Implement query caching

2. **Caching**
   - Cache agent states in Redis
   - Use Redis pub/sub for real-time updates
   - Implement TTL for temporary data

3. **Message Queue**
   - Batch messages when possible
   - Use QoS settings for MQTT
   - Implement message compression

4. **Application**
   - Use async/await for I/O operations
   - Implement request batching
   - Use connection pools for external services

---

## Maintenance

### Regular Tasks
- **Daily**: Check logs, monitor metrics
- **Weekly**: Update dependencies, review alerts
- **Monthly**: Performance review, capacity planning
- **Quarterly**: Security audit, disaster recovery test

### Updates
```bash
# Update images
docker-compose pull
docker-compose up -d

# Update Kubernetes deployment
kubectl set image deployment/world-orchestrator world-orchestrator=multiagent-city/world-orchestrator:v1.1.0 -n multiagent-city

# Rolling update
kubectl rollout status deployment/world-orchestrator -n multiagent-city
```

---

## Support

- **Documentation**: https://docs.multiagentcity.com
- **GitHub Issues**: https://github.com/your-org/multi-agent-city/issues
- **Discord**: https://discord.gg/multiagentcity
- **Email**: support@multiagentcity.ai