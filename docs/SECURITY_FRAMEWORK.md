# Security & Isolation Framework

## Executive Summary

This document defines the comprehensive security architecture for the multi-agent city platform, ensuring safe operation of thousands of user-controlled AI agents. The framework implements defense-in-depth with multiple isolation layers.

## Threat Model

### Primary Threats

1. **Malicious Agent Code**
   - Risk: Code injection, system compromise
   - Impact: High
   - Mitigation: Sandboxed execution, code scanning

2. **Content Policy Violations**
   - Risk: Harmful, toxic, or illegal content
   - Impact: High
   - Mitigation: Real-time moderation, content filtering

3. **Resource Exhaustion**
   - Risk: DoS through excessive resource consumption
   - Impact: Medium
   - Mitigation: Resource quotas, rate limiting

4. **Data Leakage**
   - Risk: Cross-agent information exposure
   - Impact: High
   - Mitigation: Memory isolation, access controls

5. **Economic Exploitation**
   - Risk: Currency manipulation, fraud
   - Impact: Medium
   - Mitigation: Transaction limits, audit trails

6. **Identity Spoofing**
   - Risk: Agent impersonation
   - Impact: Medium
   - Mitigation: Cryptographic authentication

## Security Architecture Layers

### Layer 1: Network Perimeter

```yaml
Network Security:
  WAF:
    provider: Cloudflare/AWS WAF
    rules:
      - OWASP Top 10 protection
      - SQL injection prevention
      - XSS filtering
      - Custom rule sets

  DDoS Protection:
    provider: Cloudflare
    features:
      - Layer 3/4 attack mitigation
      - Layer 7 application protection
      - Rate limiting per IP
      - Geographic filtering

  TLS Configuration:
    version: TLS 1.3 minimum
    cipher_suites:
      - TLS_AES_256_GCM_SHA384
      - TLS_AES_128_GCM_SHA256
      - TLS_CHACHA20_POLY1305_SHA256
    certificate: Let's Encrypt with auto-renewal
    HSTS: max-age=31536000; includeSubDomains
```

### Layer 2: API Gateway Security

```python
# rate_limiter.py
class RateLimiter:
    def __init__(self):
        self.limits = {
            'free': {'rpm': 60, 'burst': 100},
            'standard': {'rpm': 120, 'burst': 200},
            'premium': {'rpm': 300, 'burst': 500}
        }
        self.redis = RedisClient()

    async def check_rate_limit(self, api_key, tier):
        key = f"rate_limit:{api_key}"
        current = await self.redis.incr(key)

        if current == 1:
            await self.redis.expire(key, 60)

        limit = self.limits[tier]['rpm']
        if current > limit:
            raise RateLimitExceeded(
                limit=limit,
                reset_in=await self.redis.ttl(key)
            )

        return {
            'remaining': limit - current,
            'reset': await self.redis.ttl(key)
        }
```

```yaml
API Gateway Configuration:
  Authentication:
    methods:
      - API Key (header: X-API-Key)
      - JWT Bearer token
      - OAuth 2.0

  Input Validation:
    - JSON schema validation
    - Parameter type checking
    - Size limits (max 1MB payload)
    - SQL injection prevention
    - Path traversal prevention

  Output Sanitization:
    - Remove sensitive data
    - Escape HTML/JavaScript
    - Validate response schema
```

### Layer 3: Agent Isolation

#### Container Security

```dockerfile
# Secure Agent Container
FROM python:3.11-slim

# Security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tini \
        libseccomp2 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r agent && \
    useradd -r -g agent -u 1000 agent && \
    mkdir -p /home/agent && \
    chown -R agent:agent /home/agent

# Security hardening
RUN echo "agent ALL=(ALL) NOPASSWD: /bin/false" >> /etc/sudoers && \
    chmod 700 /home/agent

# Drop capabilities
USER agent
WORKDIR /home/agent

# Copy only necessary files
COPY --chown=agent:agent requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=agent:agent agent.py .

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "agent.py"]
```

#### Runtime Security

```yaml
Docker Security Options:
  security_opt:
    - no-new-privileges:true
    - seccomp:profiles/agent.json
    - apparmor:docker-agent

  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE  # If needed

  resource_limits:
    cpus: "0.5"
    memory: "512m"
    memory-swap: "512m"
    pids_limit: 100
    ulimits:
      nofile:
        soft: 1024
        hard: 2048

  network_mode: bridge
  read_only: true
  tmpfs:
    - /tmp:noexec,nosuid,size=100M
```

#### Seccomp Profile

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close",
        "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect",
        "munmap", "brk", "rt_sigaction",
        "rt_sigprocmask", "rt_sigreturn",
        "ioctl", "pread64", "pwrite64",
        "readv", "writev", "access", "pipe",
        "select", "sched_yield", "mremap",
        "msync", "mincore", "madvise",
        "shmget", "shmat", "shmctl"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

### Layer 4: Content Moderation

```python
# content_moderation.py
class ContentModerationPipeline:
    def __init__(self):
        self.openai_client = OpenAI()
        self.custom_filters = CustomFilters()
        self.violation_store = ViolationStore()

    async def moderate(self, content, agent_id, context):
        # Stage 1: Quick pattern matching
        quick_check = self.custom_filters.quick_check(content)
        if quick_check.is_blocked:
            return self._block_content(quick_check.reason)

        # Stage 2: OpenAI Moderation
        openai_result = await self._check_openai(content)
        if openai_result.flagged:
            await self._handle_violation(
                agent_id, content, openai_result
            )
            return self._block_content(openai_result.categories)

        # Stage 3: Context-aware checks
        context_result = await self._check_context(content, context)
        if context_result.inappropriate:
            return self._filter_content(content, context_result)

        # Stage 4: Toxicity scoring
        toxicity = await self._score_toxicity(content)
        if toxicity > 0.8:
            return self._block_content("High toxicity score")

        return ModerationResult(
            allowed=True,
            content=content,
            score=toxicity
        )

    async def _check_openai(self, content):
        response = await self.openai_client.moderations.create(
            input=content
        )
        return response.results[0]

    async def _handle_violation(self, agent_id, content, result):
        violation = Violation(
            agent_id=agent_id,
            content=content[:500],  # Truncate for storage
            categories=result.categories,
            timestamp=datetime.utcnow()
        )

        await self.violation_store.record(violation)

        # Escalation logic
        count = await self.violation_store.count(agent_id, days=7)
        if count >= 3:
            await self._suspend_agent(agent_id)
        elif count >= 1:
            await self._warn_agent(agent_id)
```

#### Custom Filter Rules

```python
# custom_filters.py
class CustomFilters:
    def __init__(self):
        self.blocked_patterns = [
            r'(?i)\b(password|secret|key|token)\s*[:=]\s*\S+',
            r'(?i)\b\d{3}-?\d{2}-?\d{4}\b',  # SSN pattern
            r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
            r'(?i)\b(?:\d{4}[\s-]?){3}\d{4}\b',  # Credit card
        ]

        self.blocked_domains = [
            'malicious-site.com',
            'phishing-domain.net'
        ]

        self.sensitive_topics = [
            'violence', 'self-harm', 'illegal-activity'
        ]

    def quick_check(self, content):
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, content):
                return FilterResult(
                    is_blocked=True,
                    reason="Sensitive information detected"
                )

        # Check blocked domains
        for domain in self.blocked_domains:
            if domain in content:
                return FilterResult(
                    is_blocked=True,
                    reason="Blocked domain reference"
                )

        return FilterResult(is_blocked=False)
```

### Layer 5: Memory Isolation

```python
# memory_isolation.py
class IsolatedMemoryStore:
    def __init__(self):
        self.encryption_key = self._load_encryption_key()
        self.access_control = AccessControl()

    async def store_memory(self, agent_id, memory):
        # Validate agent owns the memory space
        if not await self.access_control.can_write(agent_id, memory):
            raise PermissionDenied()

        # Encrypt sensitive data
        encrypted = self._encrypt(memory.content)

        # Store with agent-specific prefix
        key = f"agent:{agent_id}:memory:{memory.id}"
        await self.vector_store.upsert(
            id=key,
            values=memory.embedding,
            metadata={
                'encrypted_content': encrypted,
                'agent_id': agent_id,
                'access_level': 'private'
            }
        )

    async def retrieve_memory(self, agent_id, query):
        # Ensure agent can only query own memories
        filter = {'agent_id': agent_id}

        results = await self.vector_store.query(
            vector=query.embedding,
            filter=filter,
            top_k=10
        )

        # Decrypt results
        for result in results:
            result['content'] = self._decrypt(
                result['metadata']['encrypted_content']
            )

        return results

    def _encrypt(self, data):
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.encode()).decode()

    def _decrypt(self, encrypted_data):
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data.encode()).decode()
```

### Layer 6: Transaction Security

```python
# transaction_security.py
class SecureTransactionProcessor:
    def __init__(self):
        self.fraud_detector = FraudDetector()
        self.audit_logger = AuditLogger()

    async def process_transaction(self, transaction):
        # Validate transaction integrity
        if not self._validate_signature(transaction):
            raise InvalidSignature()

        # Check fraud indicators
        fraud_score = await self.fraud_detector.score(transaction)
        if fraud_score > 0.8:
            await self._flag_suspicious(transaction)
            raise SuspiciousTransaction()

        # Enforce limits
        if not await self._check_limits(transaction):
            raise TransactionLimitExceeded()

        # Double-entry bookkeeping
        async with self.db.transaction():
            # Debit sender
            await self._debit(
                transaction.sender,
                transaction.amount
            )

            # Credit receiver
            await self._credit(
                transaction.receiver,
                transaction.amount
            )

            # Log transaction
            await self.audit_logger.log(transaction)

        return TransactionResult(
            id=transaction.id,
            status='completed',
            timestamp=datetime.utcnow()
        )

    async def _check_limits(self, transaction):
        # Daily limit check
        daily_total = await self._get_daily_total(
            transaction.sender
        )

        if daily_total + transaction.amount > 10000:
            return False

        # Single transaction limit
        if transaction.amount > 5000:
            return False

        return True
```

## Security Monitoring

### Real-time Monitoring

```yaml
Monitoring Stack:
  Prometheus:
    metrics:
      - agent_violations_total
      - api_auth_failures_total
      - rate_limit_exceeded_total
      - suspicious_transactions_total
      - container_escapes_total

  Grafana Dashboards:
    - Security Overview
    - Agent Violations
    - API Security
    - Transaction Monitoring
    - Resource Usage

  AlertManager Rules:
    - name: HighViolationRate
      expr: rate(agent_violations_total[5m]) > 10
      severity: warning

    - name: ContainerEscape
      expr: container_escapes_total > 0
      severity: critical

    - name: SuspiciousTransactionSpike
      expr: rate(suspicious_transactions_total[1m]) > 5
      severity: high
```

### Security Audit Logging

```python
# audit_logging.py
class SecurityAuditLogger:
    def __init__(self):
        self.elasticsearch = ElasticsearchClient()

    async def log_security_event(self, event):
        document = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event.type,
            'severity': event.severity,
            'agent_id': event.agent_id,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'details': event.details,
            'correlation_id': event.correlation_id
        }

        # Index in Elasticsearch
        await self.elasticsearch.index(
            index='security-audit-2025-01',
            document=document
        )

        # Alert on critical events
        if event.severity == 'critical':
            await self._send_alert(event)
```

## Incident Response

### Response Procedures

```yaml
Incident Response Plan:
  Detection:
    - Automated alerts from monitoring
    - User reports
    - Security scanning

  Triage:
    - Severity assessment (P1-P4)
    - Impact analysis
    - Resource assignment

  Containment:
    - Isolate affected agents
    - Block malicious IPs
    - Suspend compromised accounts

  Eradication:
    - Remove malicious code
    - Patch vulnerabilities
    - Update security rules

  Recovery:
    - Restore normal operations
    - Verify security controls
    - Monitor for recurrence

  Post-Incident:
    - Root cause analysis
    - Update documentation
    - Security control improvements
```

### Automated Response

```python
# incident_response.py
class AutomatedIncidentResponse:
    async def handle_security_incident(self, incident):
        severity = self.assess_severity(incident)

        if severity == 'critical':
            # Immediate automated response
            await self.isolate_agent(incident.agent_id)
            await self.block_ip(incident.source_ip)
            await self.notify_security_team(incident)

        elif severity == 'high':
            # Contain and investigate
            await self.suspend_agent(incident.agent_id)
            await self.collect_forensics(incident)

        elif severity == 'medium':
            # Monitor and warn
            await self.warn_agent(incident.agent_id)
            await self.increase_monitoring(incident.agent_id)

        # Log all incidents
        await self.log_incident(incident)
```

## Compliance & Privacy

### Data Protection

```yaml
GDPR Compliance:
  Data Minimization:
    - Collect only necessary data
    - Auto-delete after retention period

  Right to Access:
    - API endpoint for data export
    - 30-day response time

  Right to Erasure:
    - Complete data deletion
    - Cascade to all systems

  Data Portability:
    - JSON/CSV export formats
    - Standard data schemas

Encryption:
  At Rest:
    - AES-256-GCM
    - Key rotation every 90 days

  In Transit:
    - TLS 1.3 minimum
    - Perfect Forward Secrecy

  Key Management:
    - Hardware Security Module (HSM)
    - AWS KMS or HashiCorp Vault
```

### Privacy Controls

```python
# privacy_manager.py
class PrivacyManager:
    async def handle_data_request(self, user_id, request_type):
        if request_type == 'export':
            data = await self.collect_user_data(user_id)
            return self.format_export(data)

        elif request_type == 'delete':
            await self.delete_user_data(user_id)
            return {'status': 'deleted', 'timestamp': datetime.utcnow()}

        elif request_type == 'rectify':
            await self.update_user_data(user_id, request.updates)
            return {'status': 'updated'}

    async def delete_user_data(self, user_id):
        # Delete from all systems
        await self.db.delete_user(user_id)
        await self.vector_store.delete_agent_memories(user_id)
        await self.cache.delete_user_data(user_id)
        await self.audit_log.record_deletion(user_id)
```

## Security Testing

### Penetration Testing

```yaml
Testing Schedule:
  Quarterly:
    - External penetration testing
    - OWASP Top 10 verification

  Monthly:
    - Automated vulnerability scanning
    - Dependency checking

  Weekly:
    - Security regression tests
    - Container escape tests

  Daily:
    - SIEM log analysis
    - Anomaly detection
```

### Chaos Engineering

```python
# chaos_testing.py
class SecurityChaosTests:
    async def test_malicious_agent(self):
        # Deploy agent with malicious behavior
        agent = await self.deploy_test_agent(
            behavior='malicious'
        )

        # Verify containment
        assert agent.status == 'sandboxed'

        # Attempt privilege escalation
        result = await agent.attempt_escalation()
        assert result == 'blocked'

    async def test_ddos_resilience(self):
        # Simulate DDoS attack
        await self.simulate_traffic_spike(
            requests_per_second=10000
        )

        # Verify service availability
        health = await self.check_health()
        assert health.status == 'operational'
```

## Security Metrics

### Key Performance Indicators

```yaml
Security KPIs:
  Mean Time to Detect (MTTD): <5 minutes
  Mean Time to Respond (MTTR): <15 minutes
  False Positive Rate: <5%
  Violation Detection Rate: >95%
  Uptime during attacks: >99.9%
  Patch deployment time: <24 hours
```

## Conclusion

This multi-layered security framework ensures safe operation of the multi-agent city platform. Each layer provides independent protection, creating defense-in-depth that can withstand various attack vectors while maintaining performance and user experience.