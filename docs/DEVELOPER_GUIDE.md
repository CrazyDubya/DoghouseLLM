# Developer Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Adding Features](#adding-features)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [API Development](#api-development)
8. [Agent Development](#agent-development)
9. [Contributing](#contributing)
10. [Advanced Topics](#advanced-topics)

---

## Getting Started

### Development Environment Setup

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/multi-agent-city.git
cd multi-agent-city

# Add upstream remote
git remote add upstream https://github.com/original-org/multi-agent-city.git
```

#### 2. Install Development Tools
```bash
# Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install
```

#### 3. IDE Setup

##### VS Code
```json
// .vscode/settings.json
{
  "python.defaultInterpreter": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true
}
```

##### PyCharm
- Set Python interpreter to `venv/bin/python`
- Enable Black formatter
- Configure pytest as test runner

---

## Project Structure

```
multi-agent-city/
│
├── services/                    # Microservices
│   ├── world-orchestrator/     # World simulation engine
│   │   ├── main.py            # FastAPI application
│   │   ├── world_engine.py    # Core simulation logic
│   │   ├── economy_system.py  # Economic mechanics
│   │   ├── governance_system.py # Voting and proposals
│   │   └── property_manager.py # Property management
│   │
│   ├── agent-scheduler/        # Agent execution runtime
│   │   ├── main.py            # FastAPI application
│   │   ├── agent_runtime.py   # Agent execution engine
│   │   ├── memory_system.py   # Memory management
│   │   ├── llm_integration.py # LLM connections
│   │   ├── interaction_system.py # Agent interactions
│   │   └── social_system.py   # Relationships
│   │
│   ├── api-gateway/           # Central API entry
│   │   ├── main.py           # API routing
│   │   ├── auth.py           # Authentication
│   │   └── rate_limiter.py   # Rate limiting
│   │
│   └── web-dashboard/         # Monitoring interface
│       ├── main.py           # FastAPI + WebSocket
│       ├── templates/        # HTML templates
│       └── static/           # CSS/JS assets
│
├── packages/                   # Shared code
│   └── shared-types/          # Common data models
│       └── models.py          # Pydantic models
│
├── infrastructure/            # Deployment configs
│   ├── docker/               # Docker configurations
│   ├── kubernetes/           # K8s manifests
│   └── terraform/            # Cloud infrastructure
│
├── scripts/                   # Utility scripts
│   ├── init_db.sql          # Database schema
│   ├── seed_data.py         # Test data generation
│   └── migrate.py           # Database migrations
│
├── tests/                     # Test suites
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── e2e/                 # End-to-end tests
│
└── docs/                      # Documentation
    ├── API_DOCUMENTATION.md  # API reference
    ├── DEPLOYMENT_GUIDE.md   # Deployment instructions
    └── DEVELOPER_GUIDE.md    # This file
```

---

## Development Workflow

### Branch Strategy

```
main
 ├── develop
 │    ├── feature/add-weather-system
 │    ├── feature/improve-llm-integration
 │    └── bugfix/fix-memory-leak
 └── release/v1.1.0
```

### Workflow Steps

1. **Create Feature Branch**
```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

2. **Make Changes**
```bash
# Edit files
# Add tests
# Update documentation
```

3. **Test Locally**
```bash
# Run unit tests
pytest tests/unit/

# Run specific service
docker-compose up world-orchestrator

# Test with curl
curl http://localhost:8001/health
```

4. **Commit Changes**
```bash
# Stage changes
git add .

# Commit with conventional commits
git commit -m "feat(economy): add dynamic pricing model"
# or
git commit -m "fix(agents): resolve memory leak in scheduler"
```

5. **Push and Create PR**
```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

### Conventional Commits

Use these prefixes:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style
- `refactor:` Code refactoring
- `test:` Tests
- `chore:` Maintenance

---

## Adding Features

### Example: Adding Weather System

#### 1. Plan the Feature
```python
# services/world-orchestrator/weather_system.py
"""
Weather system for the multi-agent city.
Features:
- Dynamic weather patterns
- Seasonal changes
- Weather events
- Agent behavior impact
"""
```

#### 2. Define Data Models
```python
# packages/shared-types/models.py
from enum import Enum
from pydantic import BaseModel

class WeatherType(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"

class Weather(BaseModel):
    type: WeatherType
    temperature: float  # Celsius
    humidity: float  # 0-1
    wind_speed: float  # km/h
    visibility: float  # 0-1
```

#### 3. Implement Core Logic
```python
# services/world-orchestrator/weather_system.py
import random
from datetime import datetime, timedelta
from typing import Optional

class WeatherSystem:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.current_weather = None
        self.forecast = []

    async def initialize(self):
        """Initialize weather system"""
        self.current_weather = await self.generate_weather()
        self.forecast = await self.generate_forecast()

    async def generate_weather(self, previous: Optional[Weather] = None) -> Weather:
        """Generate realistic weather based on previous conditions"""
        if previous:
            # Gradual transitions
            temperature_change = random.uniform(-2, 2)
            new_temp = max(-20, min(45, previous.temperature + temperature_change))
        else:
            # Random initial weather
            new_temp = random.uniform(10, 30)

        return Weather(
            type=self._determine_weather_type(new_temp),
            temperature=new_temp,
            humidity=random.uniform(0.3, 0.9),
            wind_speed=random.uniform(0, 50),
            visibility=random.uniform(0.5, 1.0)
        )

    def _determine_weather_type(self, temperature: float) -> WeatherType:
        """Determine weather type based on temperature"""
        if temperature < 0:
            return WeatherType.SNOWY
        elif temperature < 10:
            return random.choice([WeatherType.CLOUDY, WeatherType.RAINY])
        elif temperature < 25:
            return random.choice([WeatherType.SUNNY, WeatherType.CLOUDY])
        else:
            return WeatherType.SUNNY

    async def update(self):
        """Update weather every tick"""
        self.current_weather = await self.generate_weather(self.current_weather)
        await self._save_to_redis()
        await self._notify_agents()

    async def get_weather_at_location(self, location: Location) -> Weather:
        """Get localized weather (can vary by district)"""
        # Add local variations
        local_weather = self.current_weather.copy()
        if location.district == "Coastal":
            local_weather.humidity *= 1.2
            local_weather.wind_speed *= 1.5
        return local_weather
```

#### 4. Integrate with World Engine
```python
# services/world-orchestrator/world_engine.py
from weather_system import WeatherSystem

class WorldEngine:
    def __init__(self, database, redis_client, mqtt_client):
        # ... existing code ...
        self.weather = None

    async def initialize(self):
        # ... existing code ...
        # Initialize weather system
        from weather_system import WeatherSystem
        self.weather = WeatherSystem(self.redis)
        await self.weather.initialize()

    async def tick(self):
        """Process one simulation tick"""
        # ... existing code ...
        # Update weather
        if self.current_tick % 10 == 0:  # Update every 10 ticks
            await self.weather.update()
```

#### 5. Add API Endpoints
```python
# services/world-orchestrator/main.py
@app.get("/weather/current")
async def get_current_weather():
    """Get current weather"""
    if world_engine.weather:
        return world_engine.weather.current_weather.dict()
    else:
        raise HTTPException(status_code=503, detail="Weather system not available")

@app.get("/weather/forecast")
async def get_weather_forecast():
    """Get weather forecast"""
    if world_engine.weather:
        return {
            "forecast": [w.dict() for w in world_engine.weather.forecast]
        }
    else:
        raise HTTPException(status_code=503, detail="Weather system not available")
```

#### 6. Add Tests
```python
# tests/unit/test_weather_system.py
import pytest
from services.world_orchestrator.weather_system import WeatherSystem

@pytest.fixture
async def weather_system(redis_mock):
    system = WeatherSystem(redis_mock)
    await system.initialize()
    return system

async def test_weather_generation(weather_system):
    weather = await weather_system.generate_weather()
    assert weather.temperature >= -20
    assert weather.temperature <= 45
    assert weather.humidity >= 0
    assert weather.humidity <= 1

async def test_weather_transitions(weather_system):
    weather1 = await weather_system.generate_weather()
    weather2 = await weather_system.generate_weather(weather1)
    # Temperature shouldn't change too drastically
    assert abs(weather2.temperature - weather1.temperature) <= 5
```

#### 7. Update Documentation
```markdown
# docs/FEATURES.md
## Weather System

The weather system provides dynamic weather conditions that affect agent behavior and world events.

### Features
- Real-time weather updates
- Seasonal patterns
- Localized weather by district
- Weather forecasting
- Agent behavior impact

### API Endpoints
- `GET /weather/current` - Get current weather
- `GET /weather/forecast` - Get 24-hour forecast
- `GET /weather/history` - Get weather history

### Configuration
```env
WEATHER_UPDATE_INTERVAL=600  # Seconds
WEATHER_SEASONAL_VARIATION=true
WEATHER_EXTREME_EVENTS=true
```
```

---

## Testing

### Test Structure
```
tests/
├── unit/                      # Fast, isolated tests
│   ├── test_economy.py
│   ├── test_agents.py
│   └── test_governance.py
├── integration/               # Service integration tests
│   ├── test_api_gateway.py
│   └── test_world_agent_interaction.py
└── e2e/                      # Full system tests
    ├── test_agent_lifecycle.py
    └── test_governance_flow.py
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/unit/test_economy.py

# With coverage
pytest --cov=services --cov-report=html

# Parallel execution
pytest -n 4

# Verbose output
pytest -v

# Specific test
pytest tests/unit/test_economy.py::test_transaction_processing
```

### Writing Tests

#### Unit Test Example
```python
# tests/unit/test_economy.py
import pytest
from unittest.mock import Mock, AsyncMock
from services.world_orchestrator.economy_system import EconomySystem

@pytest.fixture
async def economy_system():
    db_mock = AsyncMock()
    redis_mock = AsyncMock()
    system = EconomySystem(db_mock, redis_mock)
    await system.initialize()
    return system

@pytest.mark.asyncio
async def test_process_transaction(economy_system):
    # Arrange
    sender_id = uuid4()
    receiver_id = uuid4()
    amount = 100

    # Set initial balances
    economy_system.agent_balances[sender_id] = 500
    economy_system.agent_balances[receiver_id] = 200

    # Act
    transaction = await economy_system.process_transaction(
        sender_id, receiver_id, amount, "payment"
    )

    # Assert
    assert transaction is not None
    assert economy_system.agent_balances[sender_id] == 400
    assert economy_system.agent_balances[receiver_id] == 300
```

#### Integration Test Example
```python
# tests/integration/test_agent_world_interaction.py
import pytest
import httpx
from uuid import uuid4

@pytest.mark.asyncio
async def test_agent_movement():
    async with httpx.AsyncClient() as client:
        # Register agent
        agent_id = str(uuid4())
        response = await client.post(
            f"http://localhost:8002/agents/{agent_id}/register",
            json={"name": "Test Agent", "persona": "Explorer"}
        )
        assert response.status_code == 200

        # Start agent
        response = await client.post(
            f"http://localhost:8002/agents/{agent_id}/start"
        )
        assert response.status_code == 200

        # Move agent
        response = await client.post(
            f"http://localhost:8001/agents/{agent_id}/move",
            json={"location": {"district": "Downtown"}}
        )
        assert response.status_code == 200

        # Verify location
        response = await client.get(
            f"http://localhost:8001/agents/{agent_id}/observation"
        )
        assert response.json()["location"]["district"] == "Downtown"
```

### Test Fixtures
```python
# tests/conftest.py
import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def redis_mock():
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.hget.return_value = None
    mock.hset.return_value = True
    return mock

@pytest.fixture
async def db_mock():
    mock = AsyncMock()
    mock.execute.return_value = None
    mock.fetch_one.return_value = None
    mock.fetch_all.return_value = []
    return mock
```

---

## Code Style

### Python Style Guide

Follow PEP 8 with these additions:

#### Imports
```python
# Standard library
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Third-party
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local
from packages.shared_types.models import Agent, Location
from .economy_system import EconomySystem
```

#### Type Hints
```python
from typing import Dict, List, Optional, Union, Tuple
from uuid import UUID

async def process_transaction(
    sender_id: UUID,
    receiver_id: UUID,
    amount: float,
    transaction_type: str = "payment",
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Transaction]:
    """Process a transaction between agents"""
    pass
```

#### Docstrings
```python
def calculate_gdp(transactions: List[Transaction], period_days: int = 30) -> float:
    """
    Calculate the Gross Domestic Product of the economy.

    Args:
        transactions: List of transactions to analyze
        period_days: Number of days to calculate GDP for

    Returns:
        float: The calculated GDP value

    Raises:
        ValueError: If period_days is less than 1

    Example:
        >>> gdp = calculate_gdp(transactions, 30)
        >>> print(f"Monthly GDP: {gdp}")
    """
    if period_days < 1:
        raise ValueError("Period must be at least 1 day")

    # Implementation
    return sum(t.amount for t in transactions)
```

### Linting and Formatting

```bash
# Format with Black
black services/

# Lint with Flake8
flake8 services/

# Type check with MyPy
mypy services/

# Sort imports
isort services/

# All checks
pre-commit run --all-files
```

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## API Development

### Adding New Endpoints

#### 1. Define Request/Response Models
```python
# packages/shared-types/models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class BusinessCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., regex="^(shop|restaurant|service)$")
    owner_id: str
    location: Location
    initial_capital: float = Field(..., ge=0)

class BusinessResponse(BaseModel):
    id: str
    name: str
    type: str
    owner_id: str
    balance: float
    employees: List[str]
    created_at: datetime
    status: str
```

#### 2. Implement Business Logic
```python
# services/world-orchestrator/business_system.py
class BusinessSystem:
    async def create_business(
        self,
        request: BusinessCreateRequest
    ) -> BusinessResponse:
        """Create a new business"""
        # Validate owner has sufficient capital
        if not await self.economy.has_balance(
            request.owner_id,
            request.initial_capital
        ):
            raise InsufficientFundsError()

        # Create business entity
        business = Business(
            id=str(uuid4()),
            name=request.name,
            type=request.type,
            owner_id=request.owner_id,
            balance=request.initial_capital,
            created_at=datetime.utcnow()
        )

        # Deduct capital from owner
        await self.economy.process_transaction(
            sender_id=request.owner_id,
            receiver_id=business.id,
            amount=request.initial_capital,
            transaction_type="business_investment"
        )

        # Save to database
        await self.db.save_business(business)

        return BusinessResponse(**business.dict())
```

#### 3. Add API Endpoint
```python
# services/world-orchestrator/main.py
from fastapi import Depends, HTTPException
from packages.shared_types.models import BusinessCreateRequest, BusinessResponse

@app.post("/businesses", response_model=BusinessResponse)
async def create_business(
    request: BusinessCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new business"""
    try:
        business = await world_engine.business_system.create_business(request)
        return business
    except InsufficientFundsError:
        raise HTTPException(
            status_code=400,
            detail="Insufficient funds to create business"
        )
    except Exception as e:
        logger.error(f"Error creating business: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### 4. Add Validation and Middleware
```python
# Custom validation
from fastapi import Request
from fastapi.responses import JSONResponse

@app.middleware("http")
async def validate_request_size(request: Request, call_next):
    """Limit request body size"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1_000_000:  # 1MB limit
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"}
        )
    response = await call_next(request)
    return response

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/businesses")
@limiter.limit("5/minute")
async def create_business(request: Request, ...):
    pass
```

---

## Agent Development

### Creating Custom Agent Types

#### 1. Define Agent Behavior
```python
# services/agent-scheduler/behaviors/merchant_agent.py
from typing import List, Dict
from packages.shared_types.models import Agent, Action

class MerchantAgentBehavior:
    """Behavior for merchant-type agents"""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.inventory: Dict[str, int] = {}
        self.pricing_strategy = "market_based"

    async def decide_action(
        self,
        observation: Observation,
        memories: List[Memory]
    ) -> Action:
        """Decide next action based on merchant goals"""

        # Check if shop is open
        if self.should_open_shop(observation.time_of_day):
            return Action(type="open_shop")

        # Check for customers
        nearby_agents = observation.visible_agents
        if nearby_agents:
            potential_customers = self.identify_customers(nearby_agents, memories)
            if potential_customers:
                return Action(
                    type="advertise",
                    parameters={"message": self.create_advertisement()}
                )

        # Restock if needed
        if self.needs_restocking():
            return Action(
                type="purchase_inventory",
                parameters={"items": self.calculate_restock_list()}
            )

        # Default action
        return Action(type="wait")

    def should_open_shop(self, time_of_day: str) -> bool:
        """Determine if shop should be open"""
        return time_of_day in ["morning", "afternoon", "evening"]

    def identify_customers(
        self,
        nearby_agents: List[Agent],
        memories: List[Memory]
    ) -> List[Agent]:
        """Identify potential customers from nearby agents"""
        customers = []
        for agent in nearby_agents:
            # Check if agent has bought before
            past_customer = any(
                m for m in memories
                if m.type == "transaction" and agent.id in m.participants
            )
            if past_customer or random.random() > 0.3:
                customers.append(agent)
        return customers
```

#### 2. Register Agent Type
```python
# services/agent-scheduler/agent_registry.py
from behaviors.merchant_agent import MerchantAgentBehavior
from behaviors.explorer_agent import ExplorerAgentBehavior
from behaviors.social_agent import SocialAgentBehavior

AGENT_BEHAVIORS = {
    "merchant": MerchantAgentBehavior,
    "explorer": ExplorerAgentBehavior,
    "social": SocialAgentBehavior,
    "default": DefaultAgentBehavior
}

def get_agent_behavior(agent_type: str):
    """Get behavior class for agent type"""
    return AGENT_BEHAVIORS.get(agent_type, DefaultAgentBehavior)
```

### LLM Integration for Agents

#### Custom LLM Providers
```python
# services/agent-scheduler/llm_providers/custom_llm.py
from typing import Dict, Any
from langchain.llms.base import LLM

class CustomLLM(LLM):
    """Custom LLM implementation"""

    api_url: str
    api_key: str
    model_name: str = "custom-model"

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """Call the custom LLM API"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "prompt": prompt,
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7)
        }

        response = requests.post(
            f"{self.api_url}/generate",
            json=data,
            headers=headers
        )

        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"LLM API error: {response.text}")
```

---

## Contributing

### Contribution Process

1. **Find an Issue**
   - Check [GitHub Issues](https://github.com/your-org/multi-agent-city/issues)
   - Look for `good first issue` or `help wanted` labels
   - Comment on the issue to claim it

2. **Development**
   - Follow the development workflow
   - Write tests for new features
   - Update documentation

3. **Pull Request**
   - Fill out the PR template
   - Link related issues
   - Ensure CI passes

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Code Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code quality checks
   - Security scanning

2. **Human Review**
   - At least one maintainer review
   - Focus on architecture, performance, security
   - Constructive feedback

3. **Merge Requirements**
   - All CI checks pass
   - Approved by maintainer
   - No merge conflicts
   - Documentation updated

---

## Advanced Topics

### Performance Optimization

#### Profiling
```python
# Profile a function
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your code
    result = expensive_function()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

#### Async Optimization
```python
# Batch database queries
async def get_agents_batch(agent_ids: List[UUID]) -> List[Agent]:
    """Fetch multiple agents in one query"""
    query = """
        SELECT * FROM agents
        WHERE id = ANY($1)
    """
    rows = await db.fetch_all(query, agent_ids)
    return [Agent(**row) for row in rows]

# Use asyncio.gather for parallel operations
async def process_agents_parallel(agents: List[Agent]):
    """Process multiple agents in parallel"""
    tasks = [process_agent(agent) for agent in agents]
    results = await asyncio.gather(*tasks)
    return results
```

#### Caching Strategies
```python
# Redis caching decorator
from functools import wraps
import pickle

def redis_cache(expire_seconds: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            cached = await redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await redis_client.set(
                cache_key,
                pickle.dumps(result),
                ex=expire_seconds
            )

            return result
        return wrapper
    return decorator

# Usage
@redis_cache(expire_seconds=60)
async def get_expensive_data(param: str):
    # Expensive operation
    return result
```

### Security Best Practices

#### Input Validation
```python
from pydantic import BaseModel, validator, Field
import re

class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    persona: str = Field(..., max_length=500)

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Name contains invalid characters')
        return v

    @validator('persona')
    def validate_persona(cls, v):
        # Check for injection attempts
        dangerous_patterns = ['<script', 'javascript:', 'onclick']
        if any(pattern in v.lower() for pattern in dangerous_patterns):
            raise ValueError('Persona contains potentially dangerous content')
        return v
```

#### SQL Injection Prevention
```python
# Always use parameterized queries
async def get_agent_by_name(name: str) -> Optional[Agent]:
    # Good - Parameterized
    query = "SELECT * FROM agents WHERE name = $1"
    row = await db.fetch_one(query, name)

    # Bad - String concatenation
    # query = f"SELECT * FROM agents WHERE name = '{name}'"

    return Agent(**row) if row else None
```

#### Authentication
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await get_user(username=username)
    if user is None:
        raise credentials_exception

    return user
```

### Debugging Tips

#### Logging
```python
import structlog

logger = structlog.get_logger()

# Structured logging
logger.info(
    "agent_action",
    agent_id=agent.id,
    action_type=action.type,
    parameters=action.parameters,
    timestamp=datetime.utcnow()
)

# Context binding
log = logger.bind(request_id=request_id)
log.info("processing_request")
# All subsequent logs include request_id
```

#### Debug Mode
```python
# Enable debug mode for detailed output
if os.getenv("DEBUG") == "true":
    logging.basicConfig(level=logging.DEBUG)

    # Add debug endpoints
    @app.get("/debug/state")
    async def get_debug_state():
        return {
            "agents": len(world_engine.agents),
            "memory_usage": get_memory_usage(),
            "cache_stats": await get_cache_stats()
        }
```

#### Remote Debugging
```python
# Using debugpy for remote debugging
import debugpy

if os.getenv("ENABLE_DEBUGPY") == "true":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
```

---

## Resources

### Documentation
- [API Reference](./API_DOCUMENTATION.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Docker Documentation](https://docs.docker.com/)

### Community
- [GitHub Discussions](https://github.com/your-org/multi-agent-city/discussions)
- [Discord Server](https://discord.gg/multiagentcity)
- [Stack Overflow Tag](https://stackoverflow.com/questions/tagged/multiagent-city)

### Papers and Research
- [Generative Agents (Stanford)](https://arxiv.org/abs/2304.03442)
- [AgentSociety](https://github.com/agentsociety/agentsociety)
- [Multi-Agent Systems](https://www.cambridge.org/core/books/multiagent-systems/)