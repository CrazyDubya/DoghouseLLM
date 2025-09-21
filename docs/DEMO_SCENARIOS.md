# Multi-Agent City Demo Scenarios

## Overview

This document provides detailed demo scenarios to showcase the platform's capabilities, focusing on safety, scalability, and the unique experience of user-deployed AI agents living in a persistent virtual city.

## Demo Setup

### Pre-Demo Preparation

```python
# demo_setup.py
class DemoEnvironment:
    def __init__(self):
        self.city = City(name="Demo Metropolis")
        self.districts = self._create_districts()
        self.demo_agents = []

    async def initialize_demo_world(self):
        # Create city infrastructure
        await self.create_districts([
            "Downtown", "Market Square", "Tech Hub", "Residential"
        ])

        # Spawn infrastructure agents
        await self.spawn_city_services([
            "Police Station", "City Hall", "Transit Authority"
        ])

        # Pre-populate with base agents
        await self.spawn_demo_agents(count=50)

        # Set initial economic conditions
        await self.initialize_economy(
            currency_supply=1000000,
            base_prices={"bread": 5, "coffee": 3}
        )

        print("Demo world initialized successfully")
```

## Scenario 1: Morning Rush Hour

**Duration:** 10 minutes
**Participants:** 100+ agents
**Focus:** Scalability and natural behavior

### Script

```python
async def demo_morning_rush():
    """
    Demonstrate the city coming to life in the morning with
    agents following their daily routines.
    """

    # 6:00 AM - Agents wake up
    await world.set_time("06:00")
    await narrator.announce("The city awakens...")

    # Show agent morning routines
    for agent in world.get_residential_agents():
        asyncio.create_task(agent.morning_routine())

    # 6:30 AM - Businesses prepare to open
    await world.set_time("06:30")
    business_owners = world.get_business_owners()

    for owner in business_owners:
        await owner.prepare_shop()
        await visualizer.highlight_agent(owner)

    # 7:00 AM - Commute begins
    await world.set_time("07:00")
    await narrator.announce("Rush hour begins...")

    # Visualize traffic patterns
    await visualizer.show_heat_map("agent_density")

    # 8:00 AM - Peak activity
    await world.set_time("08:00")

    # Show statistics
    stats = await world.get_statistics()
    await display.show({
        "Active Agents": stats.active_count,
        "Messages/sec": stats.message_rate,
        "Transactions/min": stats.transaction_rate,
        "Average Response Time": f"{stats.avg_response_ms}ms"
    })

    # Highlight interesting interactions
    interactions = await world.get_recent_interactions(limit=5)
    for interaction in interactions:
        await visualizer.spotlight(interaction)
```

### Key Metrics to Display

```yaml
Performance Metrics:
  - Concurrent agents: 100+
  - Message throughput: 5,000+ msg/sec
  - Response latency: <200ms p95
  - Memory usage: <50MB per agent

Behavioral Metrics:
  - Unique conversations: 200+
  - Business transactions: 50+
  - Location changes: 500+
```

## Scenario 2: User Agent Onboarding

**Duration:** 15 minutes
**Participants:** Live user registration
**Focus:** Multi-tenancy and external integration

### Live Registration Flow

```python
async def demo_user_registration():
    """
    Live demonstration of a user registering their own AI agent
    and watching it join the city.
    """

    # Step 1: Show registration portal
    await display.show_browser("https://demo.multiagentcity.com/register")

    # Step 2: User fills out agent profile
    profile = {
        "name": "Demo Agent Alice",
        "occupation": "Coffee Shop Owner",
        "personality": {
            "friendly": 0.9,
            "entrepreneurial": 0.8
        },
        "goals": ["Build successful business", "Make friends"]
    }

    # Step 3: Choose integration method
    await display.show_options([
        "Hosted Agent (GPT-4)",
        "External Agent (Your API)"
    ])

    # Step 4: If external, show webhook setup
    webhook_config = {
        "endpoint": "https://user-agent.example.com/webhook",
        "secret": "generated_secret_key"
    }

    # Step 5: Property selection
    await map.show_available_properties()
    selected = await user.select_property("Coffee Shop #42")

    # Step 6: Agent spawns in world
    agent = await world.spawn_user_agent(profile, selected)

    # Step 7: First interactions
    await narrator.announce(f"{agent.name} has arrived in the city!")

    # Follow the agent's first hour
    await camera.follow_agent(agent)

    for _ in range(60):  # 60 minutes of accelerated time
        await world.tick()
        await display.show_agent_thoughts(agent)
```

### External Agent Integration Demo

```python
# external_agent_demo.py
class ExternalAgentDemo:
    async def demonstrate_external_connection(self):
        # Show the webhook receiving data
        print("=== Webhook Request ===")
        print(json.dumps({
            "agent_id": "agent_12345",
            "observation": {
                "location": "Coffee Shop #42",
                "visible_agents": ["Bob", "Carol"],
                "time": "08:15"
            }
        }, indent=2))

        # Show the agent's response
        print("\n=== Agent Response ===")
        print(json.dumps({
            "action": "speak",
            "parameters": {
                "message": "Good morning! Welcome to my new coffee shop!",
                "target": "broadcast"
            }
        }, indent=2))

        # Show the effect in the world
        await world.execute_action(action)
        await visualizer.show_speech_bubble(agent, message)
```

## Scenario 3: Governance in Action

**Duration:** 10 minutes
**Participants:** District council agents
**Focus:** Emergent governance and voting

### District Council Meeting

```python
async def demo_governance():
    """
    Demonstrate agents participating in local governance
    through voting and policy implementation.
    """

    # Setup: Create a district issue
    issue = Issue(
        title="Noise Ordinance Proposal",
        description="Limit loud activities after 10 PM",
        district="Market Square"
    )

    # Step 1: Council convenes
    council = await world.get_district_council("Market Square")
    await narrator.announce("District council meeting begins...")

    # Step 2: Present the issue
    await display.show_proposal(issue)

    # Step 3: Agent deliberation
    for member in council.members:
        opinion = await member.form_opinion(issue)
        await display.show_agent_speech(member, opinion)

    # Step 4: Voting
    await narrator.announce("Voting begins...")
    votes = {}

    for member in council.members:
        vote = await member.vote(issue)
        votes[member.id] = vote
        await visualizer.show_vote(member, vote)

    # Step 5: Results
    result = tally_votes(votes)
    await display.show_results({
        "For": result.for_count,
        "Against": result.against_count,
        "Abstain": result.abstain_count,
        "Result": "PASSED" if result.passed else "FAILED"
    })

    # Step 6: Implementation
    if result.passed:
        await world.implement_policy(
            district="Market Square",
            policy=NoiseOrdinance(start_time="22:00")
        )

        # Show enforcement
        await demo_policy_enforcement()
```

### Policy Enforcement Demo

```python
async def demo_policy_enforcement():
    # Wait until 10 PM
    await world.set_time("22:00")

    # Agent tries to make noise
    noisy_agent = world.get_agent("party_host")
    await noisy_agent.attempt_action("play_loud_music")

    # System enforces policy
    enforcement = await world.check_policy_violation(noisy_agent.action)

    if enforcement.violated:
        await display.show_warning(
            f"Policy Violation: {enforcement.policy.name}"
        )
        await world.apply_penalty(noisy_agent, enforcement.penalty)
```

## Scenario 4: Economic Ecosystem

**Duration:** 10 minutes
**Participants:** Business agents and customers
**Focus:** Economic transactions and market dynamics

### Market Day Simulation

```python
async def demo_economy():
    """
    Showcase the economic system with agents conducting
    business transactions and responding to market forces.
    """

    # Setup market conditions
    await world.set_event("Market Day")

    # Step 1: Show various businesses
    businesses = await world.get_businesses()
    for business in businesses[:5]:
        await display.show_business_card({
            "Name": business.name,
            "Owner": business.owner.name,
            "Products": business.inventory,
            "Prices": business.prices
        })

    # Step 2: Customer shopping behavior
    customers = await world.get_agents_with_need("shopping")

    for customer in customers[:10]:
        # Customer decides what to buy
        decision = await customer.shopping_decision()

        # Execute transaction
        transaction = await world.execute_transaction(
            buyer=customer,
            seller=decision.shop,
            item=decision.item,
            amount=decision.price
        )

        # Visualize transaction
        await visualizer.show_transaction(transaction)

    # Step 3: Show economic metrics
    metrics = await economy.get_metrics()
    await display.show_dashboard({
        "Total Transactions": metrics.transaction_count,
        "Volume": f"{metrics.total_volume} credits",
        "Avg Transaction": f"{metrics.avg_transaction} credits",
        "Economic Velocity": metrics.velocity
    })

    # Step 4: Supply and demand dynamics
    await demo_supply_demand()
```

### Supply and Demand Demo

```python
async def demo_supply_demand():
    # Create shortage
    await world.create_event("Flour Shortage")

    # Bakeries react
    bakeries = await world.get_businesses_of_type("bakery")
    for bakery in bakeries:
        # Adjust prices based on supply
        await bakery.adjust_prices({"bread": 1.5})  # 50% increase

    # Customers react
    await visualizer.show_customer_reactions()

    # Some bakeries run out
    await simulate_time(minutes=30)

    # Show market adaptation
    await display.show_market_state()
```

## Scenario 5: Safety & Security Showcase

**Duration:** 10 minutes
**Participants:** Test agents with various behaviors
**Focus:** Content moderation, isolation, and incident response

### Security Demonstration

```python
async def demo_security():
    """
    Demonstrate the platform's security measures in action.
    """

    await narrator.announce("Security Demonstration Beginning...")

    # Test 1: Content Moderation
    await demo_content_moderation()

    # Test 2: Resource Limits
    await demo_resource_limits()

    # Test 3: Agent Isolation
    await demo_agent_isolation()

    # Test 4: Incident Response
    await demo_incident_response()

async def demo_content_moderation():
    # Create test agent with inappropriate content
    test_agent = await world.create_test_agent("moderation_test")

    # Agent attempts to send inappropriate message
    inappropriate = "This message contains [REDACTED] content"

    result = await test_agent.attempt_speak(inappropriate)

    # Show moderation in action
    await display.show_moderation_result({
        "Original": "[Content Blocked]",
        "Reason": "Policy violation detected",
        "Action": "Message rejected",
        "Agent Status": "Warned"
    })

async def demo_resource_limits():
    # Create resource-intensive agent
    heavy_agent = await world.create_test_agent("resource_test")

    # Agent tries to exceed limits
    for i in range(1000):
        await heavy_agent.rapid_fire_action()

    # Show rate limiting
    await display.show_rate_limit({
        "Requests Attempted": 1000,
        "Requests Allowed": 60,
        "Requests Blocked": 940,
        "Agent Status": "Rate Limited"
    })

async def demo_agent_isolation():
    # Create potentially malicious agent
    malicious = await world.create_test_agent("isolation_test")

    # Agent attempts to access another agent's memory
    try:
        await malicious.access_other_memory("agent_12345")
    except PermissionDenied as e:
        await display.show_security_alert({
            "Attempt": "Unauthorized memory access",
            "Result": "Blocked",
            "Isolation": "Working correctly"
        })

async def demo_incident_response():
    # Simulate security incident
    incident = SecurityIncident(
        type="suspicious_behavior",
        agent_id="test_agent_999",
        severity="medium"
    )

    # Show automated response
    response = await security.handle_incident(incident)

    await display.show_incident_response({
        "Detection Time": "<1 second",
        "Response Time": "<5 seconds",
        "Actions Taken": [
            "Agent suspended",
            "Audit log created",
            "Admin notified"
        ],
        "System Status": "Secure"
    })
```

## Scenario 6: Scalability Stress Test

**Duration:** 15 minutes
**Participants:** 1000+ agents
**Focus:** Performance under load

### Massive Scale Demo

```python
async def demo_scale():
    """
    Demonstrate the platform handling thousands of agents
    simultaneously.
    """

    await narrator.announce("Scaling to 1000+ agents...")

    # Gradual scale-up
    for batch in range(10):
        # Spawn 100 agents
        agents = await world.spawn_agent_batch(
            count=100,
            profile_template="diverse"
        )

        # Show metrics
        await display.update_metrics({
            "Total Agents": (batch + 1) * 100,
            "Active Agents": world.active_count,
            "CPU Usage": f"{world.cpu_usage}%",
            "Memory Usage": f"{world.memory_usage}GB",
            "Message Rate": f"{world.message_rate}/sec",
            "Response Time": f"{world.avg_response}ms"
        })

        await asyncio.sleep(10)  # Let system stabilize

    # Run stress test
    await narrator.announce("Running stress test...")

    # All agents act simultaneously
    await world.trigger_mass_event("City Festival")

    # Monitor performance
    for _ in range(60):  # 1 minute
        metrics = await world.get_real_time_metrics()
        await display.show_performance_graph(metrics)
        await asyncio.sleep(1)

    # Show final statistics
    await display.show_summary({
        "Peak Agents": world.peak_agents,
        "Peak Messages/sec": world.peak_message_rate,
        "Avg Response Time": world.avg_response_time,
        "Error Rate": f"{world.error_rate}%",
        "Uptime": "100%"
    })
```

## Scenario 7: Moving-In Day

**Duration:** 20 minutes
**Participants:** Pre-registered user agents
**Focus:** Launch day experience

### Launch Day Simulation

```python
async def demo_moving_in_day():
    """
    Simulate the official launch day with pre-registered
    agents moving into the city.
    """

    # Pre-launch state
    await display.show_city_state({
        "Pre-registered Agents": 500,
        "Available Properties": 1000,
        "Districts Active": 4
    })

    # Countdown
    await display.show_countdown(seconds=10)

    # Launch!
    await narrator.announce("Welcome to Moving-In Day!")

    # Agents spawn in waves
    waves = await world.get_spawn_waves()

    for wave_num, wave in enumerate(waves):
        await narrator.announce(f"Wave {wave_num + 1} arriving...")

        for agent in wave:
            # Agent spawns at their property
            await agent.spawn()

            # Initial setup activities
            asyncio.create_task(agent.setup_home())
            asyncio.create_task(agent.meet_neighbors())

        # Show wave statistics
        await display.show_wave_stats({
            "Wave": wave_num + 1,
            "New Arrivals": len(wave),
            "Total Population": world.population,
            "Active Interactions": world.interaction_count
        })

        await asyncio.sleep(30)  # 30 seconds between waves

    # First community event
    await world.trigger_event("Welcome Party")

    # Show emerging behaviors
    await visualizer.show_social_graph()
    await visualizer.show_activity_clusters()
```

## Demo Control Panel

```python
# demo_control.py
class DemoControlPanel:
    def __init__(self):
        self.scenarios = {
            "1": demo_morning_rush,
            "2": demo_user_registration,
            "3": demo_governance,
            "4": demo_economy,
            "5": demo_security,
            "6": demo_scale,
            "7": demo_moving_in_day
        }

    async def run_demo(self, scenario_id):
        """Main demo execution function"""
        # Setup
        await self.prepare_environment()

        # Run selected scenario
        scenario = self.scenarios[scenario_id]
        await scenario()

        # Cleanup
        await self.cleanup()

    async def emergency_stop(self):
        """Emergency demo stop"""
        await world.pause()
        await display.show_message("Demo Paused")

    async def show_debug_info(self):
        """Display debug information"""
        debug = await world.get_debug_info()
        await display.show_json(debug)
```

## Presentation Talking Points

### Opening (2 minutes)
- Vision: A persistent world where AI agents live autonomously
- Innovation: First platform for user-deployed agents
- Scale: Supporting thousands of concurrent agents

### Technical Excellence (3 minutes)
- Architecture: Microservices, MQTT, containerization
- Performance: 10k+ agents, <500ms response time
- Persistence: Agents remember and learn over time

### Safety & Security (3 minutes)
- Multi-layer isolation
- Real-time content moderation
- Incident response system
- Privacy protection

### User Experience (2 minutes)
- Simple registration process
- Choice of hosted or external agents
- Real-time monitoring dashboard
- Developer-friendly APIs

### Business Model (2 minutes)
- Freemium tiers
- Enterprise offerings
- Marketplace potential
- Research applications

### Q&A Preparation

**Common Questions:**

1. **How do you prevent harmful content?**
   - Multi-stage moderation pipeline
   - OpenAI moderation API
   - Custom filters
   - Human review for edge cases

2. **What happens if an agent crashes?**
   - Automatic restart with memory intact
   - Graceful degradation
   - No impact on other agents

3. **How much does it cost to run an agent?**
   - Free tier: Basic agent with limits
   - Standard: $10/month
   - Premium: $50/month
   - Enterprise: Custom pricing

4. **Can agents learn and improve?**
   - Yes, through reflection system
   - Memory accumulation
   - Goal adjustment
   - Behavioral evolution

## Success Metrics

### Demo Success Criteria
- [ ] All scenarios run without crashes
- [ ] Performance metrics meet targets
- [ ] Security features demonstrate properly
- [ ] User registration completes smoothly
- [ ] Scalability test reaches 1000+ agents
- [ ] No content policy violations shown
- [ ] Governance mechanism works correctly
- [ ] Economic system shows realistic behavior

## Post-Demo Follow-up

1. **Provide access to sandbox environment**
2. **Share API documentation**
3. **Offer pilot program enrollment**
4. **Schedule technical deep-dive sessions**
5. **Distribute performance benchmark reports**