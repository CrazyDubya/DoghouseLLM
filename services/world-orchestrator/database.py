import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Integer, JSON, Boolean, Float, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

from packages.shared_types.models import Agent, Event, District, Property, Transaction

logger = logging.getLogger(__name__)

Base = declarative_base()


class AgentTable(Base):
    __tablename__ = "agents"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(PGUUID(as_uuid=True), nullable=False)
    profile = Column(JSON, nullable=False)
    state = Column(JSON, nullable=False)
    model_config = Column(JSON, nullable=False)
    external_endpoint = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventTable(Base):
    __tablename__ = "events"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    type = Column(String, nullable=False)
    agent_id = Column(PGUUID(as_uuid=True), nullable=True)
    data = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    district = Column(String, nullable=True)
    processed = Column(Boolean, default=False)


class DistrictTable(Base):
    __tablename__ = "districts"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    neighborhoods = Column(JSON, nullable=False)
    population = Column(Integer, default=0)
    governance = Column(JSON, nullable=False)
    economy = Column(JSON, nullable=False)


class PropertyTable(Base):
    __tablename__ = "properties"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    type = Column(String, nullable=False)
    district = Column(String, nullable=False)
    neighborhood = Column(String, nullable=False)
    address = Column(String, nullable=False)
    owner_id = Column(PGUUID(as_uuid=True), nullable=True)
    lease_terms = Column(JSON, nullable=True)
    features = Column(JSON, nullable=False, default=list)
    size = Column(Float, nullable=False)
    price = Column(Float, nullable=False)


class TransactionTable(Base):
    __tablename__ = "transactions"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    sender_id = Column(PGUUID(as_uuid=True), nullable=False)
    receiver_id = Column(PGUUID(as_uuid=True), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="credits")
    type = Column(String, nullable=False)
    metadata = Column(JSON, nullable=False, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")


class MemoryTable(Base):
    __tablename__ = "memories"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    agent_id = Column(PGUUID(as_uuid=True), nullable=False)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    importance = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    participants = Column(JSON, nullable=False, default=list)
    emotions = Column(JSON, nullable=False, default=list)
    tags = Column(JSON, nullable=False, default=list)


class Database:
    """Database manager for world state persistence"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None

    async def initialize(self):
        """Initialize database connection and create tables"""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_size=20,
            max_overflow=30
        )

        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    async def is_healthy(self) -> bool:
        """Check database health"""
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def save_agent(self, agent: Agent) -> Agent:
        """Save or update an agent"""
        async with self.session_factory() as session:
            # Check if agent exists
            result = await session.execute(
                select(AgentTable).where(AgentTable.id == agent.id)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.name = agent.name
                existing.profile = agent.profile.dict()
                existing.state = agent.state.dict()
                existing.model_config = agent.model_config
                existing.external_endpoint = agent.external_endpoint
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                agent_table = AgentTable(
                    id=agent.id,
                    name=agent.name,
                    user_id=agent.user_id,
                    profile=agent.profile.dict(),
                    state=agent.state.dict(),
                    model_config=agent.model_config,
                    external_endpoint=agent.external_endpoint
                )
                session.add(agent_table)

            await session.commit()
            return agent

    async def get_agent(self, agent_id: UUID) -> Optional[Agent]:
        """Get an agent by ID"""
        async with self.session_factory() as session:
            result = await session.execute(
                select(AgentTable).where(AgentTable.id == agent_id)
            )
            agent_table = result.scalar_one_or_none()

            if agent_table:
                return self._table_to_agent(agent_table)
            return None

    async def get_all_agents(self) -> List[Agent]:
        """Get all agents"""
        async with self.session_factory() as session:
            result = await session.execute(select(AgentTable))
            agent_tables = result.scalars().all()

            return [self._table_to_agent(at) for at in agent_tables]

    def _table_to_agent(self, agent_table: AgentTable) -> Agent:
        """Convert database table to Agent model"""
        from packages.shared_types.models import AgentProfile, AgentState

        return Agent(
            id=agent_table.id,
            name=agent_table.name,
            user_id=agent_table.user_id,
            profile=AgentProfile(**agent_table.profile),
            state=AgentState(**agent_table.state),
            model_config=agent_table.model_config,
            external_endpoint=agent_table.external_endpoint,
            created_at=agent_table.created_at,
            updated_at=agent_table.updated_at
        )

    async def save_event(self, event: Event) -> Event:
        """Save an event"""
        async with self.session_factory() as session:
            event_table = EventTable(
                id=event.id,
                type=event.type,
                agent_id=event.agent_id,
                data=event.data,
                timestamp=event.timestamp,
                district=event.district,
                processed=event.processed
            )
            session.add(event_table)
            await session.commit()
            return event

    async def get_unprocessed_events(self, limit: int = 100) -> List[Event]:
        """Get unprocessed events"""
        async with self.session_factory() as session:
            result = await session.execute(
                select(EventTable)
                .where(EventTable.processed == False)
                .order_by(EventTable.timestamp)
                .limit(limit)
            )
            event_tables = result.scalars().all()

            return [self._table_to_event(et) for et in event_tables]

    def _table_to_event(self, event_table: EventTable) -> Event:
        """Convert database table to Event model"""
        return Event(
            id=event_table.id,
            type=event_table.type,
            agent_id=event_table.agent_id,
            data=event_table.data,
            timestamp=event_table.timestamp,
            district=event_table.district,
            processed=event_table.processed
        )

    async def save_district(self, district: District) -> District:
        """Save or update a district"""
        async with self.session_factory() as session:
            # Check if district exists
            result = await session.execute(
                select(DistrictTable).where(DistrictTable.name == district.name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.neighborhoods = district.neighborhoods
                existing.population = district.population
                existing.governance = district.governance
                existing.economy = district.economy
            else:
                # Create new
                district_table = DistrictTable(
                    id=district.id,
                    name=district.name,
                    neighborhoods=district.neighborhoods,
                    population=district.population,
                    governance=district.governance,
                    economy=district.economy
                )
                session.add(district_table)

            await session.commit()
            return district

    async def get_districts(self) -> List[District]:
        """Get all districts"""
        async with self.session_factory() as session:
            result = await session.execute(select(DistrictTable))
            district_tables = result.scalars().all()

            return [self._table_to_district(dt) for dt in district_tables]

    def _table_to_district(self, district_table: DistrictTable) -> District:
        """Convert database table to District model"""
        return District(
            id=district_table.id,
            name=district_table.name,
            neighborhoods=district_table.neighborhoods,
            population=district_table.population,
            governance=district_table.governance,
            economy=district_table.economy
        )

    async def save_property(self, property: Property) -> Property:
        """Save a property"""
        async with self.session_factory() as session:
            property_table = PropertyTable(
                id=property.id,
                type=property.type,
                district=property.district,
                neighborhood=property.neighborhood,
                address=property.address,
                owner_id=property.owner_id,
                lease_terms=property.lease_terms,
                features=property.features,
                size=property.size,
                price=property.price
            )
            session.add(property_table)
            await session.commit()
            return property

    async def get_available_properties(self, district: Optional[str] = None) -> List[Property]:
        """Get available properties"""
        async with self.session_factory() as session:
            query = select(PropertyTable).where(PropertyTable.owner_id.is_(None))

            if district:
                query = query.where(PropertyTable.district == district)

            result = await session.execute(query)
            property_tables = result.scalars().all()

            return [self._table_to_property(pt) for pt in property_tables]

    def _table_to_property(self, property_table: PropertyTable) -> Property:
        """Convert database table to Property model"""
        return Property(
            id=property_table.id,
            type=property_table.type,
            district=property_table.district,
            neighborhood=property_table.neighborhood,
            address=property_table.address,
            owner_id=property_table.owner_id,
            lease_terms=property_table.lease_terms,
            features=property_table.features,
            size=property_table.size,
            price=property_table.price
        )

    async def save_transaction(self, transaction: Transaction) -> Transaction:
        """Save a transaction"""
        async with self.session_factory() as session:
            transaction_table = TransactionTable(
                id=transaction.id,
                sender_id=transaction.sender_id,
                receiver_id=transaction.receiver_id,
                amount=transaction.amount,
                currency=transaction.currency,
                type=transaction.type,
                metadata=transaction.metadata,
                timestamp=transaction.timestamp,
                status=transaction.status
            )
            session.add(transaction_table)
            await session.commit()
            return transaction

    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()