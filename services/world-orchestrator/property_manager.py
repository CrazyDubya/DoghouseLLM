import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from packages.shared_types.models import Property, Agent

logger = logging.getLogger(__name__)


class PropertyManager:
    """Property ownership and management system"""

    def __init__(self, database, redis_client, economy_system):
        self.db = database
        self.redis = redis_client
        self.economy = economy_system

        # Property configuration
        self.lease_durations = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "permanent": 365 * 10  # 10 years as "permanent"
        }

    async def initialize(self):
        """Initialize property manager"""
        logger.info("Initializing property manager...")

        # Load existing properties from database
        await self._load_properties()

        # Start background tasks
        asyncio.create_task(self._lease_monitor())

        logger.info("Property manager initialized")

    async def _load_properties(self):
        """Load properties from database"""
        try:
            properties = await self.db.get_available_properties()
            logger.info(f"Loaded {len(properties)} available properties")

            # Cache property data in Redis for quick access
            for prop in properties:
                await self._cache_property(prop)

        except Exception as e:
            logger.error(f"Error loading properties: {e}")

    async def _cache_property(self, property: Property):
        """Cache property data in Redis"""
        try:
            property_key = f"property:{property.id}"
            property_data = {
                "id": str(property.id),
                "type": property.type,
                "district": property.district,
                "neighborhood": property.neighborhood,
                "owner_id": str(property.owner_id) if property.owner_id else "",
                "price": property.price,
                "available": "true" if property.owner_id is None else "false"
            }

            await self.redis.hset(property_key, mapping=property_data)

            # Add to district index
            district_key = f"properties:district:{property.district}"
            await self.redis.sadd(district_key, str(property.id))

            # Add to available properties set if no owner
            if property.owner_id is None:
                await self.redis.sadd("properties:available", str(property.id))

        except Exception as e:
            logger.error(f"Error caching property: {e}")

    async def get_available_properties(
        self,
        district: Optional[str] = None,
        property_type: Optional[str] = None,
        max_price: Optional[float] = None
    ) -> List[Property]:
        """Get list of available properties"""
        try:
            # Get from database with filters
            properties = await self.db.get_available_properties(district)

            # Apply additional filters
            filtered = []
            for prop in properties:
                if property_type and prop.type != property_type:
                    continue
                if max_price and prop.price > max_price:
                    continue
                filtered.append(prop)

            return filtered

        except Exception as e:
            logger.error(f"Error getting available properties: {e}")
            return []

    async def claim_property(
        self,
        agent_id: UUID,
        property_id: UUID,
        lease_type: str = "monthly",
        auto_renew: bool = True
    ) -> Optional[Property]:
        """Claim a property for an agent"""
        try:
            # Get property from database
            property = await self.db.get_property(property_id)

            if not property:
                logger.warning(f"Property {property_id} not found")
                return None

            if property.owner_id:
                logger.warning(f"Property {property_id} already owned by {property.owner_id}")
                return None

            # Calculate lease cost
            duration_days = self.lease_durations.get(lease_type, 30)
            total_cost = property.price * duration_days

            # Check agent balance
            agent_balance = await self.economy.get_balance(agent_id)
            if agent_balance < total_cost:
                logger.warning(f"Agent {agent_id} has insufficient funds for property {property_id}")
                return None

            # Process payment to treasury (or previous owner if applicable)
            treasury_id = UUID("00000000-0000-0000-0000-000000000000")  # Special treasury ID
            transaction = await self.economy.process_transaction(
                sender_id=agent_id,
                receiver_id=treasury_id,
                amount=total_cost,
                transaction_type="lease",
                metadata={
                    "property_id": str(property_id),
                    "lease_type": lease_type,
                    "duration_days": duration_days
                }
            )

            if not transaction:
                logger.warning(f"Property lease payment failed for {property_id}")
                return None

            # Update property ownership
            property.owner_id = agent_id
            property.lease_terms = {
                "type": lease_type,
                "start_date": datetime.utcnow().isoformat(),
                "end_date": (datetime.utcnow() + timedelta(days=duration_days)).isoformat(),
                "auto_renew": auto_renew,
                "daily_rate": property.price
            }

            # Save to database
            await self.db.save_property(property)

            # Update Redis cache
            await self._cache_property(property)
            await self.redis.srem("properties:available", str(property_id))

            # Record ownership in agent data
            agent_properties_key = f"agent_properties:{agent_id}"
            await self.redis.sadd(agent_properties_key, str(property_id))

            logger.info(f"Property {property_id} claimed by agent {agent_id} for {duration_days} days")
            return property

        except Exception as e:
            logger.error(f"Error claiming property: {e}")
            return None

    async def release_property(self, property_id: UUID) -> bool:
        """Release a property (make it available again)"""
        try:
            # Get property
            property = await self.db.get_property(property_id)

            if not property:
                logger.warning(f"Property {property_id} not found")
                return False

            if property.owner_id:
                # Remove from agent's property list
                agent_properties_key = f"agent_properties:{property.owner_id}"
                await self.redis.srem(agent_properties_key, str(property_id))

            # Clear ownership
            property.owner_id = None
            property.lease_terms = None

            # Save to database
            await self.db.save_property(property)

            # Update Redis cache
            await self._cache_property(property)
            await self.redis.sadd("properties:available", str(property_id))

            logger.info(f"Property {property_id} released and made available")
            return True

        except Exception as e:
            logger.error(f"Error releasing property: {e}")
            return False

    async def transfer_property(
        self,
        property_id: UUID,
        from_agent_id: UUID,
        to_agent_id: UUID,
        sale_price: Optional[float] = None
    ) -> bool:
        """Transfer property ownership between agents"""
        try:
            # Get property
            property = await self.db.get_property(property_id)

            if not property:
                logger.warning(f"Property {property_id} not found")
                return False

            if property.owner_id != from_agent_id:
                logger.warning(f"Agent {from_agent_id} does not own property {property_id}")
                return False

            # If sale price specified, process transaction
            if sale_price:
                transaction = await self.economy.process_transaction(
                    sender_id=to_agent_id,
                    receiver_id=from_agent_id,
                    amount=sale_price,
                    transaction_type="property_sale",
                    metadata={
                        "property_id": str(property_id),
                        "property_type": property.type,
                        "location": f"{property.district}/{property.neighborhood}"
                    }
                )

                if not transaction:
                    logger.warning(f"Property sale transaction failed")
                    return False

            # Update ownership
            property.owner_id = to_agent_id

            # Save to database
            await self.db.save_property(property)

            # Update Redis cache
            await self._cache_property(property)

            # Update agent property lists
            from_agent_key = f"agent_properties:{from_agent_id}"
            to_agent_key = f"agent_properties:{to_agent_id}"
            await self.redis.srem(from_agent_key, str(property_id))
            await self.redis.sadd(to_agent_key, str(property_id))

            logger.info(f"Property {property_id} transferred from {from_agent_id} to {to_agent_id}")
            return True

        except Exception as e:
            logger.error(f"Error transferring property: {e}")
            return False

    async def get_agent_properties(self, agent_id: UUID) -> List[Property]:
        """Get all properties owned by an agent"""
        try:
            # Get property IDs from Redis
            agent_properties_key = f"agent_properties:{agent_id}"
            property_ids = await self.redis.smembers(agent_properties_key)

            properties = []
            for prop_id in property_ids:
                prop_id_str = prop_id.decode() if isinstance(prop_id, bytes) else prop_id
                property = await self.db.get_property(UUID(prop_id_str))
                if property:
                    properties.append(property)

            return properties

        except Exception as e:
            logger.error(f"Error getting agent properties: {e}")
            return []

    async def calculate_property_tax(self, property: Property) -> float:
        """Calculate property tax for a property"""
        try:
            # Simple tax calculation: 1% of property value per month
            monthly_tax = property.price * 0.01

            # Adjust based on district
            district_multipliers = {
                "Downtown": 1.5,
                "Market Square": 1.2,
                "Tech Hub": 1.3,
                "Residential": 1.0
            }

            multiplier = district_multipliers.get(property.district, 1.0)
            return monthly_tax * multiplier

        except Exception as e:
            logger.error(f"Error calculating property tax: {e}")
            return 0.0

    async def collect_property_taxes(self):
        """Collect property taxes from all property owners"""
        try:
            # Get all properties with owners
            all_properties = await self.db.get_all_properties()
            owned_properties = [p for p in all_properties if p.owner_id]

            treasury_id = UUID("00000000-0000-0000-0000-000000000000")
            total_collected = 0.0

            for property in owned_properties:
                tax_amount = await self.calculate_property_tax(property)

                # Try to collect tax
                transaction = await self.economy.process_transaction(
                    sender_id=property.owner_id,
                    receiver_id=treasury_id,
                    amount=tax_amount,
                    transaction_type="property_tax",
                    metadata={
                        "property_id": str(property.id),
                        "property_type": property.type,
                        "district": property.district
                    }
                )

                if transaction:
                    total_collected += tax_amount
                else:
                    # Handle tax delinquency
                    logger.warning(f"Agent {property.owner_id} failed to pay tax for property {property.id}")
                    # Could implement property seizure here

            logger.info(f"Collected {total_collected} in property taxes")

        except Exception as e:
            logger.error(f"Error collecting property taxes: {e}")

    async def _lease_monitor(self):
        """Monitor property leases and handle expirations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Get all properties with leases
                all_properties = await self.db.get_all_properties()
                leased_properties = [p for p in all_properties if p.owner_id and p.lease_terms]

                for property in leased_properties:
                    lease_terms = property.lease_terms
                    end_date = datetime.fromisoformat(lease_terms["end_date"])

                    if datetime.utcnow() > end_date:
                        if lease_terms.get("auto_renew"):
                            # Auto-renew lease
                            await self._renew_lease(property)
                        else:
                            # Lease expired, release property
                            await self.release_property(property.id)

            except Exception as e:
                logger.error(f"Error in lease monitor: {e}")
                await asyncio.sleep(60)

    async def _renew_lease(self, property: Property):
        """Renew a property lease"""
        try:
            if not property.owner_id or not property.lease_terms:
                return

            # Calculate renewal cost
            lease_type = property.lease_terms.get("type", "monthly")
            duration_days = self.lease_durations.get(lease_type, 30)
            renewal_cost = property.price * duration_days

            # Process renewal payment
            treasury_id = UUID("00000000-0000-0000-0000-000000000000")
            transaction = await self.economy.process_transaction(
                sender_id=property.owner_id,
                receiver_id=treasury_id,
                amount=renewal_cost,
                transaction_type="lease_renewal",
                metadata={
                    "property_id": str(property.id),
                    "lease_type": lease_type
                }
            )

            if transaction:
                # Update lease terms
                property.lease_terms["start_date"] = datetime.utcnow().isoformat()
                property.lease_terms["end_date"] = (datetime.utcnow() + timedelta(days=duration_days)).isoformat()

                # Save to database
                await self.db.save_property(property)

                logger.info(f"Lease renewed for property {property.id}")
            else:
                # Renewal failed, release property
                logger.warning(f"Lease renewal failed for property {property.id}, releasing")
                await self.release_property(property.id)

        except Exception as e:
            logger.error(f"Error renewing lease: {e}")

    async def get_property_metrics(self) -> Dict:
        """Get property system metrics"""
        try:
            all_properties = await self.db.get_all_properties()

            metrics = {
                "total_properties": len(all_properties),
                "available_properties": len([p for p in all_properties if not p.owner_id]),
                "occupied_properties": len([p for p in all_properties if p.owner_id]),
                "average_price": sum(p.price for p in all_properties) / len(all_properties) if all_properties else 0,
                "total_value": sum(p.price * p.size for p in all_properties),
                "by_district": {}
            }

            # Count by district
            for prop in all_properties:
                if prop.district not in metrics["by_district"]:
                    metrics["by_district"][prop.district] = {
                        "total": 0,
                        "available": 0,
                        "occupied": 0
                    }

                metrics["by_district"][prop.district]["total"] += 1
                if prop.owner_id:
                    metrics["by_district"][prop.district]["occupied"] += 1
                else:
                    metrics["by_district"][prop.district]["available"] += 1

            return metrics

        except Exception as e:
            logger.error(f"Error getting property metrics: {e}")
            return {}