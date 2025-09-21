import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from packages.shared_types.models import Transaction, Agent

logger = logging.getLogger(__name__)


class EconomySystem:
    """Economic system with currency, balance tracking, and transactions"""

    def __init__(self, database, redis_client):
        self.db = database
        self.redis = redis_client

        # Economic parameters
        self.currency_name = "credits"
        self.initial_balance = 1000.0
        self.transaction_fee = 0.01  # 1% fee
        self.daily_basic_income = 50.0

        # Market parameters
        self.base_prices = {
            "bread": 5.0,
            "coffee": 3.0,
            "pastry": 4.0,
            "meal": 15.0,
            "rent_daily": 50.0,
            "service_hour": 20.0
        }

        # Economic metrics
        self.metrics = {
            "total_supply": 0,
            "total_transactions": 0,
            "transaction_volume": 0.0,
            "average_balance": 0.0,
            "gdp": 0.0
        }

    async def initialize(self):
        """Initialize the economic system"""
        logger.info("Initializing economic system...")

        # Load or create economic state
        await self._load_economic_state()

        # Start background tasks
        asyncio.create_task(self._economic_cycle())
        asyncio.create_task(self._update_metrics())

        logger.info("Economic system initialized")

    async def _load_economic_state(self):
        """Load economic state from database"""
        try:
            # Load total money supply
            supply = await self.redis.get("economy:total_supply")
            if supply:
                self.metrics["total_supply"] = float(supply)
            else:
                self.metrics["total_supply"] = 0

            # Load transaction count
            count = await self.redis.get("economy:transaction_count")
            if count:
                self.metrics["total_transactions"] = int(count)

        except Exception as e:
            logger.error(f"Error loading economic state: {e}")

    async def create_agent_account(self, agent_id: UUID) -> bool:
        """Create economic account for new agent"""
        try:
            # Set initial balance
            balance_key = f"balance:{agent_id}"
            await self.redis.set(balance_key, self.initial_balance)

            # Update total supply
            self.metrics["total_supply"] += self.initial_balance
            await self.redis.set("economy:total_supply", self.metrics["total_supply"])

            # Initialize transaction history
            await self.redis.lpush(f"transactions:{agent_id}", "account_created")

            logger.info(f"Created economic account for agent {agent_id} with {self.initial_balance} {self.currency_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating agent account: {e}")
            return False

    async def get_balance(self, agent_id: UUID) -> float:
        """Get agent's current balance"""
        try:
            balance_key = f"balance:{agent_id}"
            balance = await self.redis.get(balance_key)

            if balance is None:
                # Create account if doesn't exist
                await self.create_agent_account(agent_id)
                return self.initial_balance

            return float(balance)

        except Exception as e:
            logger.error(f"Error getting balance for agent {agent_id}: {e}")
            return 0.0

    async def process_transaction(
        self,
        sender_id: UUID,
        receiver_id: UUID,
        amount: float,
        transaction_type: str = "payment",
        metadata: Optional[Dict] = None
    ) -> Optional[Transaction]:
        """Process a transaction between agents"""
        try:
            # Validate amount
            if amount <= 0:
                logger.warning(f"Invalid transaction amount: {amount}")
                return None

            # Get sender balance
            sender_balance = await self.get_balance(sender_id)

            # Calculate total with fee
            fee = amount * self.transaction_fee
            total_amount = amount + fee

            # Check sufficient funds
            if sender_balance < total_amount:
                logger.warning(f"Insufficient funds: {sender_id} has {sender_balance}, needs {total_amount}")
                return None

            # Create transaction record
            transaction = Transaction(
                id=uuid4(),
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount=amount,
                currency=self.currency_name,
                type=transaction_type,
                metadata=metadata or {},
                status="pending"
            )

            # Execute transaction atomically
            success = await self._execute_transaction(transaction, fee)

            if success:
                transaction.status = "completed"
                await self.db.save_transaction(transaction)

                # Update metrics
                self.metrics["total_transactions"] += 1
                self.metrics["transaction_volume"] += amount
                await self.redis.incr("economy:transaction_count")

                logger.info(f"Transaction {transaction.id}: {sender_id} -> {receiver_id}, {amount} {self.currency_name}")
                return transaction
            else:
                transaction.status = "failed"
                return None

        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return None

    async def _execute_transaction(self, transaction: Transaction, fee: float) -> bool:
        """Execute the actual balance transfers"""
        try:
            sender_key = f"balance:{transaction.sender_id}"
            receiver_key = f"balance:{transaction.receiver_id}"
            treasury_key = "balance:treasury"

            # Use Redis pipeline for atomicity
            pipe = self.redis.pipeline()

            # Get current balances
            sender_balance = float(await self.redis.get(sender_key) or 0)
            receiver_balance = float(await self.redis.get(receiver_key) or 0)
            treasury_balance = float(await self.redis.get(treasury_key) or 0)

            # Calculate new balances
            new_sender = sender_balance - (transaction.amount + fee)
            new_receiver = receiver_balance + transaction.amount
            new_treasury = treasury_balance + fee

            # Verify sender has enough
            if new_sender < 0:
                return False

            # Update balances
            pipe.set(sender_key, new_sender)
            pipe.set(receiver_key, new_receiver)
            pipe.set(treasury_key, new_treasury)

            # Record in transaction history
            pipe.lpush(f"transactions:{transaction.sender_id}",
                      f"-{transaction.amount + fee}")
            pipe.lpush(f"transactions:{transaction.receiver_id}",
                      f"+{transaction.amount}")

            # Execute pipeline
            await pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Error executing transaction: {e}")
            return False

    async def distribute_basic_income(self):
        """Distribute daily basic income to all active agents"""
        try:
            # Get all active agents
            agents = await self.db.get_all_agents()
            active_agents = [a for a in agents if a.state.status == "active"]

            for agent in active_agents:
                balance_key = f"balance:{agent.id}"
                current = float(await self.redis.get(balance_key) or 0)
                new_balance = current + self.daily_basic_income

                await self.redis.set(balance_key, new_balance)

                # Update supply
                self.metrics["total_supply"] += self.daily_basic_income

            total_distributed = len(active_agents) * self.daily_basic_income
            logger.info(f"Distributed {total_distributed} {self.currency_name} as basic income to {len(active_agents)} agents")

        except Exception as e:
            logger.error(f"Error distributing basic income: {e}")

    async def get_market_price(self, item: str) -> float:
        """Get current market price for an item"""
        try:
            # Check for dynamic pricing
            dynamic_price_key = f"market_price:{item}"
            dynamic_price = await self.redis.get(dynamic_price_key)

            if dynamic_price:
                return float(dynamic_price)

            # Return base price
            return self.base_prices.get(item, 10.0)

        except Exception as e:
            logger.error(f"Error getting market price for {item}: {e}")
            return 10.0

    async def update_market_price(self, item: str, new_price: float):
        """Update market price for an item"""
        try:
            dynamic_price_key = f"market_price:{item}"
            await self.redis.set(dynamic_price_key, new_price)

            logger.info(f"Updated market price for {item}: {new_price} {self.currency_name}")

        except Exception as e:
            logger.error(f"Error updating market price: {e}")

    async def simulate_market_dynamics(self):
        """Simulate supply and demand dynamics"""
        try:
            # Get transaction history for market analysis
            recent_transactions = await self.db.get_recent_transactions(hours=24)

            # Count item transactions
            item_counts = {}
            for transaction in recent_transactions:
                if "item" in transaction.metadata:
                    item = transaction.metadata["item"]
                    item_counts[item] = item_counts.get(item, 0) + 1

            # Adjust prices based on demand
            for item, base_price in self.base_prices.items():
                demand = item_counts.get(item, 0)

                # Simple demand-based pricing
                if demand > 20:  # High demand
                    new_price = base_price * 1.2
                elif demand > 10:  # Medium demand
                    new_price = base_price * 1.1
                elif demand < 5:  # Low demand
                    new_price = base_price * 0.9
                else:
                    new_price = base_price

                await self.update_market_price(item, new_price)

            logger.info("Market prices adjusted based on demand")

        except Exception as e:
            logger.error(f"Error simulating market dynamics: {e}")

    async def calculate_agent_wealth(self, agent_id: UUID) -> Dict[str, float]:
        """Calculate total wealth of an agent"""
        try:
            # Get liquid balance
            balance = await self.get_balance(agent_id)

            # Get property values
            properties = await self.db.get_agent_properties(agent_id)
            property_value = sum(p.price for p in properties)

            # Get inventory value (simplified)
            inventory_value = 0  # TODO: Implement inventory valuation

            total_wealth = balance + property_value + inventory_value

            return {
                "balance": balance,
                "property_value": property_value,
                "inventory_value": inventory_value,
                "total_wealth": total_wealth
            }

        except Exception as e:
            logger.error(f"Error calculating agent wealth: {e}")
            return {"balance": 0, "property_value": 0, "inventory_value": 0, "total_wealth": 0}

    async def get_economic_metrics(self) -> Dict:
        """Get current economic metrics"""
        try:
            # Calculate average balance
            all_balances = []
            pattern = "balance:*"

            # Get all balance keys (simplified for demo)
            agents = await self.db.get_all_agents()
            for agent in agents:
                balance = await self.get_balance(agent.id)
                all_balances.append(balance)

            if all_balances:
                self.metrics["average_balance"] = sum(all_balances) / len(all_balances)

            # Calculate GDP (simplified as transaction volume)
            self.metrics["gdp"] = self.metrics["transaction_volume"]

            # Add current prices
            current_prices = {}
            for item in self.base_prices.keys():
                current_prices[item] = await self.get_market_price(item)

            return {
                **self.metrics,
                "current_prices": current_prices,
                "active_accounts": len(all_balances),
                "currency": self.currency_name
            }

        except Exception as e:
            logger.error(f"Error getting economic metrics: {e}")
            return self.metrics

    async def _economic_cycle(self):
        """Background task for economic cycles"""
        while True:
            try:
                # Daily economic activities
                await asyncio.sleep(3600)  # Every hour (accelerated day)

                # Distribute basic income
                await self.distribute_basic_income()

                # Simulate market dynamics
                await self.simulate_market_dynamics()

                # Clean old transaction history
                await self._cleanup_old_transactions()

            except Exception as e:
                logger.error(f"Error in economic cycle: {e}")
                await asyncio.sleep(60)

    async def _update_metrics(self):
        """Background task to update economic metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Get and store metrics
                metrics = await self.get_economic_metrics()
                await self.redis.hset("economy:metrics", mapping=metrics)

                logger.debug(f"Economic metrics updated: {metrics}")

            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_transactions(self):
        """Clean up old transaction history"""
        try:
            # Keep only last 1000 transactions per agent
            agents = await self.db.get_all_agents()

            for agent in agents:
                key = f"transactions:{agent.id}"
                await self.redis.ltrim(key, 0, 999)

            logger.debug("Cleaned up old transaction history")

        except Exception as e:
            logger.error(f"Error cleaning up transactions: {e}")

    async def handle_shop_transaction(
        self,
        customer_id: UUID,
        shop_owner_id: UUID,
        item: str,
        quantity: int = 1
    ) -> Optional[Transaction]:
        """Handle a shop purchase transaction"""
        try:
            # Get item price
            unit_price = await self.get_market_price(item)
            total_price = unit_price * quantity

            # Process transaction
            transaction = await self.process_transaction(
                sender_id=customer_id,
                receiver_id=shop_owner_id,
                amount=total_price,
                transaction_type="purchase",
                metadata={
                    "item": item,
                    "quantity": quantity,
                    "unit_price": unit_price
                }
            )

            if transaction:
                logger.info(f"Shop transaction: {customer_id} bought {quantity} {item} from {shop_owner_id} for {total_price}")

            return transaction

        except Exception as e:
            logger.error(f"Error handling shop transaction: {e}")
            return None