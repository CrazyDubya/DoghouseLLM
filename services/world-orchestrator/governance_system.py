"""Governance and voting system for the multi-agent city"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4
from enum import Enum

from packages.shared_types.models import Agent

logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Types of proposals that can be voted on"""
    POLICY_CHANGE = "policy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    DISTRICT_DEVELOPMENT = "district_development"
    TAX_ADJUSTMENT = "tax_adjustment"
    EVENT_ORGANIZATION = "event_organization"
    RULE_MODIFICATION = "rule_modification"
    ECONOMIC_STIMULUS = "economic_stimulus"


class VoteChoice(Enum):
    """Vote choices"""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class ProposalStatus(Enum):
    """Proposal status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"
    IMPLEMENTED = "implemented"


class Proposal:
    """A proposal for governance voting"""

    def __init__(
        self,
        id: UUID,
        proposer_id: UUID,
        type: ProposalType,
        title: str,
        description: str,
        metadata: Dict = None,
        voting_duration_hours: int = 24,
        quorum_percentage: float = 0.1,  # 10% of active agents
        pass_threshold: float = 0.5  # 50% majority
    ):
        self.id = id
        self.proposer_id = proposer_id
        self.type = type
        self.title = title
        self.description = description
        self.metadata = metadata or {}
        self.status = ProposalStatus.DRAFT
        self.created_at = datetime.utcnow()
        self.voting_start = None
        self.voting_end = None
        self.voting_duration_hours = voting_duration_hours
        self.quorum_percentage = quorum_percentage
        self.pass_threshold = pass_threshold

        # Voting data
        self.votes: Dict[UUID, VoteChoice] = {}
        self.vote_weights: Dict[UUID, float] = {}
        self.comments: List[Dict] = []

        # Results
        self.final_results = None
        self.implementation_status = None

    def to_dict(self) -> Dict:
        """Convert proposal to dictionary"""
        return {
            "id": str(self.id),
            "proposer_id": str(self.proposer_id),
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "voting_start": self.voting_start.isoformat() if self.voting_start else None,
            "voting_end": self.voting_end.isoformat() if self.voting_end else None,
            "voting_duration_hours": self.voting_duration_hours,
            "quorum_percentage": self.quorum_percentage,
            "pass_threshold": self.pass_threshold,
            "total_votes": len(self.votes),
            "yes_votes": sum(1 for v in self.votes.values() if v == VoteChoice.YES),
            "no_votes": sum(1 for v in self.votes.values() if v == VoteChoice.NO),
            "abstain_votes": sum(1 for v in self.votes.values() if v == VoteChoice.ABSTAIN),
            "final_results": self.final_results,
            "implementation_status": self.implementation_status
        }


class GovernanceSystem:
    """Governance and voting system for the city"""

    def __init__(self, database, redis_client, economy_system):
        self.db = database
        self.redis = redis_client
        self.economy = economy_system

        # Configuration
        self.min_reputation_to_propose = 100
        self.min_reputation_to_vote = 10
        self.proposal_cost = 100.0  # Credits to submit a proposal
        self.max_active_proposals = 10

        # Active proposals
        self.active_proposals: Dict[UUID, Proposal] = {}
        self.proposal_history: List[Proposal] = []

        # Reputation tracking
        self.agent_reputation: Dict[UUID, float] = {}

        # Council members (agents with special voting weight)
        self.council_members: Set[UUID] = set()
        self.council_vote_multiplier = 5.0

    async def initialize(self):
        """Initialize the governance system"""
        logger.info("Initializing governance system...")

        # Load existing proposals from database
        await self._load_proposals()

        # Load reputation data
        await self._load_reputation()

        # Start background tasks
        asyncio.create_task(self._voting_monitor())
        asyncio.create_task(self._implementation_monitor())

        logger.info("Governance system initialized")

    async def _load_proposals(self):
        """Load proposals from database"""
        try:
            # Load active proposals from Redis
            proposal_keys = await self.redis.keys("proposal:active:*")

            for key in proposal_keys:
                proposal_data = await self.redis.hgetall(key)
                if proposal_data:
                    proposal = await self._deserialize_proposal(proposal_data)
                    if proposal:
                        self.active_proposals[proposal.id] = proposal

            logger.info(f"Loaded {len(self.active_proposals)} active proposals")

        except Exception as e:
            logger.error(f"Error loading proposals: {e}")

    async def _load_reputation(self):
        """Load agent reputation data"""
        try:
            # Load reputation scores from Redis
            reputation_keys = await self.redis.keys("reputation:*")

            for key in reputation_keys:
                agent_id_str = key.decode().split(":")[1] if isinstance(key, bytes) else key.split(":")[1]
                reputation_score = await self.redis.get(key)

                if reputation_score:
                    self.agent_reputation[UUID(agent_id_str)] = float(reputation_score)

            logger.info(f"Loaded reputation for {len(self.agent_reputation)} agents")

        except Exception as e:
            logger.error(f"Error loading reputation: {e}")

    async def create_proposal(
        self,
        proposer_id: UUID,
        proposal_type: ProposalType,
        title: str,
        description: str,
        metadata: Dict = None,
        voting_duration_hours: int = 24
    ) -> Optional[Proposal]:
        """Create a new proposal"""
        try:
            # Check proposer reputation
            proposer_reputation = await self.get_reputation(proposer_id)
            if proposer_reputation < self.min_reputation_to_propose:
                logger.warning(f"Agent {proposer_id} has insufficient reputation to propose")
                return None

            # Check proposal limit
            if len(self.active_proposals) >= self.max_active_proposals:
                logger.warning("Maximum active proposals reached")
                return None

            # Charge proposal fee
            if self.economy:
                treasury_id = UUID("00000000-0000-0000-0000-000000000000")
                transaction = await self.economy.process_transaction(
                    sender_id=proposer_id,
                    receiver_id=treasury_id,
                    amount=self.proposal_cost,
                    transaction_type="proposal_fee",
                    metadata={"proposal_title": title}
                )

                if not transaction:
                    logger.warning(f"Agent {proposer_id} cannot afford proposal fee")
                    return None

            # Create proposal
            proposal = Proposal(
                id=uuid4(),
                proposer_id=proposer_id,
                type=proposal_type,
                title=title,
                description=description,
                metadata=metadata,
                voting_duration_hours=voting_duration_hours
            )

            # Save to Redis
            await self._save_proposal(proposal)

            logger.info(f"Proposal {proposal.id} created by agent {proposer_id}")
            return proposal

        except Exception as e:
            logger.error(f"Error creating proposal: {e}")
            return None

    async def start_voting(self, proposal_id: UUID) -> bool:
        """Start voting on a proposal"""
        try:
            proposal = self.active_proposals.get(proposal_id)

            if not proposal:
                logger.warning(f"Proposal {proposal_id} not found")
                return False

            if proposal.status != ProposalStatus.DRAFT:
                logger.warning(f"Proposal {proposal_id} is not in draft status")
                return False

            # Start voting period
            proposal.status = ProposalStatus.ACTIVE
            proposal.voting_start = datetime.utcnow()
            proposal.voting_end = proposal.voting_start + timedelta(hours=proposal.voting_duration_hours)

            # Save updated proposal
            await self._save_proposal(proposal)

            # Notify agents via MQTT
            await self._notify_voting_started(proposal)

            logger.info(f"Voting started for proposal {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting voting: {e}")
            return False

    async def cast_vote(
        self,
        proposal_id: UUID,
        voter_id: UUID,
        vote: VoteChoice,
        comment: Optional[str] = None
    ) -> bool:
        """Cast a vote on a proposal"""
        try:
            proposal = self.active_proposals.get(proposal_id)

            if not proposal:
                logger.warning(f"Proposal {proposal_id} not found")
                return False

            if proposal.status != ProposalStatus.ACTIVE:
                logger.warning(f"Proposal {proposal_id} is not active for voting")
                return False

            # Check if voting period is still open
            if datetime.utcnow() > proposal.voting_end:
                logger.warning(f"Voting period for proposal {proposal_id} has ended")
                return False

            # Check voter reputation
            voter_reputation = await self.get_reputation(voter_id)
            if voter_reputation < self.min_reputation_to_vote:
                logger.warning(f"Agent {voter_id} has insufficient reputation to vote")
                return False

            # Calculate vote weight
            vote_weight = 1.0
            if voter_id in self.council_members:
                vote_weight *= self.council_vote_multiplier

            # Reputation-weighted voting
            vote_weight *= (1.0 + voter_reputation / 1000.0)

            # Record vote
            proposal.votes[voter_id] = vote
            proposal.vote_weights[voter_id] = vote_weight

            # Add comment if provided
            if comment:
                proposal.comments.append({
                    "voter_id": str(voter_id),
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment": comment
                })

            # Save updated proposal
            await self._save_proposal(proposal)

            # Update voter reputation for participating
            await self.update_reputation(voter_id, 1.0)

            logger.info(f"Agent {voter_id} voted {vote.value} on proposal {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Error casting vote: {e}")
            return False

    async def tally_votes(self, proposal_id: UUID) -> Dict:
        """Tally votes for a proposal"""
        try:
            proposal = self.active_proposals.get(proposal_id)

            if not proposal:
                logger.warning(f"Proposal {proposal_id} not found")
                return {}

            # Calculate weighted vote totals
            weighted_yes = sum(
                proposal.vote_weights[voter_id]
                for voter_id, vote in proposal.votes.items()
                if vote == VoteChoice.YES
            )

            weighted_no = sum(
                proposal.vote_weights[voter_id]
                for voter_id, vote in proposal.votes.items()
                if vote == VoteChoice.NO
            )

            weighted_abstain = sum(
                proposal.vote_weights[voter_id]
                for voter_id, vote in proposal.votes.items()
                if vote == VoteChoice.ABSTAIN
            )

            total_weighted = weighted_yes + weighted_no + weighted_abstain

            # Calculate percentages
            results = {
                "yes_votes": len([v for v in proposal.votes.values() if v == VoteChoice.YES]),
                "no_votes": len([v for v in proposal.votes.values() if v == VoteChoice.NO]),
                "abstain_votes": len([v for v in proposal.votes.values() if v == VoteChoice.ABSTAIN]),
                "total_votes": len(proposal.votes),
                "weighted_yes": weighted_yes,
                "weighted_no": weighted_no,
                "weighted_abstain": weighted_abstain,
                "total_weighted": total_weighted,
                "yes_percentage": (weighted_yes / total_weighted * 100) if total_weighted > 0 else 0,
                "no_percentage": (weighted_no / total_weighted * 100) if total_weighted > 0 else 0,
                "abstain_percentage": (weighted_abstain / total_weighted * 100) if total_weighted > 0 else 0
            }

            return results

        except Exception as e:
            logger.error(f"Error tallying votes: {e}")
            return {}

    async def finalize_proposal(self, proposal_id: UUID) -> bool:
        """Finalize a proposal after voting ends"""
        try:
            proposal = self.active_proposals.get(proposal_id)

            if not proposal:
                logger.warning(f"Proposal {proposal_id} not found")
                return False

            # Tally final results
            results = await self.tally_votes(proposal_id)
            proposal.final_results = results

            # Check quorum
            total_agents = await self.db.get_active_agent_count()
            quorum_required = int(total_agents * proposal.quorum_percentage)

            if results["total_votes"] < quorum_required:
                proposal.status = ProposalStatus.FAILED
                logger.info(f"Proposal {proposal_id} failed due to lack of quorum")
            else:
                # Check if proposal passed
                yes_percentage = results["yes_percentage"] / 100.0
                if yes_percentage >= proposal.pass_threshold:
                    proposal.status = ProposalStatus.PASSED
                    logger.info(f"Proposal {proposal_id} passed with {results['yes_percentage']:.1f}% support")

                    # Reward proposer
                    await self.update_reputation(proposal.proposer_id, 10.0)
                else:
                    proposal.status = ProposalStatus.FAILED
                    logger.info(f"Proposal {proposal_id} failed with only {results['yes_percentage']:.1f}% support")

            # Save final state
            await self._save_proposal(proposal)

            # Move to history
            self.proposal_history.append(proposal)
            del self.active_proposals[proposal_id]

            # Implement if passed
            if proposal.status == ProposalStatus.PASSED:
                await self.implement_proposal(proposal)

            return True

        except Exception as e:
            logger.error(f"Error finalizing proposal: {e}")
            return False

    async def implement_proposal(self, proposal: Proposal):
        """Implement a passed proposal"""
        try:
            logger.info(f"Implementing proposal {proposal.id}: {proposal.title}")

            if proposal.type == ProposalType.TAX_ADJUSTMENT:
                # Adjust tax rates
                if "new_tax_rate" in proposal.metadata:
                    await self._adjust_tax_rate(proposal.metadata["new_tax_rate"])

            elif proposal.type == ProposalType.ECONOMIC_STIMULUS:
                # Distribute funds to agents
                if "stimulus_amount" in proposal.metadata:
                    await self._distribute_stimulus(proposal.metadata["stimulus_amount"])

            elif proposal.type == ProposalType.DISTRICT_DEVELOPMENT:
                # Allocate resources to district
                if "district" in proposal.metadata and "resources" in proposal.metadata:
                    await self._allocate_district_resources(
                        proposal.metadata["district"],
                        proposal.metadata["resources"]
                    )

            elif proposal.type == ProposalType.EVENT_ORGANIZATION:
                # Schedule event
                if "event_details" in proposal.metadata:
                    await self._schedule_event(proposal.metadata["event_details"])

            # Mark as implemented
            proposal.status = ProposalStatus.IMPLEMENTED
            proposal.implementation_status = "completed"
            await self._save_proposal(proposal)

        except Exception as e:
            logger.error(f"Error implementing proposal: {e}")
            proposal.implementation_status = f"failed: {str(e)}"
            await self._save_proposal(proposal)

    async def get_reputation(self, agent_id: UUID) -> float:
        """Get agent's reputation score"""
        if agent_id in self.agent_reputation:
            return self.agent_reputation[agent_id]

        # Initialize with base reputation
        base_reputation = 10.0
        self.agent_reputation[agent_id] = base_reputation
        await self.redis.set(f"reputation:{agent_id}", base_reputation)

        return base_reputation

    async def update_reputation(self, agent_id: UUID, change: float):
        """Update agent's reputation"""
        try:
            current = await self.get_reputation(agent_id)
            new_reputation = max(0, current + change)

            self.agent_reputation[agent_id] = new_reputation
            await self.redis.set(f"reputation:{agent_id}", new_reputation)

            logger.debug(f"Updated reputation for agent {agent_id}: {current} -> {new_reputation}")

        except Exception as e:
            logger.error(f"Error updating reputation: {e}")

    async def add_council_member(self, agent_id: UUID):
        """Add an agent to the council"""
        self.council_members.add(agent_id)
        await self.redis.sadd("council_members", str(agent_id))
        logger.info(f"Agent {agent_id} added to council")

    async def remove_council_member(self, agent_id: UUID):
        """Remove an agent from the council"""
        self.council_members.discard(agent_id)
        await self.redis.srem("council_members", str(agent_id))
        logger.info(f"Agent {agent_id} removed from council")

    async def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals"""
        return list(self.active_proposals.values())

    async def get_proposal(self, proposal_id: UUID) -> Optional[Proposal]:
        """Get a specific proposal"""
        if proposal_id in self.active_proposals:
            return self.active_proposals[proposal_id]

        # Check history
        for proposal in self.proposal_history:
            if proposal.id == proposal_id:
                return proposal

        return None

    async def get_agent_voting_history(self, agent_id: UUID) -> List[Dict]:
        """Get voting history for an agent"""
        history = []

        for proposal in self.proposal_history:
            if agent_id in proposal.votes:
                history.append({
                    "proposal_id": str(proposal.id),
                    "title": proposal.title,
                    "vote": proposal.votes[agent_id].value,
                    "weight": proposal.vote_weights.get(agent_id, 1.0),
                    "result": proposal.status.value,
                    "voting_end": proposal.voting_end.isoformat() if proposal.voting_end else None
                })

        return history

    async def _voting_monitor(self):
        """Monitor active proposals and finalize when voting ends"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                proposals_to_finalize = []

                for proposal_id, proposal in self.active_proposals.items():
                    if proposal.status == ProposalStatus.ACTIVE and proposal.voting_end:
                        if now > proposal.voting_end:
                            proposals_to_finalize.append(proposal_id)

                for proposal_id in proposals_to_finalize:
                    await self.finalize_proposal(proposal_id)

            except Exception as e:
                logger.error(f"Error in voting monitor: {e}")
                await asyncio.sleep(60)

    async def _implementation_monitor(self):
        """Monitor and execute proposal implementations"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check for proposals that need implementation
                for proposal in self.proposal_history:
                    if proposal.status == ProposalStatus.PASSED and not proposal.implementation_status:
                        await self.implement_proposal(proposal)

            except Exception as e:
                logger.error(f"Error in implementation monitor: {e}")
                await asyncio.sleep(60)

    async def _save_proposal(self, proposal: Proposal):
        """Save proposal to Redis"""
        try:
            key = f"proposal:{'active' if proposal.id in self.active_proposals else 'history'}:{proposal.id}"

            proposal_data = {
                "id": str(proposal.id),
                "proposer_id": str(proposal.proposer_id),
                "type": proposal.type.value,
                "title": proposal.title,
                "description": proposal.description,
                "status": proposal.status.value,
                "created_at": proposal.created_at.isoformat(),
                "voting_start": proposal.voting_start.isoformat() if proposal.voting_start else "",
                "voting_end": proposal.voting_end.isoformat() if proposal.voting_end else "",
                "votes": str(proposal.votes),
                "vote_weights": str(proposal.vote_weights),
                "final_results": str(proposal.final_results) if proposal.final_results else ""
            }

            await self.redis.hset(key, mapping=proposal_data)

            # Set expiry for historical proposals
            if proposal.status in [ProposalStatus.FAILED, ProposalStatus.EXPIRED, ProposalStatus.IMPLEMENTED]:
                await self.redis.expire(key, 86400 * 30)  # Keep for 30 days

        except Exception as e:
            logger.error(f"Error saving proposal: {e}")

    async def _deserialize_proposal(self, data: Dict) -> Optional[Proposal]:
        """Deserialize proposal from Redis data"""
        try:
            # Handle bytes from Redis
            if isinstance(data.get(b'id'), bytes):
                data = {k.decode(): v.decode() for k, v in data.items()}

            proposal = Proposal(
                id=UUID(data["id"]),
                proposer_id=UUID(data["proposer_id"]),
                type=ProposalType(data["type"]),
                title=data["title"],
                description=data["description"]
            )

            proposal.status = ProposalStatus(data["status"])
            proposal.created_at = datetime.fromisoformat(data["created_at"])

            if data.get("voting_start"):
                proposal.voting_start = datetime.fromisoformat(data["voting_start"])

            if data.get("voting_end"):
                proposal.voting_end = datetime.fromisoformat(data["voting_end"])

            return proposal

        except Exception as e:
            logger.error(f"Error deserializing proposal: {e}")
            return None

    async def _notify_voting_started(self, proposal: Proposal):
        """Notify agents that voting has started"""
        # This would integrate with MQTT to notify agents
        pass

    async def _adjust_tax_rate(self, new_rate: float):
        """Adjust tax rate based on proposal"""
        # Implementation would adjust economy system parameters
        pass

    async def _distribute_stimulus(self, amount: float):
        """Distribute economic stimulus"""
        # Implementation would distribute funds via economy system
        pass

    async def _allocate_district_resources(self, district: str, resources: Dict):
        """Allocate resources to a district"""
        # Implementation would update district resources
        pass

    async def _schedule_event(self, event_details: Dict):
        """Schedule a city event"""
        # Implementation would create an event in the world engine
        pass

    async def get_governance_metrics(self) -> Dict:
        """Get governance system metrics"""
        try:
            total_proposals = len(self.active_proposals) + len(self.proposal_history)

            metrics = {
                "total_proposals": total_proposals,
                "active_proposals": len(self.active_proposals),
                "passed_proposals": len([p for p in self.proposal_history if p.status == ProposalStatus.PASSED]),
                "failed_proposals": len([p for p in self.proposal_history if p.status == ProposalStatus.FAILED]),
                "council_members": len(self.council_members),
                "total_reputation": sum(self.agent_reputation.values()),
                "average_reputation": sum(self.agent_reputation.values()) / len(self.agent_reputation) if self.agent_reputation else 0
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting governance metrics: {e}")
            return {}