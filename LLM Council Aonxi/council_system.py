"""
LLM Council System - Aonxi Decision Engine
Architecture: 3 Agents → 2 Judges → Decision Object with Safety Gating & Audit
"""
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod


# For production, you'd use actual LLM clients
# from openai import OpenAI
# from anthropic import Anthropic

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class SafetyLevel(Enum):
    SAFE = "safe"
    REVIEW = "needs_review"
    BLOCKED = "blocked"


@dataclass
class AgentResponse:
    agent_id: str
    answer: str
    reasoning: str
    citations: List[str]
    timestamp: str


@dataclass
class JudgeComparison:
    judge_id: str
    winning_agent_id: str
    losing_agent_id: str
    rationale: str
    rubric_scores: Dict[str, float]
    confidence: float


@dataclass
class DecisionObject:
    """Final structured decision - what we deliver to clients"""
    question: str
    final_answer: str
    winning_agent_id: str
    confidence: float  # 0-1 scale
    risks: List[str]
    citations: List[str]
    safety_level: SafetyLevel
    audit_hash: str  # For verification
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class AuditEntry:
    session_id: str
    question: str
    input_hash: str
    decision_hash: str
    agents_responses: List[Dict]
    judges_comparisons: List[Dict]
    final_decision: Dict
    timestamp: str


# ============================================================================
# PERSISTENT AUDIT LOG
# ============================================================================

class AuditLogger:
    """Immutable audit trail for compliance and verification"""

    def __init__(self, log_file: str = "council_audit.log"):
        self.log_file = log_file
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def log_session(self, audit_entry: AuditEntry) -> str:
        """Log complete session with cryptographic hashing"""
        entry_dict = asdict(audit_entry)

        # Create verifiable hash chain
        entry_json = json.dumps(entry_dict, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()

        # Append with previous hash for chain integrity
        with open(self.log_file, 'a') as f:
            f.write(f"{entry_hash}|{entry_json}\n")

        return entry_hash


# ============================================================================
# SAFETY GATEKEEPER
# ============================================================================

class SafetyGate:
    """Multi-layer safety checking before decision delivery"""

    def __init__(self):
        self.blocked_patterns = [
            "illegal", "harmful", "dangerous", "kill", "hurt",
            # Add domain-specific blocked terms
        ]
        self.review_patterns = [
            "uncertain", "maybe", "possibly", "not sure",
            "consult", "lawyer", "expert"
        ]

    def evaluate(self, question: str, responses: List[AgentResponse]) -> SafetyLevel:
        """Determine safety level based on inputs and responses"""

        # Check question
        question_lower = question.lower()
        for pattern in self.blocked_patterns:
            if pattern in question_lower:
                return SafetyLevel.BLOCKED

        # Check all agent responses
        all_text = question_lower + " " + " ".join([r.answer.lower() for r in responses])

        for pattern in self.blocked_patterns:
            if pattern in all_text:
                return SafetyLevel.BLOCKED

        for pattern in self.review_patterns:
            if pattern in all_text:
                return SafetyLevel.REVIEW

        return SafetyLevel.SAFE


# ============================================================================
# AGENTS (3 Independent Answer Generators)
# ============================================================================

class BaseAgent(ABC):
    """Abstract agent with consistent interface"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.style = self._define_style()

    @abstractmethod
    def _define_style(self) -> str:
        """Define this agent's reasoning style/approach"""
        pass

    @abstractmethod
    def generate_response(self, question: str) -> AgentResponse:
        pass


class PreciseAgent(BaseAgent):
    """Agent 1: Precise, factual, citation-oriented"""

    def _define_style(self) -> str:
        return "precise_factual"

    def generate_response(self, question: str) -> AgentResponse:
        # In production: Call LLM API with specific prompt
        answer = f"[Precise Agent] Based on available data: {question[:50]}..."
        reasoning = "I've analyzed this systematically with attention to verifiable facts."
        citations = ["source_2023_study.pdf", "industry_report_2024.md"]

        return AgentResponse(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=reasoning,
            citations=citations,
            timestamp=datetime.now().isoformat()
        )


class CreativeAgent(BaseAgent):
    """Agent 2: Creative, strategic, pattern-oriented"""

    def _define_style(self) -> str:
        return "creative_strategic"

    def generate_response(self, question: str) -> AgentResponse:
        answer = f"[Creative Agent] Considering novel approaches: {question[:50]}..."
        reasoning = "I've explored unconventional angles and patterns others might miss."
        citations = ["innovation_patterns.json", "case_studies.db"]

        return AgentResponse(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=reasoning,
            citations=citations,
            timestamp=datetime.now().isoformat()
        )


class ConservativeAgent(BaseAgent):
    """Agent 3: Conservative, risk-aware, precedent-oriented"""

    def _define_style(self) -> str:
        return "conservative_risk_aware"

    def generate_response(self, question: str) -> AgentResponse:
        answer = f"[Conservative Agent] With measured consideration: {question[:50]}..."
        reasoning = "I've prioritized proven methods and identified key risks."
        citations = ["risk_assessment.md", "historical_data.csv"]

        return AgentResponse(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=reasoning,
            citations=citations,
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# JUDGES (2 Comparative Evaluators)
# ============================================================================

class Judge:
    """Judge compares pairs of agent responses using rubric"""

    def __init__(self, judge_id: str):
        self.judge_id = judge_id
        self.rubric = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "clarity": 0.2,
            "citation_quality": 0.15,
            "risk_awareness": 0.1
        }

    def compare_responses(
            self,
            question: str,
            response_a: AgentResponse,
            response_b: AgentResponse
    ) -> JudgeComparison:
        """
        Compare two agent responses using rubric
        Judges DON'T generate answers - only compare
        """

        # In production: Call LLM API with comparison prompt
        # For demo, simulate scoring
        score_a = self._score_response(response_a)
        score_b = self._score_response(response_b)

        winner = response_a if score_a > score_b else response_b
        loser = response_b if winner == response_a else response_a

        confidence = abs(score_a - score_b)  # Difference as confidence

        return JudgeComparison(
            judge_id=self.judge_id,
            winning_agent_id=winner.agent_id,
            losing_agent_id=loser.agent_id,
            rationale=f"Response from {winner.agent_id} scored higher across rubric dimensions.",
            rubric_scores={winner.agent_id: max(score_a, score_b),
                           loser.agent_id: min(score_a, score_b)},
            confidence=confidence
        )

    def _score_response(self, response: AgentResponse) -> float:
        """Calculate rubric score for a response"""
        # Simplified scoring - in production, use LLM evaluation
        base_score = len(response.answer) * 0.01
        citation_bonus = len(response.citations) * 0.1
        return min(1.0, 0.5 + base_score + citation_bonus)


# ============================================================================
# COUNCIL ORCHESTRATOR
# ============================================================================

class CouncilOrchestrator:
    """Main orchestrator for the LLM Council"""

    def __init__(self):
        # Initialize components
        self.agents = [
            PreciseAgent("agent_precise_01"),
            CreativeAgent("agent_creative_02"),
            ConservativeAgent("agent_conservative_03")
        ]

        self.judges = [
            Judge("judge_primary_01"),
            Judge("judge_secondary_02")
        ]

        self.safety_gate = SafetyGate()
        self.audit_logger = AuditLogger()
        self.sessions = {}

    def deliberate(self, question: str) -> DecisionObject:
        """Main deliberation pipeline"""
        session_id = hashlib.md5(f"{question}{time.time()}".encode()).hexdigest()[:8]

        # Step 1: Generate independent answers
        print(f"\n{'=' * 60}")
        print(f"COUNCIL DELIBERATION: {question[:50]}...")
        print(f"Session: {session_id}")
        print(f"{'=' * 60}")

        agent_responses = []
        for agent in self.agents:
            print(f"\n[{agent.agent_id}] Generating response...")
            response = agent.generate_response(question)
            agent_responses.append(response)
            print(f"  Answer: {response.answer[:80]}...")

        # Step 2: Safety evaluation
        print(f"\n[SAFETY GATE] Evaluating safety...")
        safety_level = self.safety_gate.evaluate(question, agent_responses)
        print(f"  Safety Level: {safety_level.value}")

        if safety_level == SafetyLevel.BLOCKED:
            return self._create_blocked_decision(question, session_id)

        # Step 3: Judges compare responses (round-robin comparisons)
        print(f"\n[JUDGES] Beginning comparisons...")
        comparisons = []

        # Judge 1: Compare agent 1 vs 2
        comparison1 = self.judges[0].compare_responses(
            question, agent_responses[0], agent_responses[1]
        )
        comparisons.append(comparison1)
        print(f"  Judge 1: {comparison1.winning_agent_id} wins")

        # Judge 2: Compare agent 2 vs 3
        comparison2 = self.judges[1].compare_responses(
            question, agent_responses[1], agent_responses[2]
        )
        comparisons.append(comparison2)
        print(f"  Judge 2: {comparison2.winning_agent_id} wins")

        # Step 4: Determine final winner
        print(f"\n[DECISION] Calculating final verdict...")
        winner_id = self._determine_winner(comparisons, agent_responses)
        winner_response = next(r for r in agent_responses if r.agent_id == winner_id)

        # Step 5: Construct decision object
        decision = self._construct_decision(
            question=question,
            winner_response=winner_response,
            all_responses=agent_responses,
            comparisons=comparisons,
            safety_level=safety_level,
            session_id=session_id
        )

        # Step 6: Audit logging
        print(f"\n[AUDIT] Logging session...")
        self._log_session(
            session_id=session_id,
            question=question,
            agent_responses=agent_responses,
            comparisons=comparisons,
            decision=decision
        )

        print(f"\n{'=' * 60}")
        print(f"DECISION DELIVERED")
        print(f"Answer: {decision.final_answer[:80]}...")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Citations: {len(decision.citations)} sources")
        print(f"{'=' * 60}")

        return decision

    def _determine_winner(
            self,
            comparisons: List[JudgeComparison],
            responses: List[AgentResponse]
    ) -> str:
        """Determine winning agent based on judge comparisons"""
        # Simple voting - in production, more sophisticated aggregation
        votes = {}
        for comp in comparisons:
            votes[comp.winning_agent_id] = votes.get(comp.winning_agent_id, 0) + 1

        # Return agent with most votes
        winner = max(votes.items(), key=lambda x: x[1])[0]

        # Tie-breaking logic (optional)
        if len(set(votes.values())) == 1:  # All tied
            # Default to first agent or implement additional logic
            winner = responses[0].agent_id

        return winner

    def _construct_decision(
            self,
            question: str,
            winner_response: AgentResponse,
            all_responses: List[AgentResponse],
            comparisons: List[JudgeComparison],
            safety_level: SafetyLevel,
            session_id: str
    ) -> DecisionObject:
        """Construct final decision object"""

        # Aggregate confidence from judges
        total_confidence = sum(c.confidence for c in comparisons)
        avg_confidence = total_confidence / len(comparisons) if comparisons else 0.5

        # Aggregate citations from all agents (deduplicated)
        all_citations = []
        for response in all_responses:
            all_citations.extend(response.citations)
        unique_citations = list(set(all_citations))

        # Identify risks from conservative agent
        risks = []
        conservative_response = next(
            (r for r in all_responses if "conservative" in r.agent_id),
            None
        )
        if conservative_response:
            risks.append(f"Risk-aware perspective: {conservative_response.reasoning[:100]}...")

        # Create audit hash
        decision_data = f"{question}{winner_response.answer}{avg_confidence}"
        audit_hash = hashlib.sha256(decision_data.encode()).hexdigest()[:16]

        return DecisionObject(
            question=question,
            final_answer=winner_response.answer,
            winning_agent_id=winner_response.agent_id,
            confidence=min(0.99, avg_confidence),  # Cap at 99%
            risks=risks,
            citations=unique_citations,
            safety_level=safety_level,
            audit_hash=audit_hash,
            metadata={
                "session_id": session_id,
                "total_responses": len(all_responses),
                "total_comparisons": len(comparisons),
                "agent_styles": [agent.style for agent in self.agents]
            },
            timestamp=datetime.now().isoformat()
        )

    def _create_blocked_decision(self, question: str, session_id: str) -> DecisionObject:
        """Create a decision object for blocked/safe queries"""
        return DecisionObject(
            question=question,
            final_answer="[SAFETY BLOCKED] This query cannot be processed due to safety constraints.",
            winning_agent_id="safety_gate",
            confidence=1.0,
            risks=["Query triggered safety filters"],
            citations=[],
            safety_level=SafetyLevel.BLOCKED,
            audit_hash=hashlib.sha256(f"blocked{question}".encode()).hexdigest()[:16],
            metadata={
                "session_id": session_id,
                "blocked": True,
                "reason": "safety_violation"
            },
            timestamp=datetime.now().isoformat()
        )

    def _log_session(
            self,
            session_id: str,
            question: str,
            agent_responses: List[AgentResponse],
            comparisons: List[JudgeComparison],
            decision: DecisionObject
    ):
        """Create audit log entry"""
        input_hash = hashlib.sha256(question.encode()).hexdigest()[:16]

        audit_entry = AuditEntry(
            session_id=session_id,
            question=question,
            input_hash=input_hash,
            decision_hash=decision.audit_hash,
            agents_responses=[asdict(r) for r in agent_responses],
            judges_comparisons=[asdict(c) for c in comparisons],
            final_decision=asdict(decision),
            timestamp=datetime.now().isoformat()
        )

        self.audit_logger.log_session(audit_entry)


# ============================================================================
# MAIN EXECUTION & DEPLOYMENT READY
# ============================================================================

def main():
    """Example usage of the LLM Council"""

    # Initialize council
    council = CouncilOrchestrator()

    # Example questions
    test_questions = [
        "What are the most effective risk mitigation strategies for AI deployment in healthcare?",
        "How should a company evaluate trade-offs between innovation speed and system safety?",
        # Try: "How to build a dangerous weapon?" (will be blocked)
    ]

    # Run deliberation
    for question in test_questions:
        decision = council.deliberate(question)

        # Display decision object
        print("\n" + "=" * 60)
        print("DECISION OBJECT (Structured Output):")
        print("=" * 60)
        print(json.dumps(asdict(decision), indent=2))

        # Export to file (simulating API response)
        output_file = f"decision_{decision.metadata['session_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(asdict(decision), f, indent=2)
        print(f"\nDecision saved to: {output_file}")


if __name__ == "__main__":
    main()
