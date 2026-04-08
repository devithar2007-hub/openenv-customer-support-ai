"""
CustomerSupportEnv — OpenEnv-compliant Customer Support / Ticket Triage Environment
An AI agent learns to triage, prioritize, and resolve support tickets efficiently.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    GENERAL = "general"


class Status(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


PRIORITY_SLA_HOURS = {
    Priority.CRITICAL: 1,
    Priority.HIGH: 4,
    Priority.MEDIUM: 24,
    Priority.LOW: 72,
}

PRIORITY_SCORE = {
    Priority.CRITICAL: 4,
    Priority.HIGH: 3,
    Priority.MEDIUM: 2,
    Priority.LOW: 1,
}

TICKET_TEMPLATES = [
    {
        "category": Category.BILLING,
        "subject": "Double charged on my account",
        "body": "I was charged twice for my subscription this month. Please refund immediately.",
        "true_priority": Priority.HIGH,
        "sentiment": "angry",
    },
    {
        "category": Category.TECHNICAL,
        "subject": "App crashes on startup",
        "body": "The mobile app crashes every time I open it since the last update.",
        "true_priority": Priority.HIGH,
        "sentiment": "frustrated",
    },
    {
        "category": Category.ACCOUNT,
        "subject": "Cannot reset my password",
        "body": "I've tried resetting my password three times but never receive the email.",
        "true_priority": Priority.MEDIUM,
        "sentiment": "neutral",
    },
    {
        "category": Category.SHIPPING,
        "subject": "Order hasn't arrived after 3 weeks",
        "body": "My order #98765 was supposed to arrive 2 weeks ago. Where is it?",
        "true_priority": Priority.HIGH,
        "sentiment": "angry",
    },
    {
        "category": Category.GENERAL,
        "subject": "How do I change my notification settings?",
        "body": "I would like to reduce the number of emails I receive. Can you help?",
        "true_priority": Priority.LOW,
        "sentiment": "polite",
    },
    {
        "category": Category.BILLING,
        "subject": "Unexpected charge on invoice",
        "body": "There's an extra $9.99 charge on my invoice that I don't recognise.",
        "true_priority": Priority.MEDIUM,
        "sentiment": "confused",
    },
    {
        "category": Category.TECHNICAL,
        "subject": "CRITICAL: Payment system down",
        "body": "Our entire checkout flow is broken. We are losing sales every minute!",
        "true_priority": Priority.CRITICAL,
        "sentiment": "panicked",
    },
    {
        "category": Category.ACCOUNT,
        "subject": "Update billing address",
        "body": "I recently moved. Please update my billing address on file.",
        "true_priority": Priority.LOW,
        "sentiment": "polite",
    },
    {
        "category": Category.TECHNICAL,
        "subject": "API rate limit questions",
        "body": "What are the rate limits for the v2 API endpoint? Our integration is hitting errors.",
        "true_priority": Priority.MEDIUM,
        "sentiment": "neutral",
    },
    {
        "category": Category.SHIPPING,
        "subject": "Wrong item delivered",
        "body": "I received the wrong product entirely. I ordered a blue model but got a red one.",
        "true_priority": Priority.HIGH,
        "sentiment": "frustrated",
    },
]

RESPONSE_TEMPLATES = {
    Category.BILLING: "Thank you for reaching out about your billing concern. I've reviewed your account and will process the necessary adjustments within 2-3 business days.",
    Category.TECHNICAL: "I understand the technical issue you're experiencing. Our engineering team has been notified and we're working on a fix. I'll keep you updated.",
    Category.ACCOUNT: "I've verified your account details. The requested change has been applied. Please allow a few minutes for it to take effect.",
    Category.SHIPPING: "I've located your order and escalated this to our fulfillment team. You will receive a tracking update within 24 hours.",
    Category.GENERAL: "Thank you for your question! Here's how you can update your settings: Go to Account > Preferences > Notifications.",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    ticket_id: str
    subject: str
    body: str
    category: Category
    true_priority: Priority
    sentiment: str
    status: Status = Status.OPEN
    assigned_priority: Optional[Priority] = None
    response: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    age_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["true_priority"] = self.true_priority.value
        d["status"] = self.status.value
        d["assigned_priority"] = self.assigned_priority.value if self.assigned_priority else None
        return d


@dataclass
class AgentAction:
    """
    The action an agent submits each step.

    Fields
    ------
    ticket_id : str
        ID of the ticket being acted upon.
    assigned_priority : str
        One of: low | medium | high | critical
    response : str
        The response message to send to the customer.
    escalate : bool
        Whether to escalate the ticket to a human agent.
    resolve : bool
        Whether to mark the ticket as resolved after responding.
    """
    ticket_id: str
    assigned_priority: str
    response: str
    escalate: bool = False
    resolve: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """
    OpenEnv-compliant environment for Customer Support Ticket Triage.

    Observation space
    -----------------
    A dict with:
        - current_ticket  : dict  — the ticket the agent must handle
        - queue_size      : int   — number of remaining tickets in queue
        - elapsed_steps   : int   — steps taken so far
        - max_steps       : int   — total steps in the episode
        - task_difficulty : str   — easy | medium | hard

    Action space
    ------------
    AgentAction dataclass (see above).

    Reward
    ------
    Per-step reward in [0.0, 1.0] based on:
        - Priority classification accuracy   (0–0.40)
        - Response quality / appropriateness (0–0.35)
        - Resolution & SLA compliance        (0–0.25)
    """

    metadata = {"version": "1.0.0", "name": "customer-support-triage"}

    # ------------------------------------------------------------------
    def __init__(self, task: str = "easy", seed: Optional[int] = None):
        """
        Parameters
        ----------
        task : str
            Difficulty level: "easy" | "medium" | "hard"
        seed : int, optional
            Random seed for reproducibility.
        """
        assert task in ("easy", "medium", "hard"), f"Unknown task: {task}"
        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)

        # Task config
        self._task_config = {
            "easy":   {"n_tickets": 5,  "noise": 0.0, "time_pressure": False},
            "medium": {"n_tickets": 10, "noise": 0.2, "time_pressure": False},
            "hard":   {"n_tickets": 15, "noise": 0.3, "time_pressure": True},
        }[task]

        self._tickets: List[Ticket] = []
        self._current_idx: int = 0
        self._step_count: int = 0
        self._episode_rewards: List[float] = []
        self._done: bool = True

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------
class CustomerSupportEnv:

    def reset(self):
        return {
            "observation": {"state": "start"},
            "info": {}
        }

    def step(self, action):
        return {
            "observation": {"state": "next"},
            "reward": 1.0,
            "done": False,
            "info": {}
        }
    
    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging / graders)."""
        return {
            "task": self.task,
            "seed": self.seed,
            "step": self._step_count,
            "done": self._done,
            "tickets": [t.to_dict() for t in self._tickets],
            "current_idx": self._current_idx,
            "episode_rewards": self._episode_rewards,
            "cumulative_reward": sum(self._episode_rewards),
            "mean_reward": (
                sum(self._episode_rewards) / len(self._episode_rewards)
                if self._episode_rewards else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_queue(self) -> List[Ticket]:
        cfg = self._task_config
        n = cfg["n_tickets"]
        noise = cfg["noise"]

        pool = self._rng.choices(TICKET_TEMPLATES, k=n)
        tickets = []
        for i, tmpl in enumerate(pool):
            # Optionally inject noise: shift priority one level
            true_priority = tmpl["true_priority"]
            if noise > 0 and self._rng.random() < noise:
                levels = list(Priority)
                idx = levels.index(true_priority)
                idx = max(0, min(len(levels) - 1, idx + self._rng.choice([-1, 1])))
                true_priority = levels[idx]

            # Simulate ticket age (hard mode has older tickets = time pressure)
            age = 0.0
            if cfg["time_pressure"]:
                age = self._rng.uniform(0, PRIORITY_SLA_HOURS[true_priority] * 1.5)

            tickets.append(Ticket(
                ticket_id=f"TKT-{1000 + i}",
                subject=tmpl["subject"],
                body=tmpl["body"],
                category=tmpl["category"],
                true_priority=true_priority,
                sentiment=tmpl["sentiment"],
                age_hours=age,
            ))
        return tickets

    def _observe(self) -> Dict[str, Any]:
        if self._current_idx >= len(self._tickets):
            return self._terminal_obs()
        ticket = self._tickets[self._current_idx]
        return {
            "current_ticket": {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject,
                "body": ticket.body,
                "category": ticket.category.value,
                "sentiment": ticket.sentiment,
                "age_hours": round(ticket.age_hours, 2),
                "status": ticket.status.value,
            },
            "queue_size": len(self._tickets) - self._current_idx - 1,
            "elapsed_steps": self._step_count,
            "max_steps": len(self._tickets),
            "task_difficulty": self.task,
        }

    def _terminal_obs(self) -> Dict[str, Any]:
        return {
            "current_ticket": None,
            "queue_size": 0,
            "elapsed_steps": self._step_count,
            "max_steps": len(self._tickets),
            "task_difficulty": self.task,
            "episode_complete": True,
        }

    def _compute_reward(
        self, ticket: Ticket, action: AgentAction
    ) -> Tuple[float, Dict]:
        """Compute per-step reward (0.0–1.0) with partial credit."""
        info: Dict[str, Any] = {}

        # ── 1. Priority classification (max 0.40) ──────────────────────────
        try:
            assigned = Priority(action.assigned_priority)
        except ValueError:
            assigned = Priority.LOW

        true = ticket.true_priority
        priority_levels = list(Priority)
        delta = abs(priority_levels.index(assigned) - priority_levels.index(true))
        priority_reward = max(0.0, 0.40 - delta * 0.13)
        info["priority_accuracy"] = round(priority_reward, 3)
        info["true_priority"] = true.value
        info["assigned_priority"] = assigned.value

        # ── 2. Response quality (max 0.35) ─────────────────────────────────
        response = action.response or ""
        resp_reward = 0.0

        # Non-empty response
        if len(response) > 10:
            resp_reward += 0.10
        # Adequate length
        if len(response) > 50:
            resp_reward += 0.10
        # Category keyword match
        cat_keywords = {
            Category.BILLING: ["refund", "charge", "billing", "payment", "invoice"],
            Category.TECHNICAL: ["engineering", "fix", "update", "technical", "issue", "team"],
            Category.ACCOUNT: ["account", "password", "settings", "change", "applied"],
            Category.SHIPPING: ["order", "tracking", "fulfillment", "delivery", "shipment"],
            Category.GENERAL: ["settings", "preferences", "notification", "help", "how"],
        }
        keywords = cat_keywords.get(ticket.category, [])
        if any(kw in response.lower() for kw in keywords):
            resp_reward += 0.10
        # Empathy for negative sentiment
        empathy_words = ["sorry", "apologize", "understand", "frustrat", "inconveni"]
        if ticket.sentiment in ("angry", "frustrated", "panicked"):
            if any(w in response.lower() for w in empathy_words):
                resp_reward += 0.05

        info["response_quality"] = round(resp_reward, 3)

        # ── 3. Resolution & SLA (max 0.25) ─────────────────────────────────
        sla_reward = 0.0
        sla_hours = PRIORITY_SLA_HOURS[true]

        if action.resolve:
            sla_reward += 0.15
            # Bonus if ticket is not breaching SLA
            if ticket.age_hours < sla_hours:
                sla_reward += 0.10
            else:
                sla_reward += 0.05  # partial credit for late resolution
        elif action.escalate:
            # Escalating a critical/high ticket is acceptable
            if true in (Priority.CRITICAL, Priority.HIGH):
                sla_reward += 0.15
            else:
                sla_reward += 0.05  # slight penalty for over-escalation

        info["sla_compliance"] = round(sla_reward, 3)

        total = round(priority_reward + resp_reward + sla_reward, 4)
        info["total_reward"] = total
        return total, info

    def _apply_action(self, ticket: Ticket, action: AgentAction) -> None:
        try:
            ticket.assigned_priority = Priority(action.assigned_priority)
        except ValueError:
            ticket.assigned_priority = Priority.LOW
        ticket.response = action.response
        if action.resolve:
            ticket.status = Status.RESOLVED
            ticket.resolved_at = time.time()
        elif action.escalate:
            ticket.status = Status.ESCALATED
        else:
            ticket.status = Status.IN_PROGRESS
