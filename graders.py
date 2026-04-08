"""
graders.py — Agent graders for CustomerSupportEnv (easy / medium / hard)

Each grader runs a deterministic baseline agent against its task level and
reports a score in [0.0, 1.0].

Usage
-----
    python graders.py            # runs all three graders
    python graders.py --task easy
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Any

from env import CustomerSupportEnv, AgentAction, Priority, Category, RESPONSE_TEMPLATES


# ---------------------------------------------------------------------------
# Baseline agents (one per difficulty)
# ---------------------------------------------------------------------------

class EasyAgent:
    """
    Rule-based agent for the easy task.
    Uses simple keyword matching on subject/body to classify priority.
    """

    CRITICAL_KEYWORDS = ["critical", "down", "broken", "losing", "emergency"]
    HIGH_KEYWORDS = ["double charged", "crash", "hasn't arrived", "wrong item", "not working"]
    LOW_KEYWORDS = ["how do i", "question", "update", "change", "info"]

    def act(self, obs: Dict[str, Any]) -> AgentAction:
        ticket = obs["current_ticket"]
        text = (ticket["subject"] + " " + ticket["body"]).lower()

        if any(kw in text for kw in self.CRITICAL_KEYWORDS):
            priority = "critical"
        elif any(kw in text for kw in self.HIGH_KEYWORDS):
            priority = "high"
        elif any(kw in text for kw in self.LOW_KEYWORDS):
            priority = "low"
        else:
            priority = "medium"

        category = Category(ticket["category"])
        response = RESPONSE_TEMPLATES.get(category, "Thank you for contacting support. We will assist you shortly.")

        return AgentAction(
            ticket_id=ticket["ticket_id"],
            assigned_priority=priority,
            response=response,
            resolve=(priority in ("low", "medium")),
            escalate=(priority == "critical"),
        )


class MediumAgent:
    """
    Enhanced rule-based agent for the medium task.
    Adds sentiment weighting and more nuanced keyword matching.
    """

    PRIORITY_SIGNALS = {
        "critical": (["critical", "down", "losing sales", "broken", "emergency", "urgent"], 4),
        "high": (["charged twice", "crash", "hasn't arrived", "wrong item", "refund", "cannot"], 3),
        "medium": (["unexpected charge", "rate limit", "api error", "not receiving", "issue"], 2),
        "low": (["how do i", "question", "update address", "notification", "change settings"], 1),
    }

    def act(self, obs: Dict[str, Any]) -> AgentAction:
        ticket = obs["current_ticket"]
        text = (ticket["subject"] + " " + ticket["body"]).lower()
        sentiment = ticket["sentiment"]

        best_priority = "medium"
        best_score = 0

        for priority, (keywords, weight) in self.PRIORITY_SIGNALS.items():
            matches = sum(1 for kw in keywords if kw in text)
            score = matches * weight
            if score > best_score:
                best_score = score
                best_priority = priority

        # Boost priority for angry/panicked sentiment
        if sentiment in ("angry", "panicked") and best_priority in ("low", "medium"):
            boost = {"low": "medium", "medium": "high"}
            best_priority = boost.get(best_priority, best_priority)

        category = Category(ticket["category"])
        base_response = RESPONSE_TEMPLATES.get(category, "Thank you for contacting us.")

        # Add empathy prefix for negative sentiment
        if sentiment in ("angry", "frustrated", "panicked"):
            response = f"I sincerely apologize for the inconvenience. {base_response}"
        else:
            response = base_response

        return AgentAction(
            ticket_id=ticket["ticket_id"],
            assigned_priority=best_priority,
            response=response,
            resolve=(best_priority in ("low", "medium")),
            escalate=(best_priority == "critical"),
        )


class HardAgent:
    """
    SLA-aware agent for the hard task.
    Factors in ticket age versus SLA deadlines and adjusts strategy accordingly.
    """

    SLA_HOURS = {"critical": 1, "high": 4, "medium": 24, "low": 72}

    def act(self, obs: Dict[str, Any]) -> AgentAction:
        ticket = obs["current_ticket"]
        text = (ticket["subject"] + " " + ticket["body"]).lower()
        sentiment = ticket["sentiment"]
        age_hours = ticket.get("age_hours", 0.0)

        # Priority classification
        if any(kw in text for kw in ["critical", "down", "losing", "emergency"]):
            priority = "critical"
        elif any(kw in text for kw in ["crash", "charged", "arrived", "wrong", "refund", "cannot"]):
            priority = "high"
        elif any(kw in text for kw in ["unexpected", "api", "rate limit", "not receiving"]):
            priority = "medium"
        else:
            priority = "low"

        # SLA breach check — upgrade if near/past deadline
        sla = self.SLA_HOURS.get(priority, 24)
        if age_hours >= sla * 0.85 and priority != "critical":
            upgrade_map = {"low": "medium", "medium": "high", "high": "critical"}
            priority = upgrade_map.get(priority, priority)

        category = Category(ticket["category"])
        base_response = RESPONSE_TEMPLATES.get(category, "Thank you for contacting us.")

        # Sentiment-aware prefix
        if sentiment in ("angry", "panicked"):
            response = (
                f"I sincerely apologize for the disruption this has caused. "
                f"{base_response} I am personally escalating this to ensure swift resolution."
            )
        elif sentiment == "frustrated":
            response = f"I understand your frustration and I'm sorry for the trouble. {base_response}"
        else:
            response = base_response

        escalate = priority == "critical" or age_hours > sla
        resolve = not escalate and priority in ("low", "medium")

        return AgentAction(
            ticket_id=ticket["ticket_id"],
            assigned_priority=priority,
            response=response,
            resolve=resolve,
            escalate=escalate,
        )


# ---------------------------------------------------------------------------
# Grader runner
# ---------------------------------------------------------------------------

AGENTS = {
    "easy": EasyAgent,
    "medium": MediumAgent,
    "hard": HardAgent,
}

SEEDS = {"easy": 42, "medium": 42, "hard": 42}


def run_grader(task: str, verbose: bool = False) -> Dict[str, Any]:
    env = CustomerSupportEnv(task=task, seed=SEEDS[task])
    agent = AGENTS[task]()

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    step_details = []

    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        step_details.append({"step": steps, "reward": reward, "info": info})
        if done:
            break

    mean_reward = total_reward / steps if steps > 0 else 0.0
    result = {
        "task": task,
        "seed": SEEDS[task],
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "mean_reward": round(mean_reward, 4),
        "score": round(mean_reward, 4),   # canonical score field
    }

    if verbose:
        result["step_details"] = step_details

    return result


def main():
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Grader")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print("=" * 60)
    print("CustomerSupportEnv — Agent Grader")
    print("=" * 60)

    all_results = []
    for task in tasks:
        result = run_grader(task, verbose=args.verbose)
        all_results.append(result)
        print(f"\nTask: {task.upper()}")
        print(f"  Steps         : {result['steps']}")
        print(f"  Total Reward  : {result['total_reward']}")
        print(f"  Mean Reward   : {result['mean_reward']}")
        print(f"  Score         : {result['score']}")

        if args.verbose and "step_details" in result:
            print("\n  Step-by-step:")
            for s in result["step_details"]:
                print(f"    Step {s['step']}: reward={s['reward']:.4f} | {s['info']}")

    print("\n" + "=" * 60)
    print("Summary (JSON):")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
