"""
inference.py — Baseline inference script for CustomerSupportEnv

Runs all three task difficulties with fixed seeds and prints reproducible scores.
Can also be used as a template for custom agent implementations.

Usage
-----
    python inference.py                          # run all tasks, default agents
    python inference.py --task hard --seed 123   # specific task + seed
    python inference.py --agent random           # random baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any, Dict, Optional

from env import CustomerSupportEnv, AgentAction, Priority, Category, RESPONSE_TEMPLATES
from graders import EasyAgent, MediumAgent, HardAgent, run_grader


# ---------------------------------------------------------------------------
# Random baseline agent (lower bound)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Selects a random priority and generic response — establishes a lower bound."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def act(self, obs: Dict[str, Any]) -> AgentAction:
        ticket = obs["current_ticket"]
        priority = self._rng.choice(["low", "medium", "high", "critical"])
        return AgentAction(
            ticket_id=ticket["ticket_id"],
            assigned_priority=priority,
            response="Thank you for reaching out. We will look into your issue shortly.",
            resolve=self._rng.random() > 0.5,
            escalate=False,
        )


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(
    task: str,
    agent_name: str = "rule",
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a full episode and return results dict."""
    env = CustomerSupportEnv(task=task, seed=seed)

    if agent_name == "random":
        agent = RandomAgent(seed=seed)
    elif task == "easy":
        agent = EasyAgent()
    elif task == "medium":
        agent = MediumAgent()
    else:
        agent = HardAgent()

    obs = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    transcript = []

    print(f"\n{'─'*60}")
    print(f"  Task: {task.upper()} | Agent: {agent_name} | Seed: {seed}")
    print(f"{'─'*60}")

    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            print(f"  Step {steps:02d} | reward={reward:.4f} | "
                  f"priority={info.get('assigned_priority','?')} "
                  f"(true={info.get('true_priority','?')}) | "
                  f"response_quality={info.get('response_quality','?')}")

        transcript.append({
            "step": steps,
            "reward": round(reward, 4),
            "info": info,
        })

        if done:
            break

    mean_reward = total_reward / steps
    print(f"\n  ✓ Complete — {steps} tickets processed")
    print(f"    Total reward : {total_reward:.4f}")
    print(f"    Mean reward  : {mean_reward:.4f}  ← score")

    return {
        "task": task,
        "agent": agent_name,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "mean_reward": round(mean_reward, 4),
        "score": round(mean_reward, 4),
        "transcript": transcript if verbose else [],
    }


# ---------------------------------------------------------------------------
# Reproducibility report
# ---------------------------------------------------------------------------

REPRODUCIBLE_BASELINES = [
    {"task": "easy",   "agent": "rule", "seed": 42},
    {"task": "medium", "agent": "rule", "seed": 42},
    {"task": "hard",   "agent": "rule", "seed": 42},
    {"task": "easy",   "agent": "random", "seed": 0},
    {"task": "medium", "agent": "random", "seed": 0},
    {"task": "hard",   "agent": "random", "seed": 0},
]


def main():
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Baseline Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--agent", choices=["rule", "random"], default="rule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reproducibility-report", action="store_true",
                        help="Run all baseline configurations for a reproducibility report")
    args = parser.parse_args()

    print("=" * 60)
    print("  CustomerSupportEnv — Baseline Inference")
    print("=" * 60)

    if args.reproducibility_report:
        results = []
        for cfg in REPRODUCIBLE_BASELINES:
            r = run_inference(
                task=cfg["task"],
                agent_name=cfg["agent"],
                seed=cfg["seed"],
                verbose=False,
            )
            results.append(r)

        print("\n" + "=" * 60)
        print("REPRODUCIBILITY REPORT")
        print("=" * 60)
        header = f"{'Task':<8} {'Agent':<8} {'Seed':<6} {'Score':>8}"
        print(header)
        print("─" * len(header))
        for r in results:
            print(f"{r['task']:<8} {r['agent']:<8} {r['seed']:<6} {r['score']:>8.4f}")

        print("\nFull JSON:")
        print(json.dumps(results, indent=2))
        return

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    all_results = []
    for task in tasks:
        r = run_inference(task=task, agent_name=args.agent, seed=args.seed, verbose=args.verbose)
        all_results.append(r)

    print("\n" + "=" * 60)
    print("Results JSON:")
    print(json.dumps(
        [{k: v for k, v in r.items() if k != "transcript"} for r in all_results],
        indent=2
    ))


if __name__ == "__main__":
    main()
