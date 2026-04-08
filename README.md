# 🎫 CustomerSupportEnv

> An OpenEnv-compliant reinforcement learning environment for **Customer Support Ticket Triage**.

An AI agent learns to classify priority, craft appropriate responses, and resolve support tickets within SLA constraints — simulating a real enterprise help-desk queue.

---

## 📋 Environment Description

In each episode the agent receives a queue of support tickets. For every ticket, it must:

1. **Classify priority** — `low | medium | high | critical`
2. **Craft a response** — a customer-facing reply addressing the issue
3. **Decide resolution strategy** — resolve, escalate to a human, or leave in-progress

The environment scores the agent on how accurately it triages, how appropriate its responses are, and whether it meets SLA deadlines.

---

## 🔁 OpenEnv API

```python
from env import CustomerSupportEnv, AgentAction

env = CustomerSupportEnv(task="medium", seed=42)
obs = env.reset()

while True:
    action = AgentAction(
        ticket_id=obs["current_ticket"]["ticket_id"],
        assigned_priority="high",
        response="I sincerely apologize. Our team is investigating this immediately.",
        escalate=False,
        resolve=True,
    )
    obs, reward, done, info = env.step(action)
    if done:
        break

state = env.state()
print(f"Final score: {state['mean_reward']:.4f}")
```

---

## 📐 Observation Space

```
{
  "current_ticket": {
    "ticket_id": "TKT-1003",
    "subject": "App crashes on startup",
    "body": "The mobile app crashes every time I open it since the last update.",
    "category": "technical",           # billing | technical | account | shipping | general
    "sentiment": "frustrated",         # polite | neutral | confused | frustrated | angry | panicked
    "age_hours": 3.2,                  # hours since ticket was created
    "status": "open"
  },
  "queue_size": 7,                     # remaining tickets in queue
  "elapsed_steps": 2,
  "max_steps": 10,
  "task_difficulty": "medium"
}
```

---

## 🎮 Action Space

```python
@dataclass
class AgentAction:
    ticket_id: str          # must match current_ticket.ticket_id
    assigned_priority: str  # low | medium | high | critical
    response: str           # customer-facing message
    escalate: bool = False  # escalate to human agent
    resolve: bool = False   # close ticket after responding
```

---

## 🏆 Reward Function

Reward is in **[0.0, 1.0]** per step, with three components:

| Component | Max | Description |
|-----------|-----|-------------|
| **Priority accuracy** | `0.40` | Distance from ground-truth priority (full credit = exact match) |
| **Response quality** | `0.35` | Length, category keyword match, empathy for negative sentiment |
| **SLA compliance** | `0.25` | Bonus for resolving before SLA deadline; partial credit for late |

### Priority accuracy detail
```
delta = |assigned_index - true_index|   (0..3)
reward = max(0, 0.40 - delta × 0.13)
```

### SLA deadlines by priority
| Priority | SLA |
|----------|-----|
| critical | 1 hour |
| high | 4 hours |
| medium | 24 hours |
| low | 72 hours |

---

## 📦 Tasks

| Task | Tickets | Priority noise | Time pressure | Target score |
|------|---------|---------------|---------------|-------------|
| `easy` | 5 | 0% | ❌ | ≥ 0.75 |
| `medium` | 10 | 20% | ❌ | ≥ 0.65 |
| `hard` | 15 | 30% | ✅ | ≥ 0.55 |

---

## 🚀 Setup Instructions

### Local

```bash
git clone https://huggingface.co/spaces/<your-username>/customer-support-env
cd customer-support-env

pip install -r requirements.txt

# Run the Gradio demo
python app.py

# Run baseline inference
python inference.py --task all --verbose

# Run reproducibility report
python inference.py --reproducibility-report

# Run graders
python graders.py
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
# Open http://localhost:7860
```

---

## 📊 Baseline Scores (seed=42)

| Task | Agent | Score |
|------|-------|-------|
| easy | Rule-based | ~0.77 |
| medium | Rule-based | ~0.68 |
| hard | Rule-based | ~0.58 |
| easy | Random | ~0.20 |
| medium | Random | ~0.18 |
| hard | Random | ~0.16 |

Run `python inference.py --reproducibility-report` to verify.

---

## 📁 File Structure

```
customer-support-env/
├── env.py              # CustomerSupportEnv — core environment
├── graders.py          # Agent graders for easy / medium / hard
├── inference.py        # Baseline inference + reproducibility report
├── app.py              # Gradio demo (Hugging Face Spaces)
├── openenv.yaml        # OpenEnv spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Design Notes

- **No games or toys** — simulates a real enterprise SLA-driven support workflow
- **Partial reward signals** — agents receive credit even for imperfect actions, enabling smooth learning gradients
- **Typed models** — `AgentAction` is a typed dataclass; `Priority`, `Category`, `Status` are enums
- **Deterministic with seeds** — all randomness flows through a seeded `random.Random` instance
- **Gradio UI** — interactive demo lets humans play the environment directly

---

## 📜 License

MIT
