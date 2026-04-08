"""
app.py — Gradio demo for CustomerSupportEnv (Hugging Face Spaces)

Provides an interactive UI for:
  1. Stepping through a triage episode manually
  2. Running the baseline grader and viewing scores
  3. Viewing the environment state as JSON
"""

import json
import gradio as gr
from env import CustomerSupportEnv, AgentAction, Priority
from graders import run_grader

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

_env_cache: dict = {}


def _get_env(task: str, seed: int) -> CustomerSupportEnv:
    key = (task, seed)
    if key not in _env_cache:
        _env_cache[key] = CustomerSupportEnv(task=task, seed=seed)
    return _env_cache[key]


# ---------------------------------------------------------------------------
# Interactive step
# ---------------------------------------------------------------------------

def reset_env(task: str, seed: int):
    env = _get_env(task, int(seed))
    obs = env.reset(seed=int(seed))
    ticket = obs.get("current_ticket", {})
    state_json = json.dumps(env.state(), indent=2)
    return (
        ticket.get("ticket_id", "—"),
        ticket.get("subject", "—"),
        ticket.get("body", "—"),
        ticket.get("category", "—"),
        ticket.get("sentiment", "—"),
        f"{ticket.get('age_hours', 0):.1f}h",
        f"Queue: {obs['queue_size']} remaining",
        state_json,
        "Environment reset. Make your first triage decision!",
    )


def take_step(task: str, seed: int, ticket_id: str, priority: str, response: str, escalate: bool, resolve: bool):
    env = _get_env(task, int(seed))
    if env._done:
        return (
            "—", "—", "—", "—", "—", "—",
            "Episode complete. Click Reset to start again.",
            json.dumps(env.state(), indent=2),
            "Episode is done!",
        )

    action = AgentAction(
        ticket_id=ticket_id,
        assigned_priority=priority,
        response=response,
        escalate=escalate,
        resolve=resolve,
    )
    obs, reward, done, info = env.step(action)
    state_json = json.dumps(env.state(), indent=2)

    result_msg = (
        f"✅ Step complete! Reward: {reward:.4f}\n"
        f"  Priority accuracy: {info.get('priority_accuracy', '—')}\n"
        f"  Response quality:  {info.get('response_quality', '—')}\n"
        f"  SLA compliance:    {info.get('sla_compliance', '—')}\n"
        f"  True priority:     {info.get('true_priority', '—')}"
    )

    if done:
        ticket_id_out = "—"
        subject_out = "✅ Episode complete!"
        body_out = f"Final score: {env.state()['mean_reward']:.4f}"
        cat_out = "—"
        sent_out = "—"
        age_out = "—"
        queue_out = "Done"
    else:
        ticket = obs.get("current_ticket", {})
        ticket_id_out = ticket.get("ticket_id", "—")
        subject_out = ticket.get("subject", "—")
        body_out = ticket.get("body", "—")
        cat_out = ticket.get("category", "—")
        sent_out = ticket.get("sentiment", "—")
        age_out = f"{ticket.get('age_hours', 0):.1f}h"
        queue_out = f"Queue: {obs['queue_size']} remaining"

    return ticket_id_out, subject_out, body_out, cat_out, sent_out, age_out, queue_out, state_json, result_msg


# ---------------------------------------------------------------------------
# Grader runner
# ---------------------------------------------------------------------------

def run_all_graders():
    results = []
    for task in ["easy", "medium", "hard"]:
        r = run_grader(task)
        results.append(r)
    table = "| Task   | Steps | Score  |\n|--------|-------|--------|\n"
    for r in results:
        table += f"| {r['task']:<6} | {r['steps']:<5} | {r['score']:.4f} |\n"
    return table, json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CustomerSupportEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🎫 CustomerSupportEnv — Ticket Triage OpenEnv

An AI agent environment for learning to triage customer support tickets.
Select a task difficulty, reset the environment, and make triage decisions step-by-step.
    """)

    with gr.Tab("🕹️ Interactive Episode"):
        with gr.Row():
            task_dd = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Task Difficulty")
            seed_num = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("🔄 Reset", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current Ticket")
                t_id = gr.Textbox(label="Ticket ID", interactive=False)
                t_subject = gr.Textbox(label="Subject", interactive=False)
                t_body = gr.Textbox(label="Body", lines=3, interactive=False)
                t_cat = gr.Textbox(label="Category", interactive=False)
                t_sent = gr.Textbox(label="Sentiment", interactive=False)
                t_age = gr.Textbox(label="Age", interactive=False)
                t_queue = gr.Textbox(label="Queue Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Your Action")
                a_priority = gr.Dropdown(
                    ["low", "medium", "high", "critical"],
                    value="medium",
                    label="Assign Priority"
                )
                a_response = gr.Textbox(label="Response to Customer", lines=4,
                                        placeholder="Write your customer-facing response here...")
                a_escalate = gr.Checkbox(label="Escalate to Human Agent")
                a_resolve = gr.Checkbox(label="Mark as Resolved")
                step_btn = gr.Button("📨 Submit Action", variant="primary")
                result_box = gr.Textbox(label="Step Result", lines=7, interactive=False)

        state_json = gr.Code(label="Environment State (JSON)", language="json")

        reset_btn.click(
            reset_env,
            inputs=[task_dd, seed_num],
            outputs=[t_id, t_subject, t_body, t_cat, t_sent, t_age, t_queue, state_json, result_box],
        )
        step_btn.click(
            take_step,
            inputs=[task_dd, seed_num, t_id, a_priority, a_response, a_escalate, a_resolve],
            outputs=[t_id, t_subject, t_body, t_cat, t_sent, t_age, t_queue, state_json, result_box],
        )

    with gr.Tab("📊 Run Graders"):
        gr.Markdown("Runs the built-in rule-based agent against all three task difficulties with seed=42.")
        grade_btn = gr.Button("▶ Run All Graders", variant="primary")
        grade_table = gr.Markdown()
        grade_json = gr.Code(label="Full Results JSON", language="json")
        grade_btn.click(run_all_graders, outputs=[grade_table, grade_json])

    with gr.Tab("📖 Environment Spec"):
        gr.Markdown("""
## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `current_ticket.ticket_id` | string | Unique ticket identifier |
| `current_ticket.subject` | string | Ticket subject line |
| `current_ticket.body` | string | Full ticket body |
| `current_ticket.category` | enum | billing / technical / account / shipping / general |
| `current_ticket.sentiment` | enum | polite / neutral / confused / frustrated / angry / panicked |
| `current_ticket.age_hours` | float | Hours since ticket was created |
| `queue_size` | int | Remaining tickets to process |

## Action Space (AgentAction)
| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Must match current ticket ID |
| `assigned_priority` | enum | low / medium / high / critical |
| `response` | string | Customer-facing message |
| `escalate` | bool | Escalate to human agent |
| `resolve` | bool | Close ticket after responding |

## Reward Breakdown
| Component | Max | Description |
|-----------|-----|-------------|
| Priority accuracy | 0.40 | How close your priority is to ground truth |
| Response quality | 0.35 | Relevance, length, empathy |
| SLA compliance | 0.25 | Resolution speed vs deadline |
        """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
