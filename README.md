# ◉ The Loop Auditor

**One model. One prompt. The only variable is whether a loop wraps it.**

An AI writes a board-level competitive brief. A second AI — a critic that never writes, only judges — scores it against a five-point rubric and refuses to pass it until it earns the score.

You watch it fail. You watch it fix itself. You watch it stop.

```
iteration 1  ██▌                1.4 / 5   REJECTED
iteration 2  ███▌               1.6 / 5   REJECTED
iteration 3  ████▋              2.6 / 5   PASSED

```

Same model in both rows below. Nothing changed but the system around it.

| | Score |
|---|---|
| Single shot — prompt engineering | **1.4 / 5** |
| Wrapped in a loop — loop engineering | **2.6 / 5** |

---

## Table of contents

1. [Why this exists](#1-why-this-exists)
2. [What a loop actually is](#2-what-a-loop-actually-is)
3. [The five ingredients](#3-the-five-ingredients)
4. [Reading the code](#4-reading-the-code)
5. [Setup](#5-setup)
6. [Running it](#6-running-it)
7. [Things to try](#7-things-to-try)
8. [When loops go wrong](#8-when-loops-go-wrong)
9. [Troubleshooting](#9-troubleshooting)
10. [Extending it](#10-extending-it)

---

## 1. Why this exists

Ask a model to write something. It writes it. It hands the draft to you.

It has no idea whether the draft is any good. It cannot know — nothing checked. The model that wrote it is the only judge available, and it already thinks it did fine, because it stopped.

That is prompt engineering. You are the loop. You read the draft, notice the vague third paragraph, and paste it back with *"be more specific."* You are the feedback mechanism, the memory, and the stopping condition, running in a human body at human speed.

**Loop engineering moves that machinery into the system.**

You stop being the person who prompts the agent. You design the thing that prompts the agent — including the part that tells it *no*.

The shift, in one line:

```
Prompt Engineer  →  Context Engineer  →  Loop Engineer
```

The prompt engineer asks better. The context engineer feeds better. The loop engineer builds a system that **notices it was wrong and goes again**.

This repo is the smallest honest demonstration of that difference I could build.

---

## 2. What a loop actually is

A chain runs once:

```
prompt ──► model ──► output ──► you
```

A loop runs until a condition is met:

```
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
     ① PLAN ──► ② EXECUTE ──► ③ OBSERVE ──► ④ REFLECT
        goal        draft        score        what to fix
                                   │
                                   ▼
                              ⑤ STOP?
                            good enough
                            or out of budget
```

The five phases, precisely:

**① PLAN** — Turn a vague request into an explicit goal. *"Write about Figma"* becomes *"a one-page competitive brief for a CEO deciding whether to compete, partner, or ignore."* A loop cannot terminate against a goal it cannot state.

**② EXECUTE** — Do the work. Call the model, run the tool, write the code. This is the only phase most systems have.

**③ OBSERVE** — Get a signal about the result that **did not come from the thing that produced it**. Tests. A linter. A rubric. A separate model. Anything with the authority to say *no*. If the executor grades its own homework, you do not have a loop. You have a chain wearing a costume.

**④ REFLECT** — Read the signal. Decide what changes. Not *"try harder"* — a specific, named defect and a specific repair. Feed that into memory.

**⑤ STOP** — Two exits, and you need both:
- *Success:* the quality bar was cleared.
- *Budget:* iterations, tokens, or time ran out.

A loop with only a success condition is a bill. A loop with only a budget is a random number generator.

> The loop is where the intelligence lives. Not in the prompt.

---

## 3. The five ingredients

Every one of these is a named object in `backend.py` and a live panel in the app. Nothing is decorative.

### Context & memory
> *The agent forgets. The scratchpad does not.*

The model has no memory between calls. Iteration 3 has never heard of iteration 1. If you don't carry state forward, every pass starts from zero and the loop is just three chains stapled together.

`ContextMemory` accumulates every draft and every critique, then re-injects them. Iteration 3 is told, explicitly: *here is what you wrote, here is why it was rejected, fix that.*

It is also a **budget**, not a dump — old drafts get truncated, the latest critique goes in whole. Naive memory grows until it prices you out of your own loop.

```python
memory.add(n, "draft", draft)
memory.add(n, "critique", f"scores={...} failures={...}")

# next iteration:
user = f"MEMORY:\n{memory.render()}\n\nThe critic rejected your last draft. Fix it."
```

### Verification mechanisms
> *The one who writes is not the one who checks.*

This is the load-bearing ingredient. Everything else is scaffolding around it.

`Critic` is a separate sub-agent with a separate system prompt and a separate job. It cannot rewrite. It can only score five dimensions from 1 to 5, name the failures, and issue one repair instruction.

| Dimension | Question it asks |
|---|---|
| `evidence` | Is every claim grounded, or is this vibes? |
| `specificity` | Named competitors and real numbers, or filler? |
| `decision_value` | Could a CEO act on this? |
| `structure` | Scannable, or a wall of text? |
| `honesty` | Does it admit what it doesn't know? |

The critic is instructed to be harsh: *a first draft almost never deserves above 3.* That harshness is the whole demo. A generous critic passes iteration 1 and you learn nothing.

**Verification does not have to be a model.** A test suite is a verifier. A type checker is a verifier. A schema validator is a verifier. Prefer those where you can — they're cheap, fast, and don't hallucinate. Use a model critic when the quality you care about is a judgement call.

### Feedback cycles
> *Nothing improves without a signal that it was wrong.*

`LoopEngine.run()` is thirty lines. That's the point — the loop is not complicated, it's just usually absent.

```python
for n in range(1, max_iterations + 1):
    draft    = writer.draft(goal, memory, n)            # ② EXECUTE
    critique = critic.review(goal, draft)               # ③ OBSERVE
    stop, reason = stop_conditions.check(critique, n)   # ⑤ STOP?
    if not stop:
        memory.add(n, "instruction", critique.fix_instruction)   # ④ REFLECT
```

Watch the score curve in the app. It goes up. Not because the model got smarter between calls — it's the same weights — but because it was **told exactly what was wrong** and given its own previous attempt to repair.

### Clear stopping conditions
> *A loop without an exit is a bill.*

`StopConditions` fires on either of two rules, and the app tells you which one fired:

```
QUALITY MET      — avg ≥ 4.0 AND no single dimension below 4.0
BUDGET EXHAUSTED — hit max_iterations regardless of score
```

The `AND` matters. Averages hide failures. A brief scoring `[5, 5, 5, 5, 1]` averages 4.2 and is worthless — beautifully written, completely dishonest. The dimension floor catches that.

The budget exit matters more. Agentic systems that loop without a hard ceiling are how people wake up to a four-figure API invoice. **Every loop you ship needs a cap you'd be comfortable paying.**

### Automation & tools
> *A tool that dies on the first 429 is not a tool.*

`HFClient` logs every call: which model, how long, how many retries. Nothing hidden.

It also survives reality. Rate limits happen. Providers 503. Networks drop. The client reads the provider's own `retry-after`, backs off exactly that long, and continues:

```python
if resp.status_code == 429:
    wait = self._retry_delay(resp, attempt)   # provider said: "try again in 4.925s"
    time.sleep(min(wait, MAX_BACKOFF))
    continue
```

On a free tier you *will* hit a 429. The app shows `⏳ backing off 4.9s` and keeps going. The retry counter appears in the final metrics.

That number is the most honest thing on the screen. A demo breaks. A loop backs off.

---

## 4. Reading the code

Two files. Nothing else.

```
backend.py    the loop engine — no Streamlit, importable, testable, headless
app.py        the interface — two tabs: the brief, and the loop
```

`backend.py`, top to bottom:

| Object | Ingredient | What it does |
|---|---|---|
| `HFClient` | Automation & tools | retry, backoff, call log |
| `ContextMemory` | Context & memory | append, truncate, render |
| `Critic` | Verification | score, salvage, refuse |
| `Writer` | — | the executor |
| `StopConditions` | Stopping conditions | two rules, one exit |
| `LoopEngine.run()` | Feedback cycles | the loop itself |
| `single_shot()` | — | the control group |

`single_shot()` is the comparison. Same writer, same prompt, no loop, one critique for the scoreboard. It exists so the demo can't cheat — you see the number the loop had to beat.

Run the engine headless, no UI:

```bash
python backend.py
```
```
iterations: 3
curve:      [2.4, 3.6, 4.6]
retries:    2
stop:       QUALITY MET — avg 4.6 >= 4.0 and no dimension below 4.0     #(edidtable values according to your need)
```

---

## 5. Setup

### Requirements

- Python 3.10+
- An API key from any OpenAI-compatible provider

### Install

```bash
git clone <your-repo> loop-auditor
cd loop-auditor

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Get a free key

**Groq** is the default. Free tier, no credit card, and the fastest inference you'll find.

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign in, create a key
3. Copy it — it starts with `gsk_`

### Configure

```bash
cp .env.example .env
```

Open `.env`, paste your key:

```bash
HF_TOKEN=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_BASE_URL=https://api.groq.com/openai/v1

WRITER_MODEL=llama-3.3-70b-versatile
CRITIC_MODEL=llama-3.1-8b-instant

MAX_ITERATIONS=3
PASS_THRESHOLD=4.0
MIN_DIMENSION_SCORE=4.0
```

`.env` is gitignored. Never commit it.

> **Why is the critic a smaller model?**
> The writer needs muscle — it's producing prose a CEO will read. The critic only fills in a rubric and names failures. An 8B model does that fine, it lives in a **separate rate-limit bucket**, and it roughly halves your token burn. Verification does not have to be expensive to have teeth.

### Any provider works

The code is OpenAI-compatible. Only `.env` changes.

| Provider | `HF_BASE_URL` | `WRITER_MODEL` |
|---|---|---|
| **Groq** *(default)* | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Cerebras | `https://api.cerebras.ai/v1` | `llama-3.3-70b` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free` |
| HuggingFace | `https://router.huggingface.co/v1` | `meta-llama/Llama-3.3-70B-Instruct` |
| Ollama *(local)* | `http://localhost:11434/v1` | `llama3.1:8b` *(set `HF_TOKEN=ollama`)* |

Not one line of Python changes between them. That's what "automation & tools" means when you take it seriously — the loop doesn't care who's underneath.

---

## 6. Running it

```bash
streamlit run app.py
```

Browser opens at `localhost:8501`.

### Tab 1 — The brief

Type a company. Pick an analytical focus. Hit **Run the loop**.

You get the final brief, and beside it, the single-shot control. Two scores. Same model.

### Tab 2 — The loop

This is the tab worth watching.

- **Feedback cycles** — the score curve, and a table of all five dimensions per iteration. Watch which one was dragging the average.
- **Verification mechanisms** — expand each iteration. The critic's scores, the failures it cited, the verdict. `REJECTED. REJECTED. PASS.`
- **Clear stopping conditions** — the three rules, and which one actually fired.
- **Automation & tools** — every model call, its latency, its retries.
- **Context & memory** — for each iteration after the first: *this is what it was told before it wrote.* The fix instruction, verbatim.

### The screenshot

Tab 1, side by side. **2.4 vs 4.6.** Same model, same prompt.

That's the image. That's the whole argument.

---

## 7. Things to try

Break it on purpose. The failures teach more than the successes.

**Remove verification.** Set `PASS_THRESHOLD=1.0`. Iteration 1 passes. You now have prompt engineering with extra steps. Notice the score never moves — because nothing ever said *no*.

**Remove memory.** In `Writer.draft()`, delete the memory block from the iteration-2+ prompt. The loop still runs three times. The score does not improve. Three chains, stapled together. *This is the most instructive edit in the repo.*

**Raise the bar.** `PASS_THRESHOLD=4.8`, `MAX_ITERATIONS=6`. Watch it grind. Watch it hit `BUDGET EXHAUSTED` and stop anyway, honestly, with a mediocre score. **A loop that always succeeds is lying to you.**

**Ask about something obscure.** Try a company the model has never heard of. Watch the `honesty` dimension. A good critic punishes confident fabrication. Does yours?

**Downgrade the critic.** Set `CRITIC_MODEL` to something tiny. Watch the scores get noisy and the loop start chasing its own tail. **The quality ceiling of any loop is the quality of its verifier.**

**Cut the writer, keep the critic.** Set `WRITER_MODEL=llama-3.1-8b-instant` too. Small writer, real critic. It often still converges — just slower, over more iterations. Verification substitutes for capability, up to a point.

---

## 8. When loops go wrong

Three honest failure modes. Any of them will bite you in production.

**The loop is confidently wrong.** A critic can pass a draft that's beautifully argued and factually false. The loop only optimises what the verifier measures. If your rubric doesn't check facts, the loop will not check facts — it will just get *better at not checking facts*. `honesty` is in the rubric here for exactly this reason, and it's still not enough.

**The loop optimises the rubric, not the goal.** Score `structure` heavily and you'll get gorgeous headers over empty prose. This is Goodhart's law with an API bill. Every rubric is a proxy. Watch what the loop learns to game.

**The loop runs unattended and makes mistakes unattended.** *"Done"* is a claim, not a proof. An agent that reports success three iterations in is an agent that convinced its own critic. Read the trace. That's what Tab 2 is for.

And the one nobody says out loud:

> Two people can build the identical loop and get opposite results.
> One moves faster on work they already understand deeply.
> The other stops understanding the work at all.
>
> **The loop cannot tell the difference. You can.**

Loop design isn't easier than prompt engineering. The leverage point just moved.

---

## 9. Troubleshooting

**`402 — credits depleted`**
Your provider's free tier ran out. Switch `HF_BASE_URL` in `.env`. Groq is free. No code changes.

**`429 — rate limit reached`**
Expected on free tiers. The client already handles it — you'll see `⏳ backing off 4.9s` in the status panel and a retry count in the metrics. If it still exhausts:
- `CRITIC_MODEL=llama-3.1-8b-instant` (separate bucket, half the tokens)
- `MAX_ITERATIONS=3`
- `MAX_BACKOFF=60`

**`401 — token rejected`**
Key is wrong, or `.env` isn't being read. Confirm the file is named exactly `.env`, sits beside `app.py`, and has no quotes around the value.

**`JSONDecodeError` in the critic**
Shouldn't happen. The parser strips fences, balances braces, repairs trailing commas, smart quotes, and literal newlines inside strings; retries three times feeding the error back; then **salvages scores by regex and keeps looping**, flagging the iteration `⚠ degraded` in the UI.

That degradation path is deliberate, and it's a design principle worth stating plainly: **a verifier that crashes takes the loop down with it.** Yours should fail loudly and keep running.

**Scores never improve**
Check that memory is actually reaching the writer. Print `memory.render()` inside `Writer.draft()`. If iteration 2's prompt doesn't contain iteration 1's critique, you built a chain.

**Everything passes on iteration 1**
Your critic is too generous, or your threshold is too low. Raise `PASS_THRESHOLD`. Sharpen `CRITIC_SYSTEM`. A critic that never rejects is decoration.

---

## 10. Extending it

The loop doesn't care what task it wraps. Swap the executor and the verifier; keep the skeleton.

| Executor | Verifier | Loop until |
|---|---|---|
| Write code | Run the test suite | tests pass |
| Write SQL | `EXPLAIN` + row-count sanity check | plan is sane |
| Extract structured data | Schema validation | schema validates |
| Summarize a doc | Faithfulness check against source | no unsupported claims |
| Fix a bug | Reproduce it, confirm it's gone | reproduction fails |

Prefer verifiers that **cannot hallucinate**. A passing test suite is a stronger signal than a model saying *"looks good to me."* Where you can replace a model critic with a deterministic check, do it — the loop gets cheaper, faster, and more honest all at once.

Where you can't, use a model. But give it a rubric, forbid it from rewriting, and make it justify every score.

Then set a budget you'd be comfortable paying, and let it run.

---

## Structure

```
loop-auditor/
├── backend.py         the loop engine
├── app.py             the interface
├── .env.example       copy to .env, add your key
├── .env               your key. gitignored. never commit.
├── .gitignore
├── requirements.txt
└── README.md
```

---

## License

MIT. Take it, break it, ship a better one.

---

*Better context. Better feedback. Better loops. Better outcomes.*

*The next competitive advantage isn't what you ask AI — it's how you design the system around it.*
