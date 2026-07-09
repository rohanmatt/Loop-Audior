"""
backend.py — The Loop Auditor
Loop engineering engine: PLAN -> EXECUTE -> OBSERVE -> REFLECT -> STOP

Every ingredient from the diagram is a named, inspectable object:
  ContextMemory      -> Context & Memory
  Critic             -> Verification Mechanisms
  LoopEngine.run()   -> Feedback Cycles
  StopConditions     -> Clear Stopping Conditions
  HFClient           -> Automation & Tools

Provider-agnostic. Any OpenAI-compatible endpoint works:
  Groq        https://api.groq.com/openai/v1
  Cerebras    https://api.cerebras.ai/v1
  OpenRouter  https://openrouter.ai/api/v1
  HuggingFace https://router.huggingface.co/v1
  Ollama      http://localhost:11434/v1
"""

import os
import re
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://api.groq.com/openai/v1")
WRITER_MODEL = os.getenv("WRITER_MODEL", "llama-3.3-70b-versatile")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "llama-3.1-8b-instant")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "4.0"))
MIN_DIMENSION_SCORE = float(os.getenv("MIN_DIMENSION_SCORE", "4.0"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
MAX_BACKOFF = float(os.getenv("MAX_BACKOFF", "30"))


# ---------------------------------------------------------------------------
# AUTOMATION & TOOLS
# ---------------------------------------------------------------------------

class ToolCall:
    def __init__(self, name: str, model: str, tokens_in: int, latency_ms: int, retries: int = 0):
        self.name = name
        self.model = model
        self.tokens_in = tokens_in
        self.latency_ms = latency_ms
        self.retries = retries
        self.ts = time.strftime("%H:%M:%S")

    def to_dict(self):
        return self.__dict__


class MissingTokenError(RuntimeError):
    pass


class RateLimitExhausted(RuntimeError):
    pass


class HFClient:
    """
    Thin wrapper over any OpenAI-compatible chat endpoint.

    A tool that dies on the first 429 is not a tool. It backs off,
    it honours retry-after, and it records the fact that it had to.
    """

    def __init__(self, token: str | None = HF_TOKEN, base_url: str = HF_BASE_URL):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.call_log: list[ToolCall] = []
        self.total_retries = 0

    def available(self) -> bool:
        return bool(self.token)

    # -- rate limits ----------------------------------------------------
    @staticmethod
    def _retry_delay(resp: requests.Response, attempt: int) -> float:
        """Providers tell you exactly how long to wait. Listen to them."""
        header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        if header:
            try:
                return float(header) + 0.5
            except ValueError:
                pass

        m = re.search(r"try again in ([\d.]+)\s*s", resp.text)
        if m:
            return float(m.group(1)) + 1.0

        m = re.search(r"try again in ([\d.]+)\s*m", resp.text)
        if m:
            return float(m.group(1)) * 60 + 1.0

        return min(2 ** attempt, MAX_BACKOFF)  # exponential fallback

    # -- the call -------------------------------------------------------
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 1400,
        tool_name: str = "chat",
        on_retry: Callable[[int, float, str], None] | None = None,
    ) -> str:
        if not self.token:
            raise MissingTokenError(
                "No API token found. Create a .env file with HF_TOKEN=... "
                "(Groq keys start with gsk_, from console.groq.com/keys)"
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        retries = 0
        t_start = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
            except requests.exceptions.RequestException as e:
                wait = min(2 ** attempt, MAX_BACKOFF)
                retries += 1
                self.total_retries += 1
                if on_retry:
                    on_retry(attempt + 1, wait, f"network: {type(e).__name__}")
                time.sleep(wait)
                continue

            # rate limited — wait exactly as long as told, then retry
            if resp.status_code == 429:
                wait = min(self._retry_delay(resp, attempt), MAX_BACKOFF)
                retries += 1
                self.total_retries += 1
                if on_retry:
                    on_retry(attempt + 1, wait, "rate limit (429)")
                time.sleep(wait)
                continue

            # transient server errors — same treatment
            if resp.status_code in (500, 502, 503, 504):
                wait = min(2 ** attempt, MAX_BACKOFF)
                retries += 1
                self.total_retries += 1
                if on_retry:
                    on_retry(attempt + 1, wait, f"server {resp.status_code}")
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                raise MissingTokenError("401 — token rejected. Check HF_TOKEN in .env.")

            if resp.status_code == 402:
                raise RuntimeError(
                    "402 — credits depleted on this provider. "
                    "Switch HF_BASE_URL in .env (Groq is free: https://api.groq.com/openai/v1)"
                )

            if resp.status_code != 200:
                raise RuntimeError(f"API {resp.status_code}: {resp.text[:400]}")

            # success
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            self.call_log.append(
                ToolCall(
                    name=tool_name,
                    model=model,
                    tokens_in=len(system) + len(user),
                    latency_ms=int((time.time() - t_start) * 1000),
                    retries=retries,
                )
            )
            return content

        raise RateLimitExhausted(
            f"Gave up after {MAX_RETRIES} attempts on {tool_name}. "
            f"Try CRITIC_MODEL=llama-3.1-8b-instant or lower MAX_ITERATIONS in .env."
        )


# ---------------------------------------------------------------------------
# CONTEXT & MEMORY
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    iteration: int
    kind: str          # "draft" | "critique" | "instruction"
    content: str


@dataclass
class ContextMemory:
    """
    The agent forgets. This does not.
    Every draft and every critique is appended here and re-injected
    on the next iteration. This is the difference between a chain and a loop.
    """
    goal: str
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, iteration: int, kind: str, content: str):
        self.entries.append(MemoryEntry(iteration, kind, content))

    def latest_draft(self) -> str | None:
        for e in reversed(self.entries):
            if e.kind == "draft":
                return e.content
        return None

    def latest_critique(self) -> str | None:
        for e in reversed(self.entries):
            if e.kind == "critique":
                return e.content
        return None

    def render(self, last_n: int = 3, budget: int = 3000) -> str:
        """
        Memory is not a dump. It is a budget.
        Send the last critique in full; truncate older drafts hard.
        """
        if not self.entries:
            return "(empty — first iteration)"

        chunks = []
        for e in self.entries[-last_n:]:
            cap = budget if e.kind != "draft" else budget // 2
            body = e.content if len(e.content) <= cap else e.content[:cap] + "\n… [truncated]"
            chunks.append(f"[iter {e.iteration} · {e.kind}]\n{body}")
        return "\n\n".join(chunks)

    def size(self) -> int:
        return sum(len(e.content) for e in self.entries)


# ---------------------------------------------------------------------------
# VERIFICATION MECHANISMS
# ---------------------------------------------------------------------------

RUBRIC = {
    "evidence": "Every material claim is grounded in a named source, figure, or explicit reasoning. No vague assertions.",
    "specificity": "Uses concrete numbers, named competitors, dated events. Avoids generic filler.",
    "decision_value": "A CEO could act on this. Surfaces a tradeoff, a risk, or a recommendation — not just description.",
    "structure": "Scannable. Clear sections. No wall of text. Nothing repeated.",
    "honesty": "States what is unknown or uncertain rather than papering over gaps.",
}


CRITIC_SYSTEM = """You are a ruthless editorial critic for board-level competitive briefs.
You are NOT the author. You do not rewrite. You judge, and you tell the author exactly what to change.

Score each rubric dimension 1-5. Be harsh. A first draft almost never deserves above 3.
A 5 means a Fortune 500 board would accept it without edits.

OUTPUT CONTRACT — violate this and your review is discarded:
- Emit ONE JSON object. First character '{', last character '}'.
- No markdown fence. No preamble. No trailing commentary.
- Every string value stays on a single line. No literal newlines inside strings.
- Do not use double-quotes inside string values. Use single quotes if you must quote.
- failures: at most 4 items, each under 20 words.
- fix_instruction: one sentence, under 60 words.

Exact shape:
{"scores":{"evidence":0.0,"specificity":0.0,"decision_value":0.0,"structure":0.0,"honesty":0.0},"failures":["..."],"fix_instruction":"..."}"""


@dataclass
class Critique:
    scores: dict[str, float]
    failures: list[str]
    fix_instruction: str
    raw: str
    degraded: bool = False

    @property
    def average(self) -> float:
        return round(sum(self.scores.values()) / max(len(self.scores), 1), 2)

    @property
    def weakest(self) -> str:
        return min(self.scores, key=self.scores.get) if self.scores else "unknown"

    def to_dict(self):
        d = asdict(self)
        d["average"] = self.average
        d["weakest"] = self.weakest
        return d


class Critic:
    """The one who writes is not the one who checks."""

    def __init__(self, client: HFClient, model: str = CRITIC_MODEL):
        self.client = client
        self.model = model

    def review(self, goal: str, draft: str, attempts: int = 3, on_retry=None) -> Critique:
        rubric_text = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
        base_user = (
            f"GOAL:\n{goal}\n\n"
            f"RUBRIC:\n{rubric_text}\n\n"
            f"DRAFT TO JUDGE:\n---\n{draft}\n---\n\n"
            "Return the JSON now."
        )

        last_err = None
        raw = ""
        for attempt in range(1, attempts + 1):
            user = base_user
            if attempt > 1:
                user = (
                    f"{base_user}\n\n"
                    f"YOUR PREVIOUS OUTPUT WAS INVALID JSON ({last_err}).\n"
                    "Emit ONLY the JSON object. Start with '{' and end with '}'. "
                    "No markdown, no commentary. Escape every double-quote inside a string. "
                    "Keep every string on ONE line — no literal newlines inside strings."
                )
            raw = self.client.chat(
                model=self.model,
                system=CRITIC_SYSTEM,
                user=user,
                temperature=0.0 if attempt > 1 else 0.1,
                max_tokens=700,
                tool_name=f"critic.review[try{attempt}]",
                on_retry=on_retry,
            )
            try:
                return self._parse(raw)
            except Exception as e:
                last_err = str(e)[:120]

        # Verification must never take the loop down. Degrade, don't crash.
        return self._salvage(raw, last_err)

    # -- parsing ---------------------------------------------------------
    @classmethod
    def _parse(cls, raw: str) -> Critique:
        data = cls._extract_json(raw)
        scores = {}
        for k, v in (data.get("scores") or {}).items():
            try:
                scores[k] = max(1.0, min(5.0, float(v)))
            except (TypeError, ValueError):
                continue
        if not scores:
            raise ValueError("no usable scores in JSON")
        for k in RUBRIC:
            scores.setdefault(k, 1.0)
        scores = {k: scores[k] for k in RUBRIC}

        failures = data.get("failures") or []
        if isinstance(failures, str):
            failures = [failures]

        return Critique(
            scores=scores,
            failures=[str(f) for f in failures][:6],
            fix_instruction=str(data.get("fix_instruction") or "Improve the weakest dimension."),
            raw=raw,
            degraded=False,
        )

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Strip fences, find the outermost balanced object, then repair common LLM sins."""
        text = raw.strip()
        text = re.sub(r"```(?:json)?", "", text).strip()

        start = text.find("{")
        if start == -1:
            raise ValueError("no '{' in critic output")

        depth, end, in_str, esc = 0, -1, False, False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError("unbalanced braces in critic output")

        blob = text[start:end]

        err = None
        for candidate in (blob, Critic._repair(blob)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                err = e
        raise ValueError(f"unrepairable JSON: {err}")

    @staticmethod
    def _repair(blob: str) -> str:
        b = blob
        b = re.sub(r",\s*([}\]])", r"\1", b)                   # trailing commas
        b = b.replace("\u201c", '"').replace("\u201d", '"')    # smart double quotes
        b = b.replace("\u2018", "'").replace("\u2019", "'")    # smart single quotes
        b = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "0", b)
        b = re.sub(r"\bTrue\b", "true", b)
        b = re.sub(r"\bFalse\b", "false", b)
        b = re.sub(r"\bNone\b", "null", b)

        # escape literal newlines/tabs that appear inside string values
        out, in_str, esc = [], False, False
        for ch in b:
            if esc:
                out.append(ch); esc = False; continue
            if ch == "\\":
                out.append(ch); esc = True; continue
            if ch == '"':
                in_str = not in_str; out.append(ch); continue
            if in_str and ch == "\n":
                out.append("\\n"); continue
            if in_str and ch == "\t":
                out.append("\\t"); continue
            if in_str and ch == "\r":
                continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _salvage(raw: str, err: str | None) -> Critique:
        """
        Last resort. Pull whatever numbers we can find by key name.
        The loop keeps running; the trace records that verification degraded.
        """
        scores = {}
        for dim in RUBRIC:
            m = re.search(rf'"?{dim}"?\s*[:=]\s*([0-5](?:\.\d+)?)', raw)
            scores[dim] = float(m.group(1)) if m else 1.0

        return Critique(
            scores=scores,
            failures=[
                f"Critic emitted malformed JSON after retries ({err}). Scores recovered by pattern match."
            ],
            fix_instruction=(
                "The critic could not be parsed cleanly. Treat the lowest score as the target: "
                "add concrete evidence, name specific figures, and state what is unknown."
            ),
            raw=raw,
            degraded=True,
        )


# ---------------------------------------------------------------------------
# THE WRITER SUB-AGENT
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You write one-page competitive briefs for a CEO.
Dense, specific, decision-oriented. Use headers. No filler, no throat-clearing.
If you do not know a figure, say so explicitly rather than inventing one.

Structure:
## Position
## Moat
## Threats
## The call
"""


class Writer:
    def __init__(self, client: HFClient, model: str = WRITER_MODEL):
        self.client = client
        self.model = model

    def draft(self, goal: str, memory: ContextMemory, iteration: int, on_retry=None) -> str:
        if iteration == 1:
            user = f"TASK:\n{goal}\n\nWrite the brief."
            temp = 0.7
        else:
            user = (
                f"TASK:\n{goal}\n\n"
                f"MEMORY (your previous work and the critic's judgement):\n"
                f"{memory.render()}\n\n"
                f"The critic rejected your last draft. Fix it. "
                f"Do not start over — repair the specific failures named above. "
                f"Return the full revised brief."
            )
            temp = 0.4

        return self.client.chat(
            model=self.model,
            system=WRITER_SYSTEM,
            user=user,
            temperature=temp,
            max_tokens=1400,
            tool_name=f"writer.draft[iter{iteration}]",
            on_retry=on_retry,
        )


# ---------------------------------------------------------------------------
# CLEAR STOPPING CONDITIONS
# ---------------------------------------------------------------------------

@dataclass
class StopConditions:
    max_iterations: int = MAX_ITERATIONS
    pass_threshold: float = PASS_THRESHOLD
    min_dimension: float = MIN_DIMENSION_SCORE

    def check(self, critique: Critique, iteration: int) -> tuple[bool, str | None]:
        """Returns (should_stop, rule_that_fired)."""
        if (
            critique.average >= self.pass_threshold
            and min(critique.scores.values()) >= self.min_dimension
        ):
            return True, (
                f"QUALITY MET — avg {critique.average} >= {self.pass_threshold} "
                f"and no dimension below {self.min_dimension}"
            )
        if iteration >= self.max_iterations:
            return True, f"BUDGET EXHAUSTED — hit max_iterations={self.max_iterations}"
        return False, None

    def describe(self) -> list[str]:
        return [
            f"Exit if average score ≥ {self.pass_threshold}",
            f"…AND every dimension ≥ {self.min_dimension}",
            f"Hard exit at {self.max_iterations} iterations regardless",
        ]


# ---------------------------------------------------------------------------
# FEEDBACK CYCLES — the loop itself
# ---------------------------------------------------------------------------

@dataclass
class Iteration:
    n: int
    draft: str
    critique: Critique
    stopped: bool
    stop_reason: str | None

    def to_dict(self):
        return {
            "n": self.n,
            "draft": self.draft,
            "critique": self.critique.to_dict(),
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }


@dataclass
class LoopResult:
    goal: str
    iterations: list[Iteration]
    final_draft: str
    stop_reason: str
    tool_calls: list[dict]
    memory_size: int
    total_retries: int = 0

    @property
    def score_curve(self) -> list[float]:
        return [it.critique.average for it in self.iterations]

    @property
    def passed(self) -> bool:
        return self.stop_reason.startswith("QUALITY MET")


class LoopEngine:
    def __init__(self, client: HFClient | None = None, stop: StopConditions | None = None):
        self.client = client or HFClient()
        self.writer = Writer(self.client)
        self.critic = Critic(self.client)
        self.stop = stop or StopConditions()

    # -- PLAN -----------------------------------------------------------
    @staticmethod
    def plan(company: str, focus: str) -> str:
        return (
            f"Produce a one-page board-level competitive brief on **{company}**.\n"
            f"Analytical focus: {focus}\n"
            f"The reader is a CEO deciding whether to compete, partner, or ignore."
        )

    # -- the loop -------------------------------------------------------
    def run(
        self,
        company: str,
        focus: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> LoopResult:
        emit = on_event or (lambda phase, data: None)

        def on_retry(attempt: int, wait: float, why: str):
            emit("retry", {"attempt": attempt, "wait": round(wait, 1), "why": why})

        goal = self.plan(company, focus)
        memory = ContextMemory(goal=goal)
        emit("plan", {"goal": goal})

        iterations: list[Iteration] = []
        stop_reason = "unknown"

        for n in range(1, self.stop.max_iterations + 1):
            # EXECUTE
            emit("execute_start", {"iteration": n})
            draft = self.writer.draft(goal, memory, n, on_retry=on_retry)
            memory.add(n, "draft", draft)
            emit("execute_done", {"iteration": n, "draft": draft})

            # OBSERVE
            emit("observe_start", {"iteration": n})
            critique = self.critic.review(goal, draft, on_retry=on_retry)
            memory.add(
                n,
                "critique",
                f"scores={critique.scores}\nfailures={critique.failures}\nfix={critique.fix_instruction}",
            )
            emit("observe_done", {"iteration": n, "critique": critique.to_dict()})

            # STOP?
            should_stop, reason = self.stop.check(critique, n)

            # REFLECT
            if not should_stop:
                memory.add(n, "instruction", critique.fix_instruction)
                emit(
                    "reflect",
                    {"iteration": n, "instruction": critique.fix_instruction,
                     "weakest": critique.weakest},
                )

            iterations.append(Iteration(n, draft, critique, should_stop, reason))

            if should_stop:
                stop_reason = reason
                emit("stop", {"iteration": n, "reason": reason})
                break

        return LoopResult(
            goal=goal,
            iterations=iterations,
            final_draft=iterations[-1].draft,
            stop_reason=stop_reason,
            tool_calls=[c.to_dict() for c in self.client.call_log],
            memory_size=memory.size(),
            total_retries=self.client.total_retries,
        )


# ---------------------------------------------------------------------------
# THE CONTROL: prompt engineering, one shot, no loop
# ---------------------------------------------------------------------------

def single_shot(client: HFClient, company: str, focus: str) -> tuple[str, Critique]:
    """
    Same model. Same prompt. No loop.
    This is the number the loop is compared against.
    """
    goal = LoopEngine.plan(company, focus)
    draft = Writer(client).draft(goal, ContextMemory(goal=goal), iteration=1)
    critique = Critic(client).review(goal, draft)
    return draft, critique


if __name__ == "__main__":
    engine = LoopEngine()
    result = engine.run("Figma", "AI-native design tooling")
    print(f"iterations: {len(result.iterations)}")
    print(f"curve:      {result.score_curve}")
    print(f"retries:    {result.total_retries}")
    print(f"stop:       {result.stop_reason}")