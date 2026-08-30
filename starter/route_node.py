"""
Node 1: strict-gated dialogue-act routing for unfamiliar wrappers.

WHAT IT IS FOR, AND WHY A RULE COULD NOT DO IT
----------------------------------------------
The recognition gate is a DETECTOR, not a router. It reports that a message is not one of
the simulator's known shapes; it cannot say what that message MEANS. Two agent behaviours
depend on the meaning rather than the wording:

    override  -> the customer changed their mind, so the rejection set must be CLEARED.
                 Not clearing it keeps excluding products rejected under the OLD intent,
                 which can permanently exclude the true target. A hit-rate failure.
    no_evidence -> the customer stated they have no requirement here, so the turn carries
                 nothing and must not be mined.

Both were handled by literal patterns, and on reworded wrappers both fired 0/1600.

WIDENED REGEXES WERE TRIED FIRST AND ARE NOT SUFFICIENT. Rebuilt honestly from
train-attested vocabulary and measured on a template-disjoint held-out bank:

    signal                        regex        this model
    override -> clear rejection   37.5%            100.0%
    no-evidence -> skip turn       0.0%            100.0%

    false positives (of 6400 / 8000 non-target rows)
    override (safe kind)          0                2
    no-evidence (dangerous kind)  0                0

The no-evidence row is the one that settles it. The test bank says "indifferent",
"nothing to add", "unspecified" where the training bank said "no further preference",
"any choice is fine", "use your judgment" -- zero shared vocabulary between two banks we
generated ourselves. A lexical rule cannot survive that, and a real rewording would share
even less. Semantic classification is the only mechanism that transfers across vocabulary
shift, which is the entire threat model here.

Overall six-way accuracy under the turn mask: 0.9909 on 9,600 held-out rows. The one
material error is buying_opening -> override_opening (85/1600), and it is benign: it
clears a rejection set that is empty at turn 1.

THE GUARANTEE ON CLEAN TRAFFIC
------------------------------
`classify` returns None -- meaning "caller uses V1 unchanged" -- for a recognised message,
a disabled node, a missing checkpoint, or any inference failure. Because the recognition
gate matches 463/463 clean messages, this model is never loaded and never runs a single
inference on clean traffic. That is control flow, not a threshold, so official200 and the
unseen-800 populations are unchanged by construction. Measured cost when it does run:
3.6 ms per inference on CUDA, on unrecognised messages only.

THE TURN MASK IS A RELEASED INVARIANT, NOT A LEARNED ONE. `turn == 1` admits only opening
actions and later turns only reply/update actions. Applying it after the model is what
raised six-way accuracy materially in development, and it costs nothing.

Set `V2_ROUTE=0` to disable.
"""
from __future__ import annotations

import os
from pathlib import Path


def _recognised(message: str) -> bool:
    """Literal recognition gate, imported LAZILY to avoid a circular import.

    `agent` imports this module inside its own top-level import block, so a module-level
    `from submission.agent import recognised` here resolves against a half-initialised
    module and raises -- which silently disabled Node 1 entirely the first time this was
    wired. Deferring the lookup to call time removes the cycle. Callers normally gate on
    recognition themselves; this is the standalone-safety path.
    """
    try:
        from submission.agent import recognised
    except Exception:  # pragma: no cover - standalone use
        try:
            from agent import recognised  # type: ignore
        except Exception:
            return False
    return bool(recognised(message))

# Label order is positional: the checkpoint's config carries generic LABEL_0..LABEL_5, so
# this tuple IS the contract with the trained head. It is the alphabetical order the
# training label encoder produced, and the 0.9909 held-out accuracy confirms the mapping.
LABELS = (
    "buying_opening", "constraint_update", "no_evidence",
    "override_opening", "override_update", "plain_opening",
)
OPENING = frozenset({"buying_opening", "plain_opening", "override_opening"})
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "route_classifier"


class StrictGatedRouteNode:
    """Six-way dialogue-act classifier behind the literal recognition gate."""

    MAX_LEN = 80

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or os.environ.get("V2_ROUTE_MODEL_DIR",
                                                          DEFAULT_MODEL))
        flag = os.environ.get("V2_ROUTE", "1").strip().lower()
        self.enabled = flag not in {"0", "false", "no", "off"}
        self._model = self._tokenizer = self._torch = self._device = None
        self.model_loads = self.inferences = self.failures = 0
        self.actions: dict[str, int] = {}
        self.disabled_reason: str | None = None

    def _disable(self, reason: str) -> None:
        self.disabled_reason = self.disabled_reason or reason
        self._model = self._tokenizer = self._torch = None

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if not self.enabled or self.disabled_reason is not None:
            return False
        if not self.model_dir.is_dir():
            self._disable("model directory missing")
            return False
        try:
            import torch  # Deliberately lazy: never imported on a literal V1 path.
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, local_files_only=True)
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(self._device).eval()
            self._model, self._torch = model, torch
            self.model_loads += 1
            return True
        except Exception as exc:
            self.failures += 1
            self._disable(f"load failed: {type(exc).__name__}")
            return False

    def classify(self, message: str, turn: int) -> str | None:
        """Return a dialogue act, or None meaning "caller proceeds with V1 unchanged"."""
        if _recognised(message) or not self._ensure():
            return None
        try:
            encoded = self._tokenizer([message], padding=True, truncation=True,
                                      max_length=self.MAX_LEN, return_tensors="pt")
            # The tokenizer inherited from the scaffolding asset emits token type IDs
            # while its sequence-classification head does not accept them. Training used
            # this same two-field interface.
            encoded = {k: v.to(self._device) for k, v in encoded.items()
                       if k in {"input_ids", "attention_mask"}}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits[0]
            allowed = OPENING if turn == 1 else frozenset(LABELS).difference(OPENING)
            action = max((l for l in LABELS if l in allowed),
                         key=lambda l: float(logits[LABELS.index(l)]))
            self.inferences += 1
            self.actions[action] = self.actions.get(action, 0) + 1
            return action
        except Exception:
            self.failures += 1
            return None

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model_dir": str(self.model_dir),
                "device": str(self._device) if self._device is not None else None,
                "model_loads": self.model_loads, "inferences": self.inferences,
                "failures": self.failures, "actions": self.actions,
                "disabled_reason": self.disabled_reason}
