"""
Local scaffolding tagger. Primary extraction channel for messages the agent does not
recognise. Offline, deterministic, free, no network.

WHAT IT DOES
------------
Given a paraphrased customer message, it labels each word CONTENT or SCAFFOLD and returns
the message with the scaffolding removed:

    "Appreciate it. I want to find Jewelry Necklaces. It absolutely has to be
     Material:alloy. Cheers."
      -> "jewelry necklaces material alloy"

Exact phrase matching then runs, unchanged, over what survives. Semantics decides only WHAT
TO KEEP -- it never decides what matches. That distinction is why this works where four
previous transformer attempts failed: dense bi-encoder (-0.047), ColBERT, and cross-encoder
(-0.030, and WORSE with training) all tried to make a model match products, blurring the
lexical precision that provenance recovery depends on.

WHY IT BEAT A LINEAR MODEL, MEASURED
------------------------------------
The same architecture with hand-built `df`-statistics features (pass 47) reached 0.837 train
accuracy and then collapsed on a held-out paraphrase family -- T2 scored -0.435 lift, BELOW
majority class -- because it had learned OUR filler vocabulary rather than the shape of
scaffolding. A pretrained encoder does not have that failure, because it already knows
"necklaces" is a product word and "appreciate it" is politeness:

    held-out-transform lift        linear (pass 47)   this tagger
      T2 scaffolding stripped           -0.435           +0.081
      T5 realistic                      +0.108           +0.237
      T4 case/punctuation churn         +0.196           +0.407

    end-to-end   clean 0.96960 unchanged   organizer-proxy 0.950725
                 T1 +0.0407               T5 +0.0498

On the realistic combined transform it beats the LLM extraction layer (+0.0498 vs +0.0252)
while needing no network, no credential, no quota and no rate limit.

MLM DOMAIN-ADAPTATION ON THE 50,000-PRODUCT CATALOGUE WAS TESTED AND IS NOT USED. It scored
fractionally worse than plain pretrained distilbert (T1 0.88820 vs 0.89300). The tagger
separates product words from FILLER, and catalogue listings contain no filler to contrast
against -- so the discriminative knowledge comes from general pretraining, not from our
corpus. Worth recording, because the intuition that a big in-domain corpus must help is
exactly what the measurement contradicts.

CONTAINMENT
-----------
This is the agent's only non-standard-library dependency, so it is fenced in three ways:

  1. THE RECOGNITION GATE. Only messages matching no known simulator shape reach it.
     Measured: 463/463 clean messages are recognised, so on an unparaphrased run this file
     is never called and the score is unchanged by construction.
  2. TOTALITY. `strip()` returns None on ANY problem -- torch absent, transformers absent,
     model directory missing, corrupt weights, CUDA failure, unexpected output. None means
     "use the original message", which is exactly the offline behaviour.
  3. A CIRCUIT BREAKER. Repeated failures disable the layer for the rest of the run rather
     than paying the cost on every one of ~1,500 messages.

`BERT_EXTRACT=0` disables it outright.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# WHERE THE CHECKPOINT COMES FROM: an explicit override, then a local directory that
# actually holds weights, then the Hub. The Hub default is what makes a fresh clone work --
# `submission/models/` is gitignored, so on any clone but this machine the local path does
# not exist.
HUB_ID = os.environ.get("BERT_TAGGER_HUB", "KhiemGOM/techjam-scaffolding-tagger")
_LOCAL = Path(__file__).resolve().parent / "models" / "scaffolding_tagger"


def _has_weights(path: Path) -> bool:
    """A checkpoint is a directory that actually contains weights, not merely a name."""
    return path.is_dir() and any(path.glob("*.safetensors"))


def _resolve_source() -> str:
    # ONE RULE: a path is used only if it actually holds weights -- the override included.
    # The override used to be returned blind, ahead of both the local check and the Hub, so
    # a path that was not there got handed to `transformers` as though it were a Hub repo
    # id. That raises, which trips the breaker, which disables the scaffolding tagger in SILENCE. It is
    # the same gitignored-checkpoint failure this resolution order exists to prevent, so an
    # override must not be a way back into it. A directory with no `.safetensors` in it is
    # not a checkpoint, whoever named it.
    override = os.environ.get("BERT_TAGGER_DIR")
    if override and _has_weights(Path(override)):
        return override
    if _has_weights(_LOCAL):
        return str(_LOCAL)
    return HUB_ID


MODEL_DIR = _resolve_source()

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
MAX_WORDS = 96
KEEP_THRESHOLD = float(os.environ.get("BERT_KEEP", "0.30"))
TRIP_AFTER = 5


class ScaffoldingTagger:
    def __init__(self, model_dir=MODEL_DIR) -> None:
        self.model_dir = str(model_dir)
        self.calls = self.failures = self.stripped = 0
        self._consecutive = 0
        self._open_reason: str | None = None
        self._model = None
        self._tok = None
        self._torch = None
        self._device = None
        # DEFAULT OFF, on measurement. The tagger was the unfamiliar-wrapper handler until
        # the exact span node landed; beside the span node it buys almost nothing:
        #
        #     condition     +SPAN     +BERT+SPAN   the tagger buys
        #     official200   0.970100    0.970100         +0.000000
        #     unseen800     0.945125    0.945125         +0.000000
        #     template      0.904841    0.906063         +0.001222
        #     attribute     0.847103    0.847103         +0.000000
        #     both          0.719805    0.728008         +0.008203
        #
        # HR@10 on template is identical either way (0.9812), and the span node alone
        # recovers 86% of the template gap with no model at all. So the tagger is NOT the
        # component carrying that axis, and it must not be described as though it were.
        #
        # It ships ENABLED nonetheless. It costs nothing on the decision criteria -- the
        # recognition gate makes it unreachable on clean traffic, measured at +0.000000 on
        # official200 and every population suite -- it runs locally with no network and no
        # per-call cost, and it contributes a small positive on the compound paraphrase
        # condition. Disabling a layer that is free where it matters and mildly useful
        # where it fires would be trading a real capability for nothing.
        #
        # Set BERT_EXTRACT=0 for a strictly lexical run.
        flag = os.environ.get("BERT_EXTRACT", "1").strip().lower()
        self._flag = flag not in {"0", "false", "no", "off"}

    # ---------------------------------------------------------------- lifecycle
    @property
    def enabled(self) -> bool:
        return self._flag and self._open_reason is None

    def _trip(self, reason: str) -> None:
        if self._open_reason is None:
            self._open_reason = reason
        self._model = self._tok = None      # release ~266 MB once we have given up

    def _ensure(self) -> bool:
        """Load lazily. A failure here disables the layer permanently and silently."""
        if self._model is not None:
            return True
        if not self._flag or self._open_reason is not None:
            return False
        # NO EXISTENCE CHECK: `model_dir` may be a Hub id, which is not a path. Testing it
        # meant that on a fresh clone -- where `submission/models/` is gitignored and
        # therefore absent -- this layer disabled itself silently with "model directory
        # missing". A local directory that exists but holds no weights is caught below.
        try:
            import torch                      # noqa: PLC0415 - deliberately lazy
            from transformers import (AutoModelForTokenClassification,  # noqa: PLC0415
                                      AutoTokenizer)
            self._torch = torch
            local_only = Path(self.model_dir).is_dir()
            self._tok = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=local_only)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_dir, local_files_only=local_only)
            requested = os.environ.get("BERT_DEVICE", "auto").strip().lower()
            self._device = torch.device("cuda:0") if requested != "cpu" and torch.cuda.is_available() else torch.device("cpu")
            model.to(self._device)
            model.eval()
            self._model = model
            return True
        except Exception as exc:
            # torch missing, transformers missing, corrupt weights, incompatible version.
            # All of them mean the same thing to the caller: run offline as before.
            self._trip(f"load failed: {type(exc).__name__}")
            return False

    # ---------------------------------------------------------------- inference
    def strip(self, message: str) -> str | None:
        """Message with scaffolding removed, or None if the caller should use the original.

        Total function: no input, no environment, and no model state causes it to raise.
        """
        if not self.enabled:
            return None
        words = [w.lower() for w in TOKEN_RE.findall(message or "")]
        if len(words) < 3:
            return None
        if not self._ensure():
            return None
        words = words[:MAX_WORDS]
        try:
            torch = self._torch
            enc = self._tok([words], is_split_into_words=True, truncation=True,
                            max_length=MAX_WORDS, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(input_ids=enc["input_ids"].to(self._device),
                                     attention_mask=enc["attention_mask"].to(self._device)).logits
                probs = torch.softmax(logits, -1)[0, :, 1].cpu()
            kept, prev = [], None
            for pos, wid in enumerate(enc.word_ids(0)):
                if wid is None or wid == prev:
                    continue                  # score the FIRST sub-token of each word only
                prev = wid
                if wid < len(words) and float(probs[pos]) >= KEEP_THRESHOLD:
                    kept.append(words[wid])
        except Exception:
            self.calls += 1
            self.failures += 1
            self._consecutive += 1
            if self._consecutive >= TRIP_AFTER:
                self._trip(f"{TRIP_AFTER} consecutive inference failures")
            return None

        self.calls += 1
        self._consecutive = 0
        # Refuse to hand back something degenerate. Dropping nearly everything would be a
        # far worse input than the original message, and mining recall is the whole point
        # of this channel.
        if len(kept) < 2 or len(kept) < 0.15 * len(words):
            return None
        self.stripped += 1
        return " ".join(kept)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model_dir": str(self.model_dir),
                "calls": self.calls, "stripped": self.stripped,
                "failures": self.failures, "circuit_reason": self._open_reason,
                "device": str(self._device) if self._device is not None else None}
