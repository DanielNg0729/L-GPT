# OverrideFocus800

`OverrideFocus800` contains 800 target-distinct sessions, all labeled
`intent_override`. It reuses the fixed held-out `Unseen800` target population and profiles;
only scenario assignment changes. The purpose is to increase the override sample size from
120 to 800 without introducing a second population assumption.

This is an internal dialogue-policy stress set, not an estimate of organizer-private
performance. The companion contradiction probe alters only the initial old-value message
and suppresses its later re-disclosure. The target-derived new intent and the evaluator's
override timing remain unchanged.

Build reproducibly:

```bash
python -m robustness.override_focus.build_override_focus_set
```
