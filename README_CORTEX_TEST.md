# CORTEX — internal EEG certification test

Thank you for helping evaluate CORTEX before it goes wider. This is a
short **adaptive** test of EEG interpretation. Please take it as you
would read studies in routine clinical practice, and report anything
that looks wrong, confusing, or slow.

## 1. Requirements

- **macOS or Linux**
- **Python 3.11** — check with `python3 --version`. If you don't have it,
  install it from
  <https://www.python.org/downloads/release/python-3119/>.

## 2. Take the test

Open a terminal **in this folder** and run this one command:

```
bash run_cortex_test.sh
```

The first time you run it, it sets up a local Python environment (a few
minutes); after that it launches straight away. Always start the test
this way — it uses the bundled environment, so there is no "wrong
Python" to worry about.

A window opens — read the welcome screen, the consent notice, and a
short tutorial, then the test begins. For each EEG recording, choose the
pattern that best matches what you see and press **Confirm** (you can
change your selection before confirming; the number keys 1–6 and Enter
also work). The test is **adaptive** — its length is not fixed (roughly
20–100 recordings) and it ends on its own. Your per-task results appear
on the final screen.

## 3. What gets recorded

Your responses, reaction times, and the engine's skill estimates are
saved locally and uploaded automatically to the study's secure folder —
no action needed from you.

## Reporting bugs

Anything that crashes, looks visually wrong, is confusing, or feels
slow — note it and send it to the study team. That feedback is the whole
point of this internal round.
