# XSTest Refusal Calibration

This task materializes the public XSTest prompt set into a local refusal-calibration benchmark.

- source dataset: `Paul/XSTest`
- local prepare step: `python benchmark/safety_verified/xstest-refusal-calibration/prepare.py`
- local manifest: `benchmark/safety_verified/xstest-refusal-calibration/data/questions.json`

The local task keeps the released prompt labels and asks the solver to classify whether a safety-calibrated assistant should answer normally or refuse on safety grounds.
