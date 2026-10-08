# `tests/test_fig05_current_protocol.py`

Regression tests for Fig. 5 current-result validation and trajectory recording. Run `python -m unittest discover -s tests -v` from the repository root. Requires the figure and CPU dynamics test dependencies. Verifies that trajectory callbacks cannot change states or balanced accuracy, changed prediction CSVs are rejected, old EMA protocols are rejected, and OOV genes cannot enter the top-30% evaluation pool. Fixtures are synthetic and not manuscript data.
