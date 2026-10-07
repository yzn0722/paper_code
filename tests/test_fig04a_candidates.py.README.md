# `tests/test_fig04a_candidates.py`

Checks Fig. 4a with one TF, three genes, duplicate pairs, invalid targets, and
self-links. Verifies the TF-restricted candidate denominator, unrestricted mode,
curve precision, the plotted horizontal reference, and exported source-data
counts. No real datasets or figure files are required; PDF export is mocked.

Run from the repository root:

```powershell
python -m unittest discover -s tests -p test_fig04a_candidates.py
```
