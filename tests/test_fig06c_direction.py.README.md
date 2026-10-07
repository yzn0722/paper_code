# Fig. 6c direction regression checks

Run `python -m unittest discover -s tests -p test_fig06c_direction.py`.
The tests use synthetic fixtures only in temporary directories. They verify
that observed points and pooled null draws both come from query-to-key,
legacy-only files fail, and inconsistent trajectory provenance, missing draws,
stale summaries or incorrect lag windows cannot produce a figure.
The no-error-bar mode keeps the same medians and descriptive lag spread while
setting the displayed error extents to zero.

Scientific outputs are generated separately from the real server inputs.
