# Fig. 6a protocol regression tests

Run `python -m unittest discover -s tests -p 'test_fig06a_seeded_protocol.py' -v`.
These CPU tests verify that model seeds do not change input preprocessing
randomness, stale protocols and changed CSV files cannot be reused, and packed
QKV weights are actually reinitialized with distinct model seeds. No external
datasets or checkpoints are required.
