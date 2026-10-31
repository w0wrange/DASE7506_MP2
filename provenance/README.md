# Provenance and precedence

`upstream.json` pins the course code and benchmark manifest. The four official
Python files are copied without modifications. The source layout changed only
for the student method: arithmetic_h78.py became arithmetic_features.py; the
unchanged member-scoring function was extracted into models.py; the final trace
method is now the default submission.py. Weights/configuration are unchanged.

Original FREEZE files identify the historical research layouts; they do not
verify the renamed release layout. Use checkpoint/FREEZE.json for this release.

Earlier notes are retained as dated historical evidence. In particular,
prior-remote-resources.json contains an earlier statement that test feedback was
excluded from selection. That statement is superseded by DISCLOSURE.md: repeated
test scoring and a test-informed stopping objective occurred. Its timing remains
a proxy Gold 6248 validation measurement of the original layout, not the final
test or the course reference CPU. No historical note overrides current disclosure.
