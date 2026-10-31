# Accidental search access — 2026-10-30 (Asia/Hong_Kong)

While preparing recommendations for further accuracy improvements, a recursive
text search intended to locate arithmetic-checking code was scoped to `code/`
instead of Python source files. The search traversed `code/data/` and returned
some test reference text among a large, truncated output.

This was accidental access, not an authorized test evaluation. No test prediction
or scoring was run, and no model, configuration, feature implementation, frozen
checkpoint, or prior report was changed. The exposed test content is excluded
from recommendation evidence, training, tuning, model selection, and examples.
The user was informed in the conversation. This record contains no test items or
answers.

Further source searches must use explicit source-file paths or a Python-only
file filter, excluding data directories. Recommendations in this turn rely on
previously recorded training/validation experiments, the existing feature source,
and public methodological references. The test set must no longer be described
as wholly unread, although it remains unpredicted and unscored.
