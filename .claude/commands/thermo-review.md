---
description: Run the thermo-nuclear code quality review on a diff, branch, PR, or path
argument-hint: [branch | PR number | path] (defaults to the working tree vs main)
---

Invoke the `thermo-nuclear-code-quality-review` skill and follow its procedure in
full — every pass, the falsification step, and the completion gate.

Review target: $ARGUMENTS

If the target above is empty, review the uncommitted working tree plus any
commits on the current branch that are not on `main`. State the resolved scope
before you begin reviewing.
