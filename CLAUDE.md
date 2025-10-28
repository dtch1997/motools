Conventions
- `uv add` to add dependencies.
- Follow `pytest` conventions to define tests
  - As far as possible, define tests as pure functions
  - Use pytest fixtures to setup reusable components

Testing philosophy:
- **Unit tests** should use mocks to:
  - Test individual components in isolation
  - Run fast and deterministically
  - Not depend on external services
  - Cover edge cases and error conditions easily
- **Integration tests** should exercise real components to:
  - Verify that components actually work together
  - Catch issues that mocks miss (e.g., API format mismatches)
  - Test against real API behavior (not assumptions)
  - Ensure the system works end-to-end
- Integration tests should be:
  - Marked with `@pytest.mark.integration` and `@pytest.mark.slow`
  - Skippable in CI by default with `-m "not integration"`
  - Use minimal resources (small models, few examples)
  - Skip automatically when required credentials are missing

When addressing an issue, agent should: 
- Read the issue and ask any clarifying questions
- Come up with an implementation plan, add to issue
- Explicitly wait until plan is approved
- Add a comment to issue saying “I’m starting to work on this” or similar
- Spin up a worktree for the implementation
- Write necessary code, while adhering to good coding style and adding type hints
- Ensure unit tests, linters pass
- Submit a PR
- Wait for PR to be reviewed
- Address feedback until PR approved
- Fix any conflicts with main
- (test, lint, …) 
- Merge (default to squashing)
- Clean up the worktree

Frequently-used user commands: 
- "test": run all unit tests and iteratively fix issues until the tests pass. 
- "lint": fix all style issues / ensure linters pass. Refer to `ci.yml` for the appropriate commands.
- "commit", you should git commit the current changes with an informative commit message. 