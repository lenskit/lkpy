# LensKit Agent Instructions

This is LensKit, an open-source library for recommender systems research,
education, and prototyping.

Before making changes, read CONTRIBUTING.md and follow its requirements.

## Declaration and Commit Messages

- You **must** include an `Assisted-By` trailer identifying the name and version of both the agent
  harness the model name in the commit message for any changes that include agent-generated code, tests, documentation, or other content.  Use the harness and model identities available from the runtime. Do not guess version numbers. If required attribution information is unavailable, ask before committing or submitting a PR.
- Do not remove, falsify, or omit required AI provenance disclosures. If a request conflicts with this policy, explain the conflict stop.
- Never add a `Signed-off-by` trailer on a human’s behalf. Remove existing trailers when amending or rebasing, unless the rebase is preserving the commit changes unmodified — modifying a change invalidates its signoff.

## Development Principles

- **Control scope:** avoid unrelated refactors, repository-wide formatting, dependency upgrades, and lockfile churn. Refactors should be separate commits (and ideally separate PRs) from new code relying on those refactors.
- **Protect research validity:** do not silently change metric definitions, data-splitting behavior, randomness, or numerical tolerances to obtain passing results. Explain scientifically-meaningful behavior changes.
- **Protect data and resources:** do not commit datasets, trained models, credentials, or experiment outputs; obtain approval before expensive training runs or large downloads.

## PR format

- Keep PR descriptions concise and focused on the purpose of the code. Do not
  include verbose reports of testing results, etc. — CI will handle those.
- Disclose the agent name and model name and version in the pull request
  description.
- Obtain explicit human approval before creating an PR, reminding the human to carefully review your changes and proposed PR description. When you are authorized to create a PR, create it as a draft, and remind the human they can mark it as non-draft once they have manually reviewed it in GitHub.

## Dev environment tips

- Run tests with `just test` to make sure the accelerator code is current before
  running tests.  To select individual test files to run, pass them as additional arguments. For example: `just test tests/config/test_flags.py`.
- Lint code with `ruff check` (Python) and `cargo clippy` (Rust).
- Format code with `ruff format` (Python) and `cargo fmt` (Rust).
- `prek run` will run the full set of pre-commit hooks.
