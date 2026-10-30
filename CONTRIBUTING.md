# Contributing to LensKit

LensKit welcomes community contributions of code, documentation, bug reports,
and other improvements. Our primary mechanism for reviewing and accepting code
and documentation changes are through GitHub pull requests.

## Development Setup and Testing

See `README.md` for instructions on setting up a development environment to work
on LensKit, setting up pre-commit hooks, and for running the tests. All new
contributions to LensKit must pass the test suite and lint/formatting checks.

## Code Style

- Python, Rust, and shell code is auto-formatted with `prek` hooks.
- Ruff and Clippy lints enforce many good style elements.
- Write clear, concise code with good class, method, and variable names.
- Don't write verbose comments that communicate things that should be clear from
  the function and variable names. Use comments when further explanation is
  required for *why* things need to be done a particular way.
- All public functions, classes, and methods should have docstrings (using the
  Google syntax supported by Sphinx Napoleon) describing their purpose, inputs,
  and behavior.

## Submitting Changes

Submit your changes as a GitHub pull request, preferably with a clean sequence
of commits.  Use meaningful commit messages for your commits, including:

- A short summary on the first line (think: an e-mail subject), followed by a
  blank line and a longer description of the change. If the change is small and
  self-explanatory, only the first line summary is needed.

- All commits of code that was wholly or partly generated with a coding agent
  **must** include an `Assisted-By` trailer identifying the name and version of
  both the coding agent harness and the model. The PR description must also
  disclose the use of coding agents and LLMs to generate the proposed changes
  and/or pull request description.

  A trailer is a line at the end of the commit message with a trailer name and
  a value, like this:

  ```
  metrics: add more property-based tests of rank weights

  Improving the coverage of rank weight definitions by adding more hypothesis
  tests for behavior correctness.

  Assisted-By: Zed Agent 1.26.2 (Claude Sol 5.1)
  ```

  A trailer is not required if the only LLM assistance was code completion.
