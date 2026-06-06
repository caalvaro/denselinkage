# Security Policy

## Supported versions

The latest released `1.x` version receives security fixes.

## Reporting a vulnerability

Please report security issues **privately** via GitHub's
[private vulnerability reporting](https://github.com/caalvaro/denselinkage/security/advisories/new),
or by email to **alvarocarvalho@live.com**. Expect an initial response within a
few days. Please do **not** open a public issue for security problems.

## Notes on the data surface

`denselinkage`'s dependency-free core (numpy + pandas) has no network surface. The
optional `[langchain]` extra is different: `LangChainMatcher` sends record text to
whatever LLM endpoint you configure on the injected model. Review your provider's
data-handling policy before sending sensitive or regulated data, and prefer
blocking (`top_k` / `similarity_threshold`) to minimize what reaches the model.
