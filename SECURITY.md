# Security policy

## Reporting a vulnerability

Please report privately, not in a public issue: use [Report a vulnerability](https://github.com/pokhriyal-anmol/research-copilot/security/advisories/new) on the repository's Security tab.

Include what you found, how to reproduce it, and what it lets an attacker do. This is a one-person project, so expect a first reply within about a week. There is no bounty.

## Scope

In scope: the code in this repository and the public demo (the Render API and the Vercel page).

Out of scope: denial of service by volume (the demo is a free instance with a daily answer cap), and findings in third-party services it uses (Render, Vercel, Neon, Gemini, OpenRouter, arXiv).

## What the demo stores

Questions typed into the demo are kept for 30 days, then deleted. The code has no accounts and sets no cookies.

## Supported versions

Only the `main` branch is deployed and supported.
