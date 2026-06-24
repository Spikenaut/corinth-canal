# Security

## Reporting vulnerabilities

If you discover a security issue in `corinth-canal`, please open a private
vulnerability report via GitHub:

<https://github.com/rmems/corinth-canal/security/advisories/new>

Please do not open public issues for undisclosed security vulnerabilities.

## Automated security scanning

This repository uses [Aikido](https://aikido.dev) for continuous security
scanning. Aikido runs on every pull request and on the default branch (`main`)
to detect:

- dependency vulnerabilities
- static analysis security findings (SAST)
- infrastructure-as-code misconfigurations (IaC)
- exposed secrets

Pull requests are expected to pass the Aikido security scan before merge. The
scan results appear as a check run or PR comment once Aikido is enabled for the
repository.

## Enabling Aikido for this repository

Aikido offers a free tier and can be enabled without committing any secrets or
tokens to this repository:

1. Go to the [Aikido GitHub App](https://github.com/apps/aikido-security) page
   (or install it from the GitHub Marketplace).
2. Install the app for the `rmems` organization and grant access to the
   `corinth-canal` repository.
3. Log in to [Aikido](https://app.aikido.dev) with GitHub and select
   `corinth-canal` as an active scanned repository.
4. In Aikido, enable **PR checks** for the repository so pull requests receive
   scan results and default-branch pushes are scanned automatically.

No API keys, tokens, or repository IDs are committed to this repository.

## Optional CI release gating

Aikido also supports a CI-based release gate through the
`aikido-api-client` CLI. That approach requires a paid Aikido plan that
includes CI gating, a CI API key, and the Aikido internal repository ID. It is
not required for the default PR and branch scanning described above.
