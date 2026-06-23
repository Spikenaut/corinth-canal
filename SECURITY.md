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

A repository administrator must complete the following steps once:

1. Log in to [Aikido](https://app.aikido.dev) and connect the `rmems` GitHub
   organization.
2. Select `corinth-canal` as a scanned repository.
3. Generate a CI API key from
   <https://app.aikido.dev/settings/integrations/continuous-integration> and
   store it as a GitHub Actions secret named `AIKIDO_CLIENT_API_KEY`.
4. Copy the Aikido internal repository ID for `corinth-canal` and store it as a
   GitHub Actions variable named `AIKIDO_REPOSITORY_ID`.

No API keys, tokens, or repository IDs are committed to this repository.
