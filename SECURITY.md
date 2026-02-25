# Security Policy

## Supported Versions

We release patches for security vulnerabilities in these versions:

| Version | Supported |
|---------|-----------|
| `>=0.1.0` | ✅ |
| `<0.1.0` | ❌ |

## Reporting a Vulnerability

Report vulnerabilities to **security@meshal.ai**.

Include:
- CVE (if assigned)
- Reproduction steps
- Impact assessment
- Proposed fix (optional)

## Response Timeline

| Severity | Initial Response | Patch |
|----------|------------------|-------|
| Critical | 24h | 72h |
| High | 48h | 7d |
| Medium | 5d | 14d |
| Low | 14d | 30d |

## Security Features

- **Dependency scanning**: Dependabot alerts + CI `npm audit`
- **Code scanning**: GitHub Advanced Security
- **Signing**: Commits GPG-signed
- **Secrets scanning**: GitHub push protection

## Best Practices

- Pin dependencies in production
- Run `pip check` / `npm audit`
- Use virtual environments
- Review CHANGELOG.md for security fixes

Thanks for helping keep event-discovery secure!

| Version | Supported |
|---------|-----------|
| Latest (npm) | ✅ Supported |
| 0.x.x | ❌ End of life |

## Reporting a Vulnerability

We take the security of our software seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue
2. Email: **security@morphism.systems**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Your contact information

### What to Expect

| Timeline | Action |
|----------|--------|
| 24 hours | Acknowledgment of report |
| 72 hours | Initial assessment |
| 7 days | Severity determination |
| 30 days | Patch development (critical) |
| 90 days | Full disclosure |

## Severity Levels

| Level | Response Time | Example |
|-------|--------------|---------|
| Critical | 24 hours | Remote code execution, data breach |
| High | 48 hours | Authentication bypass |
| Medium | 7 days | Information disclosure |
| Low | 30 days | Documentation issues |

## Security Features

### Governance Validation
- All changes are validated against governance rules (Tenet 26)
- Drift detection prevents unauthorized modifications (Tenet 26)

### Audit Trail
- Complete commit history with conventional commits
- Governance compliance verification

### Dependency Scanning
- Automated vulnerability scanning in CI/CD
- Regular dependency updates

## Best Practices for Users

1. **Keep updated** — Always use the latest version
2. **Validate configs** — Use `morphism validate` before deployment
3. **Audit regularly** — Run drift detection periodically
4. **Limit permissions** — Follow principle of least privilege

## Acknowledgments

We thank the security community for helping us keep Morphism secure.

