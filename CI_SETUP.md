# CI/CD Setup

GitHub Actions setup for testing and publishing to PyPI.

## Setup

1. Get PyPI API token
2. Set it as a repository secret


## Automatic Testing
**Triggers:** Every push to `main` repository

```bash
git push origin main  # Tests run automatically
```

## Publishing to PyPI
**Triggers:** When you push a git tag starting with `v`

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add .
git commit -m "Release v1.0.0"

# 3. Create and push tag
git tag v1.0.0
git push origin v1.0.0  # Publishing starts automatically!
```

## Workflow Files

- `.github/workflows/test.yml` - Run tests on every push/PR
- `.github/workflows/publish.yml` - Publish on git tags
- `ci-test-unit.sh` - Test script
- `ci-packaging.sh` - Build and publish script

## Version Numbering

Use semantic versioning:
- `v1.0.0` - Major release
- `v1.1.0` - New features
- `v1.0.1` - Bug fixes
- `v1.0.0-beta.1` - Pre-release