# Build and Release Guide

This document describes the build and release process for the uubed package.

## Overview

The uubed package uses a modern Python build system:

- **Build System**: `hatch` with `hatch-vcs` for git-tag-based versioning
- **Versioning**: Semantic versioning from git tags
- **CI/CD**: GitHub Actions with cross-platform builds
- **Testing**: pytest with coverage, linting, and type checking
- **Release**: Automated PyPI releases triggered by git tags

## Local Development

### Prerequisites

- Python 3.10+
- Git
- Virtual environment support

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/twardoch/uubed.git
   cd uubed
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .[test]
   ```

### Scripts

All scripts are in the `scripts/` directory.

#### Build (`scripts/build.py`)
Builds the package:
```bash
python scripts/build.py
```

Does the job:
- Installs hatch and hatch-vcs if missing
- Cleans old build artifacts
- Creates wheel and source distribution
- Verifies the build
- Uses git tags for versioning

#### Test Runner (`scripts/test_runner.py`)
Runs the test suite:
```bash
python scripts/test_runner.py
```

Handles:
- Test dependencies
- pytest with coverage
- HTML/XML coverage reports
- Code linting (ruff)
- Type checking (mypy)
- Package installation verification
- JUnit XML output for CI

#### Release (`scripts/release.py`)
Manages releases:
```bash
python scripts/release.py v1.2.0      # Production release
python scripts/release.py --test       # Test release to TestPyPI
```

Process:
- Validates version format
- Checks clean git state
- Runs full test suite
- Builds package
- Creates git tags
- Publishes to PyPI/TestPyPI
- Pushes tags to remote

#### Build and Test (`scripts/build_and_test.py`)
Multi-tool wrapper:
```bash
python scripts/build_and_test.py --all          # Build + test
python scripts/build_and_test.py --build        # Just build
python scripts/build_and_test.py --test         # Just test
python scripts/build_and_test.py --release v1.2.0  # Release
python scripts/build_and_test.py --test-release    # Test release
```

## Versioning

Git tags drive versioning.

### Formats

- Release: `v1.2.3`
- Development: `v1.2.3.dev4+abc123.d20231215`

Components of development versions:
- `1.2.3`: Last release tag
- `dev4`: Commits since tag
- `abc123`: Git commit hash
- `d20231215`: Date

### Creating Releases

1. Commit everything
2. Test: `python scripts/test_runner.py`
3. Release: `python scripts/release.py v1.2.3`

The script handles validation, testing, building, tagging, and publishing.

## CI/CD Pipeline

GitHub Actions runs tests and builds across platforms.

### Workflows

#### Test (`.github/workflows/test.yml`)
- Runs on: Push to main/develop, pull requests
- Platforms: Ubuntu, macOS, Windows
- Python: 3.10, 3.11, 3.12
- Includes: Tests, coverage, security scans, artifact uploads

#### CI (`.github/workflows/ci.yml`)
- Runs on: Push, pull requests
- Features: Linting (ruff), type checking (mypy), security (bandit/safety), builds, integration tests

#### Release (`.github/workflows/release.yml`)
- Runs on: Git tags matching `v*`
- Features: Cross-platform testing, builds, PyPI publishing, GitHub releases, artifact management

#### Push (`.github/workflows/push.yml`)
- Runs on: Push to main/develop
- Features: Quick linting, basic tests, failure notifications

## Build Configuration

### pyproject.toml

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "uubed"
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"
```

Key points:
1. Dynamic versioning from git
2. Hatch build backend
3. VCS integration through hatch-vcs
4. Editable installs for development

## Testing

### Structure

```
tests/
├── test_api.py
├── test_cli.py
├── test_encoders.py
├── test_validation.py
├── test_integrations.py
├── test_streaming.py
├── test_config.py
├── test_error_handling.py
└── conftest.py
```

### Commands

```bash
# All tests
python scripts/test_runner.py

# Specific file
python -m pytest tests/test_api.py -v

# With coverage
python -m pytest --cov=uubed --cov-report=html

# Unit tests only
python -m pytest -m unit

# Integration tests only
python -m pytest -m integration
```

### Coverage

- Goal: >90%
- Reports: HTML (`htmlcov/`), XML (`coverage.xml`)
- CI uploads to Codecov

## Release Process

### Manual

1. Check git status and run tests:
   ```bash
   git status
   python scripts/test_runner.py
   ```

2. Release:
   ```bash
   python scripts/release.py v1.2.3
   ```

3. Verify:
   - PyPI package
   - GitHub release
   - Installation test: `pip install uubed==1.2.3`

### Automated

Triggered by git tags:
```bash
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

Workflow:
1. Multi-platform tests
2. Package builds
3. PyPI publish
4. GitHub release creation
5. Artifact uploads

### Secrets

Set in GitHub:
- `PYPI_TOKEN`: Production PyPI API token
- `TEST_PYPI_TOKEN`: TestPyPI API token

## Troubleshooting

### Common Problems

1. **Version not updating**:
   - Confirm git repository
   - List tags: `git tag -l`
   - Check hatch-vcs: `pip show hatch-vcs`

2. **Build failures**:
   - Clean: `rm -rf dist/ build/`
   - Reinstall: `pip install --force-reinstall hatch hatch-vcs`

3. **Test failures**:
   - Install dependencies: `pip install -e .[test]`
   - Run individual files to isolate issues

4. **Release failures**:
   - Clean working directory
   - Valid PyPI credentials
   - Correct semantic version format

### Debug Mode

Enable verbose output:
```bash
export HATCH_VERBOSE=1
python scripts/build.py
```

## Best Practices

1. **Versioning**:
   - Semantic versioning only
   - Consistent git tags
   - Meaningful tag messages

2. **Development**:
   - Test before commit
   - Use feature branches
   - Update changelog

3. **Releases**:
   - Multi-platform testing
   - Verify package contents
   - Update documentation

4. **CI/CD**:
   - Monitor runs
   - Rotate secrets
   - Review security reports

## Contributing

Build system changes require:
1. Local testing
2. Documentation updates
3. Backward compatibility
4. New tests
5. Guide updates

See [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for more.