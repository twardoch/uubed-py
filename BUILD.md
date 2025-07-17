# Build and Release Guide

This document describes the build and release process for the uubed package.

## Overview

The uubed package uses a modern Python build system with the following components:

- **Build System**: `hatch` with `hatch-vcs` for git-tag-based versioning
- **Versioning**: Semantic versioning based on git tags
- **CI/CD**: GitHub Actions with multiplatform builds
- **Testing**: Comprehensive test suite with coverage reporting
- **Release**: Automated releases to PyPI on git tags

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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .[test]
   ```

### Available Scripts

All build and test scripts are located in the `scripts/` directory:

#### Build Script (`scripts/build.py`)
Builds the package using hatch:
```bash
python scripts/build.py
```

Features:
- Automatically installs hatch and hatch-vcs if needed
- Cleans previous build artifacts
- Builds both wheel and source distribution
- Verifies build artifacts
- Uses git-tag-based versioning

#### Test Runner (`scripts/test_runner.py`)
Runs the comprehensive test suite:
```bash
python scripts/test_runner.py
```

Features:
- Installs test dependencies
- Runs pytest with coverage reporting
- Generates HTML and XML coverage reports
- Runs code linting (ruff)
- Runs type checking (mypy)
- Tests package installation and imports
- Generates JUnit XML for CI systems

#### Release Script (`scripts/release.py`)
Handles the complete release process:
```bash
# Create a new release
python scripts/release.py v1.2.0

# Test release to TestPyPI
python scripts/release.py --test
```

Features:
- Validates semantic versioning
- Checks git working directory is clean
- Runs full test suite
- Builds the package
- Creates git tags
- Publishes to PyPI or TestPyPI
- Pushes tags to remote repository

#### Build and Test Script (`scripts/build_and_test.py`)
Convenience script for common development tasks:
```bash
# Build and test
python scripts/build_and_test.py --all

# Just build
python scripts/build_and_test.py --build

# Just test
python scripts/build_and_test.py --test

# Create release
python scripts/build_and_test.py --release v1.2.0

# Test release
python scripts/build_and_test.py --test-release
```

## Versioning System

The package uses git-tag-based semantic versioning:

- **Release versions**: `v1.2.3` (clean git state on tag)
- **Development versions**: `v1.2.3.dev4+abc123.d20231215` (commits since last tag)

### Version Format

- `v1.2.3` - Release version
- `v1.2.3.dev4+abc123.d20231215` - Development version where:
  - `1.2.3` - Last release version
  - `dev4` - 4 commits since last release
  - `abc123` - Git commit hash
  - `d20231215` - Date

### Creating Releases

1. Ensure all changes are committed and pushed
2. Run tests: `python scripts/test_runner.py`
3. Create release: `python scripts/release.py v1.2.3`
4. The script will:
   - Validate the version format
   - Run the full test suite
   - Build the package
   - Create and push git tag
   - Publish to PyPI
   - Create GitHub release

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

### Workflows

#### Test Workflow (`.github/workflows/test.yml`)
- Triggers on: Push to main/develop, Pull requests
- Runs on: Ubuntu, macOS, Windows
- Python versions: 3.10, 3.11, 3.12
- Features:
  - Comprehensive test suite
  - Coverage reporting
  - Security scanning
  - Artifact uploads

#### CI Workflow (`.github/workflows/ci.yml`)
- Triggers on: Push, Pull requests
- Features:
  - Code quality checks (ruff, mypy)
  - Security scanning (bandit, safety)
  - Multiplatform builds
  - Integration tests

#### Release Workflow (`.github/workflows/release.yml`)
- Triggers on: Git tags (v*)
- Features:
  - Multiplatform testing
  - Multiplatform builds
  - PyPI publishing
  - GitHub release creation
  - Artifact management

#### Push Workflow (`.github/workflows/push.yml`)
- Triggers on: Push to main/develop
- Features:
  - Quick linting
  - Basic tests
  - Failure notifications

## Build Configuration

### pyproject.toml
The build configuration is defined in `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "uubed"
dynamic = ["version"]
# ... other metadata

[tool.hatch.version]
source = "vcs"
```

### Key Features

1. **Dynamic Versioning**: Version is determined from git tags
2. **Hatch Build System**: Modern Python build system
3. **VCS Integration**: Direct git integration for versioning
4. **Development Mode**: Editable installs for development

## Testing

### Test Structure

```
tests/
├── test_api.py              # Core API tests
├── test_cli.py              # CLI tests
├── test_encoders.py         # Encoder tests
├── test_validation.py       # Validation tests
├── test_integrations.py     # Integration tests
├── test_streaming.py        # Streaming API tests
├── test_config.py           # Configuration tests
├── test_error_handling.py   # Error handling tests
└── conftest.py             # Test configuration
```

### Running Tests

```bash
# Run all tests
python scripts/test_runner.py

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest --cov=uubed --cov-report=html

# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration
```

### Test Coverage

- Target coverage: >90%
- Coverage reports: HTML (htmlcov/), XML (coverage.xml)
- Coverage uploaded to Codecov in CI

## Release Process

### Manual Release

1. **Prepare for release**:
   ```bash
   # Ensure clean working directory
   git status
   
   # Run tests
   python scripts/test_runner.py
   
   # Update changelog if needed
   vim CHANGELOG.md
   ```

2. **Create release**:
   ```bash
   python scripts/release.py v1.2.3
   ```

3. **Verify release**:
   - Check PyPI: https://pypi.org/project/uubed/
   - Check GitHub releases
   - Test installation: `pip install uubed==1.2.3`

### Automated Release

Releases are automatically triggered by pushing git tags:

```bash
# Create and push tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

This triggers the release workflow which:
1. Runs tests on all platforms
2. Builds packages for all platforms
3. Publishes to PyPI
4. Creates GitHub release
5. Uploads artifacts

### Environment Variables

For releases, set these secrets in GitHub:

- `PYPI_TOKEN`: PyPI API token for publishing
- `TEST_PYPI_TOKEN`: TestPyPI API token for testing

## Troubleshooting

### Common Issues

1. **Version not updating**:
   - Ensure you're in a git repository
   - Check git tags: `git tag -l`
   - Verify hatch-vcs is installed: `pip show hatch-vcs`

2. **Build failures**:
   - Clean build artifacts: `rm -rf dist/ build/`
   - Reinstall hatch: `pip install --force-reinstall hatch hatch-vcs`

3. **Test failures**:
   - Check test dependencies: `pip install -e .[test]`
   - Run individual test files to isolate issues

4. **Release failures**:
   - Ensure working directory is clean
   - Check PyPI credentials
   - Verify version format (semantic versioning)

### Debug Mode

For detailed debugging:

```bash
# Enable verbose output
export HATCH_VERBOSE=1

# Run with debug
python scripts/build.py
```

## Best Practices

1. **Version Management**:
   - Use semantic versioning
   - Tag releases consistently
   - Include descriptive tag messages

2. **Development Workflow**:
   - Run tests before committing
   - Use feature branches
   - Keep changelog updated

3. **Release Workflow**:
   - Test on multiple platforms
   - Verify package contents
   - Update documentation

4. **CI/CD**:
   - Monitor workflow runs
   - Keep secrets updated
   - Review security reports

## Contributing

When contributing to the build system:

1. Test changes locally first
2. Update documentation
3. Ensure backward compatibility
4. Add appropriate tests
5. Update this guide if needed

For more information, see the main [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).