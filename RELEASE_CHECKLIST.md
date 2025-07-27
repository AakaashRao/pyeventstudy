# Release Checklist for EventStudyPy

## Pre-release Checklist

- [x] Update version in `setup.py` and `pyproject.toml`
- [x] Update `CHANGELOG.md` with release date and changes
- [x] Ensure all tests pass (`pytest tests/`)
- [x] Update README.md with current information
- [x] Check for hardcoded paths or debug code
- [x] Verify LICENSE file is present
- [x] Create .gitignore file
- [x] Move debug scripts to separate directory
- [x] Add basic API documentation

## Package Structure
```
python_package/
├── eventstudypy/          # Main package code
│   ├── __init__.py
│   ├── event_study.py     # Core functionality
│   ├── data_prep.py       # Data preparation utilities
│   ├── estimation.py      # Estimation functions
│   ├── plotting.py        # Plotting functions
│   ├── testing.py         # Hypothesis testing
│   └── example_data.csv   # Example dataset
├── tests/                 # Test suite
│   ├── test_*.py          # Unit tests
│   └── conftest.py        # Test configuration
├── examples/              # Usage examples
│   └── example.py
├── docs/                  # Documentation
│   └── API.md
├── debug_scripts/         # Debug utilities (not distributed)
├── setup.py               # Package setup
├── pyproject.toml         # Modern Python packaging
├── requirements.txt       # Dependencies
├── README.md              # Package documentation
├── LICENSE                # MIT License
├── CHANGELOG.md           # Version history
├── MANIFEST.in            # Distribution manifest
└── .gitignore             # Git ignore file
```

## Building the Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build source distribution and wheel
python -m build

# Check the distribution
twine check dist/*
```

## Publishing

### Test PyPI (recommended first)
```bash
twine upload --repository testpypi dist/*
# Test installation: pip install -i https://test.pypi.org/simple/ eventstudypy
```

### Production PyPI
```bash
twine upload dist/*
```

## Post-release

- [ ] Tag the release in git: `git tag -a v0.1.0 -m "Initial release"`
- [ ] Push tags: `git push origin v0.1.0`
- [ ] Create GitHub release with changelog
- [ ] Update documentation if needed

## Important Notes

1. The package is a Python port of the R eventstudyr package
2. All core functionality has been tested against R implementation
3. The proxy/IV functionality is not included in this version
4. Python 3.7+ is required
5. Main dependencies: pandas, numpy, pyfixest, matplotlib, seaborn, scipy, statsmodels