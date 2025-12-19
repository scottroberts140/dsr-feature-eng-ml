# Testing Guide - dsr-feature-eng-ml

## Running Tests

### Install test dependencies
```bash
pip install -e ".[test]"
```

### Run all tests
```bash
pytest
```

### Run tests with coverage report
```bash
pytest --cov=src/dsr_feature_eng_ml --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_evaluation.py
pytest tests/test_models.py
```

### Run tests matching a pattern
```bash
pytest -k "test_evaluation"
```

### Run tests with verbose output
```bash
pytest -v
```

## Test Structure

Tests are organized by module:
- `tests/test_evaluation.py` - Tests for model evaluation functionality
- `tests/test_models.py` - Tests for model implementations

## Writing Tests

All test files should:
1. Start with `test_` prefix
2. Use pytest conventions
3. Include docstrings explaining what is being tested
4. Use fixtures from `conftest.py` when needed

Example:
```python
import pytest
import numpy as np
from sklearn.datasets import make_classification

@pytest.fixture
def sample_data():
    """Create sample classification data."""
    X, y = make_classification(n_samples=100, random_state=42)
    return {'X': X, 'y': y}

def test_model_training(sample_data):
    """Test model can train on sample data."""
    assert len(sample_data['X']) == len(sample_data['y'])
```

## Coverage Reports

After running tests with coverage, view the HTML report:
```bash
open htmlcov/index.html
```
