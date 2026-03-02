"""
Configuration settings for the project.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Model defaults
DEFAULT_MODEL = "gpt-4"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# Dataset info (update when data is released)
DOMAINS = ["politics", "finance", "public_emergencies"]
