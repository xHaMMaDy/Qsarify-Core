"""Hermetic runtime configuration for the backend contract tests."""

import os
import tempfile
from pathlib import Path


_TEST_RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="qsarify-test-runtime-"))

# Set these before importing backend.app. load_dotenv() intentionally does not
# override values already supplied by the test harness, so local credentials
# and source-tree runtime directories cannot affect the test result.
os.environ["FLASK_ENV"] = "development"
os.environ["FLASK_SECRET_KEY"] = "qsarify-test-secret-key"
os.environ["SUPABASE_URL"] = "https://auth.example.invalid"
os.environ["SUPABASE_JWT_SECRET"] = ""
os.environ["QSARIFY_RUNTIME_DIR"] = str(_TEST_RUNTIME_DIR)
os.environ["QSARIFY_SESSION_DIR"] = str(_TEST_RUNTIME_DIR / "flask_session")
os.environ["QSARIFY_MODEL_FOLDER"] = str(_TEST_RUNTIME_DIR / "models")
os.environ["QSARIFY_UPLOAD_FOLDER"] = str(_TEST_RUNTIME_DIR / "uploads")
os.environ["QSARIFY_LOG_DIR"] = str(_TEST_RUNTIME_DIR / "logs")
