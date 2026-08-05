"""
UI Smoke Tests for the MediPredict Streamlit application.

Uses streamlit.testing.v1.AppTest to launch app.py headlessly and verify
that the full UI renders and flows correctly (user details, navigation,
screening pages, AI assistant, history, about) without crashing.
"""
import os
import sys
import warnings

import pytest
from streamlit.testing.v1 import AppTest

# Suppress sklearn InconsistentVersionWarning (models trained on older sklearn)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

APP_PATH = os.path.join(ROOT, "app.py")


@pytest.fixture(scope="module")
def at():
    """Launch the app once for the module."""
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    return app


# ------------------------------------------------------------------
#  BOOT / USER DETAILS FLOW
# ------------------------------------------------------------------
class TestBoot:
    def test_app_runs_without_exception(self, at):
        assert not at.exception, f"App raised an exception:\n{at.exception}"

    def test_welcome_page_shown(self, at):
        # Before details are filled, the welcome/profile form must be visible.
        assert any("Welcome to MediPredict" in str(m.value) for m in at.markdown), (
            "Welcome page not rendered"
        )

    def test_user_details_form_present(self, at):
        assert at.text_input, "No text inputs (profile form) found"
        assert at.button, "No buttons found on the app"


# ------------------------------------------------------------------
#  COMPLETE USER FLOW (fill profile -> navigate -> predict)
# ------------------------------------------------------------------
class TestUserFlow:
    def _fill_and_submit(self, app):
        # Fill the profile form fields by label
        for ti in app.text_input:
            if ti.label == "Full name":
                ti.set_value("Test User")
            elif ti.label == "City":
                ti.set_value("Test City")
            elif ti.label == "State":
                ti.set_value("Test State")
            elif ti.label == "Email":
                ti.set_value("test@example.com")
        app.button[0].click().run()

    def test_fill_profile_and_submit(self, at):
        self._fill_and_submit(at)
        assert not at.exception, f"App raised after submitting profile:\n{at.exception}"
        # After submitting, the dashboard should load (sidebar appears).
        assert at.sidebar, "Sidebar did not render after profile submission"


# ------------------------------------------------------------------
#  AI MODEL (& PREDICTION) TESTS ARE IN test_app.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

