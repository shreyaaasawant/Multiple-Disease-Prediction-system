"""
Comprehensive Unit Tests for MediPredict Application.

Tests the core business logic (models, predictions, PDF, Gemini) directly via
core.py, which is free of Streamlit side effects and fully unit-testable.
"""
import os
import sys
import warnings

import pytest

# Suppress sklearn InconsistentVersionWarning (models trained on older sklearn)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator.*")

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core  # noqa: E402


# ------------------------------------------------------------------
#  ML MODEL TESTS
# ------------------------------------------------------------------
class TestMLModels:
    @pytest.fixture(scope="class")
    def models(self):
        return core.load_models(ROOT)

    def test_models_load(self, models):
        assert set(models.keys()) == {"diabetes", "heart", "breast"}

    def test_models_have_predict_and_proba(self, models):
        for name, model in models.items():
            assert callable(getattr(model, "predict", None)), f"{name} has no predict()"
            assert callable(getattr(model, "predict_proba", None)), f"{name} has no predict_proba()"

    def test_diabetes_feature_count(self, models):
        assert models["diabetes"].n_features_in_ == 8

    def test_heart_feature_count(self, models):
        assert models["heart"].n_features_in_ == 13

    def test_breast_feature_count(self, models):
        assert models["breast"].n_features_in_ == 30

    def test_diabetes_prediction(self, models):
        result, prob = core.predict_diabetes(models["diabetes"], [6, 148, 72, 35, 0, 33.6, 0.627, 50])
        assert result in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_heart_prediction(self, models):
        result, prob = core.predict_heart(models["heart"], [50, 1, 3, 130, 250, 0, 1, 150, 0, 1.0, 2, 0, 2])
        assert result in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_breast_prediction(self, models):
        sample = [14.0, 20.0, 90.0, 600.0, 0.1, 0.2, 0.2, 0.1, 0.2, 0.06,
                  1.0, 0.9, 6.0, 100.0, 0.007, 0.02, 0.03, 0.01, 0.02, 0.003,
                  15.0, 25.0, 100.0, 800.0, 0.14, 0.3, 0.4, 0.2, 0.3, 0.1]
        result, prob = core.predict_breast(models["breast"], sample)
        assert result in (0, 1)
        assert 0.0 <= prob <= 1.0


# ------------------------------------------------------------------
#  UTILITY FUNCTION TESTS
# ------------------------------------------------------------------
class TestUtilities:
    def test_clean_text(self):
        assert core.clean_text("hello") == "hello"
        # Emoji is NOT representable in latin-1, so it should be stripped.
        assert core.clean_text("café ☕") == "café "

    def test_risk_meta_low(self):
        level, cls, label = core.risk_meta(0.2)
        assert level == "low" and cls == "risk-low" and label == "Low Risk"

    def test_risk_meta_moderate(self):
        level, cls, label = core.risk_meta(0.5)
        assert level == "moderate" and cls == "risk-moderate" and label == "Moderate Risk"

    def test_risk_meta_high(self):
        level, cls, label = core.risk_meta(0.8)
        assert level == "high" and cls == "risk-high" and label == "High Risk"


# ------------------------------------------------------------------
#  PDF REPORT GENERATION TESTS
# ------------------------------------------------------------------
class TestPDFReport:
    def test_build_pdf_creates_file(self):
        user = {
            "name": "Test User", "age": 30, "gender": "Male",
            "date": "2024-01-01", "city": "Test City",
            "state": "Test State", "email": "test@example.com",
        }
        parameters = {"Glucose": 148, "BMI": 33.6}
        suggestions = "**Healthy diet** and exercise are recommended."
        filename = core.build_pdf(user, "Diabetes", "Diabetic", parameters, suggestions)
        try:
            assert os.path.exists(filename), "PDF file was not created"
            assert filename.endswith(".pdf")
        finally:
            if os.path.exists(filename):
                os.remove(filename)


# ------------------------------------------------------------------
#  GEMINI AI INTEGRATION TESTS
# ------------------------------------------------------------------
class TestGeminiAI:
    def test_gemini_module_installed(self):
        import google.generativeai  # noqa
        assert True

    def test_get_gemini_suggestions_returns_string(self):
        # Without a configured key, this returns a graceful error string.
        result = core.get_gemini_suggestions("In one sentence, what is BMI?")
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
