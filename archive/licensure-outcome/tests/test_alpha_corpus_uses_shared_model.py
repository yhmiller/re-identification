from pathlib import Path


def test_alpha_script_has_no_private_generator(repo_root):
    source = (repo_root / "scripts" / "build_synthetic_alpha.py").read_text()
    assert "_generate_outcome" not in source


def test_alpha_script_imports_the_shared_model(repo_root):
    source = (repo_root / "scripts" / "build_synthetic_alpha.py").read_text()
    assert "import outcome_model" in source


def test_alpha_script_holds_no_duplicate_coefficients(repo_root):
    source = (repo_root / "scripts" / "build_synthetic_alpha.py").read_text()
    for constant in ("COEF_CGPA", "COEF_TREND", "NOISE_SD"):
        assert f"{constant} =" not in source, f"{constant} should live in outcome_model"
