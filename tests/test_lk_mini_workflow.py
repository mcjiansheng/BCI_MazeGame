from scripts.validate_lk_mini_workflow import run_workflow


def test_complete_lk_mini_software_workflow(tmp_path):
    report = run_workflow(tmp_path / "workflow")
    assert report["status"] == "passed"
    assert report["device_contract"]["channel_count"] == 16
    assert all(report["checks"].values())
    assert report["online"]["decisions"] == report["online"]["validation_trials"]
    assert report["online"]["accuracy"] >= 0.9
