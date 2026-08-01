from fastapi.testclient import TestClient

from api.main import app


def test_ingest_starts_a_background_sync_job(mocker):
    mocker.patch("api.main.asyncio.get_event_loop")
    client = TestClient(app)

    response = client.post("/ingest")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "started"
    assert body["job_id"]


def test_run_ingest_job_calls_sync_hansards_and_records_result(mocker):
    sync_mock = mocker.patch(
        "api.main.sync_hansards",
        return_value={"documents_downloaded": 2, "chunks_written": 14, "up_to_date": False},
    )
    from api.main import _INGEST_JOBS, _run_ingest_job

    _INGEST_JOBS["job-1"] = {"status": "started"}
    _run_ingest_job("job-1")

    sync_mock.assert_called_once()
    assert _INGEST_JOBS["job-1"]["status"] == "completed"
    assert _INGEST_JOBS["job-1"]["chunks_written"] == 14


def test_run_ingest_job_records_failure(mocker):
    mocker.patch("api.main.sync_hansards", side_effect=RuntimeError("scrape failed"))
    from api.main import _INGEST_JOBS, _run_ingest_job

    _INGEST_JOBS["job-2"] = {"status": "started"}
    _run_ingest_job("job-2")

    assert _INGEST_JOBS["job-2"]["status"] == "failed"
    assert "scrape failed" in _INGEST_JOBS["job-2"]["error"]


def test_ingest_status_unknown_job_returns_unknown():
    client = TestClient(app)

    response = client.get("/ingest/does-not-exist")

    assert response.status_code == 200
    assert response.json() == {"status": "unknown"}


def test_health_reports_latest_sitting_date(mocker):
    mocker.patch("api.main.count_documents", return_value=42)
    mocker.patch("api.main.get_latest_ingested_date", return_value="2025-02-11")
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "documents_indexed": 42, "latest_sitting_date": "2025-02-11"}
