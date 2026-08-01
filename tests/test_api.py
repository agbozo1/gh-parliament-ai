from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client(mocker):
    # Don't actually dispatch background work; just prove it was scheduled
    # with the right resolved date range.
    mocker.patch("api.main._run_ingest_job")
    return TestClient(app)


def test_ingest_with_explicit_date_range_starts_a_job(client, mocker):
    dispatch = mocker.patch("api.main.asyncio.get_event_loop")

    response = client.post(
        "/ingest", json={"start_date": "2025-02-01", "end_date": "2025-02-28"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "started"
    assert body["job_id"]

    dispatch.return_value.run_in_executor.assert_called_once()
    args = dispatch.return_value.run_in_executor.call_args.args
    # args: (executor, func, job_id, start_date, end_date)
    assert args[3] == datetime(2025, 2, 1)
    assert args[4] == datetime(2025, 2, 28)


def test_ingest_falls_back_to_days_back_when_no_dates_given(client, mocker):
    dispatch = mocker.patch("api.main.asyncio.get_event_loop")

    response = client.post("/ingest", json={"days_back": 7})

    assert response.status_code == 200
    args = dispatch.return_value.run_in_executor.call_args.args
    start_date, end_date = args[3], args[4]
    assert (end_date - start_date).days == 7


def test_ingest_rejects_end_date_before_start_date(client):
    response = client.post(
        "/ingest", json={"start_date": "2025-03-01", "end_date": "2025-01-01"}
    )

    assert response.status_code == 400
    assert "end_date" in response.json()["detail"]


def test_ingest_rejects_only_one_date_set(client):
    response = client.post("/ingest", json={"start_date": "2025-03-01"})

    assert response.status_code == 400
    assert "both" in response.json()["detail"].lower()


def test_ingest_rejects_malformed_date(client):
    response = client.post(
        "/ingest", json={"start_date": "not-a-date", "end_date": "2025-03-01"}
    )

    assert response.status_code == 400
    assert "start_date" in response.json()["detail"]


def test_ingest_status_unknown_job_returns_unknown(client):
    response = client.get("/ingest/does-not-exist")

    assert response.status_code == 200
    assert response.json() == {"status": "unknown"}
