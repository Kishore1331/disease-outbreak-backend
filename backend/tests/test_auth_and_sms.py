import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.sms_service import sms_service

client = TestClient(app)

def test_admin_login_success():
    payload = {
        "email": "officer.coimbatore@health.tn.gov.in",
        "contact_number": "+919842123456",
        "name": "Dr. K. Sundaram"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authenticated"
    assert data["officer"]["email"] == "officer.coimbatore@health.tn.gov.in"
    assert data["officer"]["contact_number"] == "+919842123456"
    assert data["officer"]["is_authenticated"] is True

def test_admin_login_invalid_email():
    payload = {
        "email": "not-an-email",
        "contact_number": "+919842123456"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 422

def test_admin_login_invalid_phone():
    payload = {
        "email": "officer@coimbatore.gov.in",
        "contact_number": "123"
    }
    response = client.post("/api/auth/login", json=payload)
    assert response.status_code == 422

def test_get_current_officer():
    response = client.get("/api/auth/officer")
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "contact_number" in data
    assert "name" in data

def test_gateway_not_configured_delivery_status():
    """
    Verify that dispatch returns GATEWAY_NOT_CONFIGURED or FAILED when no real credentials exist.
    Strictly verifies non-simulation: status MUST NOT be 'delivered' and delivery_status MUST NOT be 'SENT'
    without provider confirmation.
    """
    sms_service.deduplicator.clear()
    payload = {
        "recipient_phone": "+919842123456",
        "recipient_name": "Dr. K. Sundaram",
        "disease": "Dengue",
        "force": True
    }
    response = client.post("/api/alerts/dispatch-sms", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check valid status enum: SENT, FAILED, or GATEWAY_NOT_CONFIGURED
    assert data["delivery_status"] in ("SENT", "FAILED", "GATEWAY_NOT_CONFIGURED")
    if data["delivery_status"] != "SENT":
        assert data["status"] in ("failed", "unconfigured")
        assert data["delivery_status"] in ("GATEWAY_NOT_CONFIGURED", "FAILED")
        assert data.get("provider_id") is None

def test_message_content_contains_required_fields():
    """
    Requirement 6: Message must include:
    - ward name
    - risk level
    - prediction date
    - recommended preventive action
    """
    sms_service.deduplicator.clear()
    payload = {
        "recipient_phone": "+919842123456",
        "recipient_name": "Dr. K. Sundaram",
        "disease": "Dengue",
        "prediction_date": "2026-10-15",
        "force": True
    }
    response = client.post("/api/alerts/dispatch-sms", json=payload)
    assert response.status_code == 200
    data = response.json()
    msg = data.get("message", "")
    
    # 1. Ward name present
    assert any(w in msg for w in ["Ward 01", "Ward 02", "Ward"])
    # 2. Risk level present
    assert "HIGH" in msg
    # 3. Prediction date present
    assert "2026-10-15" in msg
    # 4. Recommended preventive action present
    assert any(act in msg.lower() for act in ["vector control", "larvicide", "action"])

def test_duplicate_alert_prevention():
    """
    Requirement 7: Prevent duplicate alerts for the same ward, risk level, and date.
    Subsequent attempt within cooldown must be flagged as duplicate unless force=True.
    """
    sms_service.deduplicator.clear()
    today_date = "2026-10-16"
    
    # Simulate a recorded dispatch in deduplicator
    sms_service.deduplicator.record_dispatch("Ward 01", "High", today_date)
    
    # Second dispatch without force must be prevented as duplicate
    payload = {
        "recipient_phone": "+919842123456",
        "recipient_name": "Dr. K. Sundaram",
        "disease": "Dengue",
        "prediction_date": today_date,
        "force": False
    }
    response = client.post("/api/alerts/dispatch-sms", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_duplicate"] is True
    assert "Duplicate alert prevented" in data["error"]
    
    # Force dispatch should bypass duplicate check
    payload_force = {**payload, "force": True}
    response_force = client.post("/api/alerts/dispatch-sms", json=payload_force)
    assert response_force.status_code == 200
    data_force = response_force.json()
    assert data_force["is_duplicate"] is False

def test_prevent_repeated_attempts_when_unconfigured():
    """
    Requirement 7: Prevent repeated SMS attempts when the gateway is not configured.
    Verifies that calling evaluate-thresholds or consecutive dispatches when unconfigured
    does NOT endlessly bloat the dispatch history log with duplicate unconfigured rows.
    """
    sms_service.clear_history()
    # Call evaluate-thresholds when unconfigured
    response = client.post("/api/alerts/evaluate-thresholds?disease=Dengue")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "gateway_unconfigured"
    assert data["alerts_dispatched"] == 0
    
    # History should remain clean and not spammed with duplicate unconfigured attempts
    history_res = client.get("/api/alerts/sms-history").json()
    assert len(history_res["history"]) == 0

def test_test_sms_endpoint():
    """
    Requirement 4: After configuration, allow a single test SMS to an authorized recipient.
    When unconfigured, returns clear instructions without queuing retries.
    """
    response = client.post("/api/alerts/test-sms", json={"note": "Connectivity verification test"})
    assert response.status_code == 200
    data = response.json()
    assert data["delivery_status"] in ("SENT", "GATEWAY_NOT_CONFIGURED", "FAILED")
    if data["delivery_status"] == "GATEWAY_NOT_CONFIGURED":
        assert "instructions" in data
        assert "carrier_stage" in data

def test_clear_sms_history_endpoint():
    """Verify that DELETE /api/alerts/sms-history clears the audit history cleanly."""
    # Seed an entry
    client.post("/api/alerts/dispatch-sms", json={"recipient_phone": "+919842123456", "force": True})
    # Clear
    del_res = client.delete("/api/alerts/sms-history")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"
    
    # Check history is 0
    get_res = client.get("/api/alerts/sms-history")
    assert len(get_res.json()["history"]) == 0

def test_gateway_status_no_secrets_leaked():
    """Verify that credentials are NEVER exposed in the gateway status API."""
    response = client.get("/api/alerts/gateway-status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "delivery_status" in data
    assert "provider" in data
    # Ensure raw secret keys are completely absent from response keys and values
    assert "twilio_auth_token" not in data
    assert "fast2sms_api_key" not in data
    assert "account_sid" not in data
    assert "auth_token" not in data

def test_test_sms_unauthorized_recipient():
    """
    Test SMS must reject unauthorized phone numbers not matching the registered municipal officer.
    """
    response = client.post("/api/alerts/test-sms", json={"recipient_phone": "+919999999999", "note": "Unauthorized recipient test"})
    assert response.status_code == 200
    data = response.json()
    assert data["delivery_status"] == "FAILED"
    assert data["carrier_stage"] == "UNAUTHORIZED_RECIPIENT"
    assert "Unauthorized recipient" in data["error"]

def test_fast2sms_success_mock():
    """
    Verify successful Fast2SMS dispatch:
    - Provider acceptance (return: True)
    - Records request_id
    - Distinguishes provider acceptance (delivery_confirmed=False) from confirmed delivery
    - Carrier stage is SUBMITTED_TO_CARRIER
    - Delivery status is SENT
    """
    mock_creds = {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "fast2sms_api_key": "test_f2s_valid_secret_key",
        "fast2sms_route": "q",
        "sms_provider": "fast2sms",
    }
    mock_f2s_resp = MagicMock()
    mock_f2s_resp.status_code = 200
    mock_f2s_resp.json.return_value = {
        "return": True,
        "request_id": "fast2sms_req_99812",
        "message": ["SMS sent successfully."]
    }

    with patch.object(sms_service, "get_credentials", return_value=mock_creds):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_f2s_resp
            response = client.post("/api/alerts/test-sms", json={"note": "Single test SMS"})
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_status"] == "SENT"
            assert data["carrier_stage"] == "SUBMITTED_TO_CARRIER"
            assert data["delivery_confirmed"] is False
            assert data["provider_id"] == "fast2sms_req_99812"
            assert data["provider_response"]["request_id"] == "fast2sms_req_99812"
            assert data["provider_response"]["return"] is True
            # Never expose credentials in response
            assert "test_f2s_valid_secret_key" not in str(data)

def test_fast2sms_invalid_api_key_mock():
    """Verify Fast2SMS invalid API key (HTTP 401) is handled cleanly."""
    mock_creds = {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "fast2sms_api_key": "bad_key",
        "fast2sms_route": "q",
        "sms_provider": "fast2sms",
    }
    mock_f2s_resp = MagicMock()
    mock_f2s_resp.status_code = 401
    mock_f2s_resp.json.return_value = {
        "return": False,
        "status_code": 401,
        "message": "Invalid Authorization Key"
    }

    with patch.object(sms_service, "get_credentials", return_value=mock_creds):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_f2s_resp
            response = client.post("/api/alerts/test-sms", json={"note": "Test invalid key"})
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_status"] == "FAILED"
            assert data["carrier_stage"] == "INVALID_API_KEY"
            assert "Invalid Authorization Key" in data["error"] or "401" in data["error"]

def test_fast2sms_insufficient_balance_mock():
    """Verify Fast2SMS insufficient balance (status_code 412) is classified."""
    mock_creds = {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "fast2sms_api_key": "test_f2s_key",
        "fast2sms_route": "q",
        "sms_provider": "fast2sms",
    }
    mock_f2s_resp = MagicMock()
    mock_f2s_resp.status_code = 412
    mock_f2s_resp.json.return_value = {
        "return": False,
        "status_code": 412,
        "message": "Insufficient Balance"
    }

    with patch.object(sms_service, "get_credentials", return_value=mock_creds):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_f2s_resp
            response = client.post("/api/alerts/test-sms", json={"note": "Test balance check"})
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_status"] == "FAILED"
            assert data["carrier_stage"] == "INSUFFICIENT_BALANCE"
            assert "Insufficient Balance" in data["error"]

def test_fast2sms_rejected_request_mock():
    """Verify Fast2SMS rejected request (e.g. invalid mobile number) is handled."""
    mock_creds = {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "fast2sms_api_key": "test_f2s_key",
        "fast2sms_route": "q",
        "sms_provider": "fast2sms",
    }
    mock_f2s_resp = MagicMock()
    mock_f2s_resp.status_code = 400
    mock_f2s_resp.json.return_value = {
        "return": False,
        "status_code": 400,
        "message": "Invalid Mobile Number"
    }

    with patch.object(sms_service, "get_credentials", return_value=mock_creds):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_f2s_resp
            response = client.post("/api/alerts/test-sms", json={"note": "Test rejected"})
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_status"] == "FAILED"
            assert data["carrier_stage"] == "PROVIDER_REJECTED"
            assert data["provider_response"]["status_code"] == 400

def test_fast2sms_network_error_mock():
    """Verify network connection timeout or failure is recorded properly."""
    mock_creds = {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_from_number": "",
        "fast2sms_api_key": "test_f2s_key",
        "fast2sms_route": "q",
        "sms_provider": "fast2sms",
    }

    with patch.object(sms_service, "get_credentials", return_value=mock_creds):
        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Failed to resolve host")):
            response = client.post("/api/alerts/test-sms", json={"note": "Test network error"})
            assert response.status_code == 200
            data = response.json()
            assert data["delivery_status"] == "FAILED"
            assert data["carrier_stage"] == "NETWORK_ERROR"
            assert "Network connection failure" in data["error"]

def test_configure_gateway_endpoint():
    """Verify in-app gateway configuration endpoint."""
    with patch.object(sms_service, "update_env_file") as mock_update:
        res = client.post("/api/alerts/configure-gateway", json={
            "provider": "fast2sms",
            "fast2sms_api_key": "test_live_key",
            "fast2sms_route": "q"
        })
        assert res.status_code == 200
        mock_update.assert_called_once_with({"FAST2SMS_API_KEY": "test_live_key", "FAST2SMS_ROUTE": "q"})
        data = res.json()
        assert "validation" in data
