import logging
import re
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.services.auth_service import auth_service
from backend.services.sms_service import sms_service
from backend.services.data_service import data_service

logger = logging.getLogger("epidemiology.api.auth")
router = APIRouter(prefix="/api", tags=["Admin Authentication & Alert Notifications"])

class AdminLoginRequest(BaseModel):
    email: str = Field(..., description="Municipal officer email address", json_schema_extra={"example": "officer@coimbatore.gov.in"})
    contact_number: str = Field(..., description="Registered contact number for alert dispatches", json_schema_extra={"example": "+919842123456"})
    name: Optional[str] = Field(None, description="Officer display name", json_schema_extra={"example": "Dr. K. Sundaram"})
    password: Optional[str] = Field(None, description="Admin password or PIN")

class DispatchAlertRequest(BaseModel):
    recipient_phone: Optional[str] = Field(None, description="Optional override contact number; defaults to registered officer number")
    recipient_name: Optional[str] = Field(None, description="Optional override officer name")
    disease: Optional[str] = Field("Dengue", description="Target disease for alert")
    custom_message: Optional[str] = Field(None, description="Custom message text override")
    prediction_date: Optional[str] = Field(None, description="Date associated with prediction / outbreak alert")
    force: Optional[bool] = Field(False, description="Set True to bypass duplicate alert suppression")

class TestSmsRequest(BaseModel):
    recipient_phone: Optional[str] = Field(None, description="Optional recipient phone number; defaults to authorized municipal officer")
    note: Optional[str] = Field(None, description="Optional test note")

@router.post("/auth/login", status_code=status.HTTP_200_OK)
async def admin_login(payload: AdminLoginRequest):
    """
    Authenticate municipal officer with email ID and contact number.
    Validates email format and mobile number, registers officer as target for high-risk SMS alerts.
    """
    try:
        officer = auth_service.authenticate(
            email=payload.email,
            contact_number=payload.contact_number,
            name=payload.name
        )
        return {
            "status": "authenticated",
            "message": f"Welcome, {officer['name']}. Contact number registered for high-risk alerts.",
            "officer": officer
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during authentication."
        )

@router.get("/auth/officer", status_code=status.HTTP_200_OK)
async def get_current_officer():
    """Returns currently authenticated municipal officer profile and alert contact number."""
    return auth_service.get_current_officer()

@router.post("/auth/logout", status_code=status.HTTP_200_OK)
async def admin_logout():
    """Logs out the active municipal officer session."""
    auth_service.logout()
    return {"status": "logged_out", "message": "Officer logged out successfully."}

@router.post("/alerts/dispatch-sms", status_code=status.HTTP_200_OK)
async def dispatch_high_risk_sms_alert(payload: Optional[DispatchAlertRequest] = None):
    """
    Sends emergency SMS notification to the registered municipal officer when wards are high risk.
    CRITICAL: Does not simulate success. Invokes configured SMS provider (Twilio/Fast2SMS)
    or honestly returns real delivery failure status (SENT, FAILED, or GATEWAY_NOT_CONFIGURED).
    """
    officer = auth_service.get_current_officer()
    
    target_phone = (payload.recipient_phone if payload and payload.recipient_phone else officer.get("contact_number"))
    target_name = (payload.recipient_name if payload and payload.recipient_name else officer.get("name", "Chief Municipal Health Officer"))
    disease = payload.disease if payload and payload.disease else "Dengue"
    custom_message = payload.custom_message if payload else None
    prediction_date = payload.prediction_date if payload else None
    force = payload.force if payload and payload.force is not None else False

    # Fetch real ward risk classifications
    raw_wards = data_service.get_wards()
    all_wards = [w.model_dump() if hasattr(w, "model_dump") else (w.dict() if hasattr(w, "dict") else dict(w)) for w in raw_wards]
    high_risk_wards = [w for w in all_wards if w.get("risk") == "High"]
    if not high_risk_wards:
        # Check predefined threshold: cases >= 24
        high_risk_wards = [w for w in all_wards if w.get("cases", 0) >= 24]
    if not high_risk_wards:
        # Fallback to elevated wards if none strictly >= 24
        high_risk_wards = [w for w in all_wards if w.get("cases", 0) >= 15]

    result = await sms_service.dispatch_high_risk_alert(
        officer_phone=target_phone,
        officer_name=target_name,
        high_risk_wards=high_risk_wards,
        disease=disease,
        custom_message=custom_message,
        prediction_date=prediction_date,
        force=force,
        is_automated=False
    )
    return result

@router.post("/alerts/test-sms", status_code=status.HTTP_200_OK)
async def send_single_test_sms(payload: Optional[TestSmsRequest] = None):
    """
    After configuration, allows a single test SMS to an authorized recipient.
    Strictly verifies:
    1. Recipient is authorized (must be the authenticated municipal officer).
    2. SMS Gateway is configured.
    Only dispatches when both requirements are met.
    """
    officer = auth_service.get_current_officer()
    if not officer or not officer.get("is_authenticated"):
        return {
            "status": "unauthorized",
            "delivery_status": "FAILED",
            "carrier_stage": "UNAUTHORIZED",
            "delivery_confirmed": False,
            "recipient_phone": None,
            "error": "Unauthorized: Municipal health officer must be logged in before sending test SMS alerts.",
            "provider_response": None
        }

    officer_phone = officer.get("contact_number")
    requested_phone = (payload.recipient_phone if payload and payload.recipient_phone else officer_phone)
    officer_name = officer.get("name", "Chief Municipal Health Officer")

    # Verify that recipient is authorized (matches registered municipal officer)
    officer_digits = re.sub(r"[^\d]", "", officer_phone or "")[-10:]
    requested_digits = re.sub(r"[^\d]", "", requested_phone or "")[-10:]
    if officer_digits and requested_digits and officer_digits != requested_digits:
        return {
            "status": "unauthorized",
            "delivery_status": "FAILED",
            "carrier_stage": "UNAUTHORIZED_RECIPIENT",
            "delivery_confirmed": False,
            "recipient_phone": requested_phone,
            "error": f"Unauthorized recipient: Test SMS can only be sent to the registered municipal officer ({officer_phone}).",
            "provider_response": None
        }

    validation = sms_service.validate_gateway()
    if not validation["configured"]:
        return {
            "status": "unconfigured",
            "delivery_status": "GATEWAY_NOT_CONFIGURED",
            "carrier_stage": "GATEWAY_NOT_CONFIGURED",
            "delivery_confirmed": False,
            "recipient_phone": officer_phone,
            "error": "SMS Gateway is not configured. Add FAST2SMS_API_KEY to backend/.env first.",
            "instructions": validation.get("instructions"),
            "provider_response": None
        }

    return await sms_service.dispatch_test_sms(
        officer_phone=officer_phone,
        officer_name=officer_name,
        note=payload.note if payload else None
    )

@router.post("/alerts/evaluate-thresholds", status_code=status.HTTP_200_OK)
async def evaluate_thresholds_and_dispatch(
    disease: str = "Dengue",
    force: bool = False
):
    """
    Scans all monitored municipal wards against predefined high-risk thresholds:
    (composite risk score >= 65.0 or cases >= 24).
    Dispatches SMS alerts with ward name, risk level, prediction date, and recommended action.
    Suppresses repeated attempts when gateway is unconfigured to prevent spamming the audit log.
    """
    validation = sms_service.validate_gateway()
    if not validation["configured"]:
        return {
            "status": "gateway_unconfigured",
            "delivery_status": "GATEWAY_NOT_CONFIGURED",
            "message": "Gateway not configured in backend/.env. Automated repeated attempts suppressed.",
            "disease": disease,
            "alerts_dispatched": 0,
            "results": []
        }

    results = await sms_service.check_and_alert_thresholds(
        disease=disease,
        force=force
    )
    return {
        "status": "evaluated",
        "disease": disease,
        "alerts_dispatched": len(results),
        "results": results
    }

@router.get("/alerts/sms-history", status_code=status.HTTP_200_OK)
async def get_sms_dispatch_history():
    """Returns the actual SMS delivery audit trail and real statuses."""
    gateway_info = sms_service.validate_gateway()
    return {
        "gateway": gateway_info["provider"],
        "is_configured": gateway_info["configured"],
        "delivery_status": "READY" if gateway_info["configured"] else "GATEWAY_NOT_CONFIGURED",
        "registered_officer_phone": auth_service.get_current_officer().get("contact_number"),
        "history": sms_service.get_history(),
        "total_dispatched": len(sms_service.get_history())
    }

@router.delete("/alerts/sms-history", status_code=status.HTTP_200_OK)
async def clear_sms_dispatch_history():
    """Clears the SMS dispatch history log."""
    sms_service.clear_history()
    return {
        "status": "cleared",
        "message": "SMS audit history cleared successfully.",
        "history": []
    }

@router.get("/alerts/gateway-status", status_code=status.HTTP_200_OK)
async def get_gateway_status():
    """Returns SMS gateway configuration readiness without leaking secrets."""
    validation = sms_service.validate_gateway()
    creds = sms_service.get_credentials()
    has_twilio = bool(creds["twilio_account_sid"] and creds["twilio_auth_token"])
    has_fast2sms = bool(creds["fast2sms_api_key"])
    
    return {
        "status": "ready" if validation["configured"] else "unconfigured",
        "delivery_status": "READY" if validation["configured"] else "GATEWAY_NOT_CONFIGURED",
        "provider": validation["provider"],
        "twilio_configured": has_twilio,
        "fast2sms_configured": has_fast2sms,
        "registered_officer": auth_service.get_current_officer(),
        "instructions": (
            "To enable live cellular SMS dispatch, configure Twilio or Fast2SMS credentials in backend/.env."
        ) if not validation["configured"] else f"SMS Gateway ({validation['provider']}) is active.",
        "config_instructions": sms_service.get_configuration_instructions() if not validation["configured"] else None
    }

class ConfigureGatewayRequest(BaseModel):
    provider: str = Field("fast2sms", description="Provider: fast2sms or twilio")
    fast2sms_api_key: Optional[str] = Field(None, description="Fast2SMS API authorization key")
    fast2sms_route: Optional[str] = Field("q", description="Fast2SMS route: q or v3")
    twilio_account_sid: Optional[str] = Field(None, description="Twilio Account SID")
    twilio_auth_token: Optional[str] = Field(None, description="Twilio Auth Token")
    twilio_from_number: Optional[str] = Field(None, description="Twilio From Number")

@router.post("/alerts/configure-gateway", status_code=status.HTTP_200_OK)
async def configure_gateway(payload: ConfigureGatewayRequest):
    """
    Saves SMS gateway credentials to backend/.env and re-validates the gateway.
    Allows municipal officers to activate Fast2SMS directly from the portal.
    """
    updates = {}
    provider_name = payload.provider.lower().strip()
    if provider_name == "fast2sms":
        if not payload.fast2sms_api_key or not payload.fast2sms_api_key.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="FAST2SMS_API_KEY cannot be empty.")
        updates["FAST2SMS_API_KEY"] = payload.fast2sms_api_key.strip()
        updates["FAST2SMS_ROUTE"] = (payload.fast2sms_route or "q").strip()
    elif provider_name == "twilio":
        if not payload.twilio_account_sid or not payload.twilio_auth_token or not payload.twilio_from_number:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Twilio SID, Auth Token, and From Number are required.")
        updates["TWILIO_ACCOUNT_SID"] = payload.twilio_account_sid.strip()
        updates["TWILIO_AUTH_TOKEN"] = payload.twilio_auth_token.strip()
        updates["TWILIO_FROM_NUMBER"] = payload.twilio_from_number.strip()
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Unsupported provider '{payload.provider}'.")

    sms_service.update_env_file(updates)
    validation = sms_service.validate_gateway()
    return {
        "status": "configured" if validation["configured"] else "unconfigured",
        "provider": validation["provider"],
        "delivery_status": validation["delivery_status"],
        "message": f"Gateway updated to {validation['provider']}.",
        "validation": validation,
    }
