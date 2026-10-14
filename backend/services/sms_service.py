import os
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import httpx
from backend.config import settings

logger = logging.getLogger("epidemiology.sms")

# Resolve backend directory and environment file paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
BACKEND_ENV_PATH = BACKEND_DIR / ".env"
ROOT_ENV_PATH = BACKEND_DIR.parent / ".env"


class AlertDeduplicator:
    """
    Prevents sending duplicate SMS alerts for the same ward, risk level, and date
    within the configured alert cooldown window.
    """
    def __init__(self, cooldown_hours: int = 12):
        self.cooldown_hours = cooldown_hours
        # Key: "ward_name:risk_level:date" -> datetime of dispatch
        self.history: Dict[str, datetime] = {}

    def get_key(self, ward_name: str, risk_level: str, alert_date: str) -> str:
        return f"{ward_name.strip().lower()}:{risk_level.strip().lower()}:{alert_date.strip().lower()}"

    def is_duplicate(self, ward_name: str, risk_level: str, alert_date: str) -> Tuple[bool, Optional[str]]:
        key = self.get_key(ward_name, risk_level, alert_date)
        last_sent = self.history.get(key)
        if not last_sent:
            return False, None
        elapsed_hours = (datetime.now(timezone.utc) - last_sent).total_seconds() / 3600.0
        if elapsed_hours < self.cooldown_hours:
            return True, f"Alert for {ward_name} ({risk_level}) was already dispatched at {last_sent.strftime('%H:%M UTC')}. Cooldown is {self.cooldown_hours}h."
        return False, None

    def record_dispatch(self, ward_name: str, risk_level: str, alert_date: str):
        key = self.get_key(ward_name, risk_level, alert_date)
        self.history[key] = datetime.now(timezone.utc)

    def clear(self):
        self.history.clear()


class SmsAlertService:
    def __init__(self):
        self.dispatch_history: List[Dict[str, Any]] = []
        self.deduplicator = AlertDeduplicator(
            cooldown_hours=getattr(settings, "ALERT_COOLDOWN_HOURS", 12)
        )

    def get_credentials(self) -> Dict[str, str]:
        """
        Reads gateway credentials securely from backend/.env first, then root .env, then os.environ.
        Filters out placeholder template strings and strips quotes/whitespace.
        Never exposes raw credentials in logs or public outputs.
        """
        backend_vars: Dict[str, str] = {}
        if BACKEND_ENV_PATH.exists():
            try:
                from dotenv import dotenv_values
                loaded = dotenv_values(BACKEND_ENV_PATH)
                if loaded:
                    for k, v in loaded.items():
                        if v is not None:
                            backend_vars[k.strip()] = str(v).strip().strip("'\"").strip()
            except Exception as e:
                logger.debug(f"Could not parse backend/.env: {e}")

        root_vars: Dict[str, str] = {}
        if ROOT_ENV_PATH.exists():
            try:
                from dotenv import dotenv_values
                loaded = dotenv_values(ROOT_ENV_PATH)
                if loaded:
                    for k, v in loaded.items():
                        if v is not None:
                            root_vars[k.strip()] = str(v).strip().strip("'\"").strip()
            except Exception as e:
                logger.debug(f"Could not parse root .env: {e}")

        def _resolve(key: str, default: str = "") -> str:
            # Priority: backend/.env > root .env > os.environ > settings
            val = (
                backend_vars.get(key)
                or root_vars.get(key)
                or os.getenv(key)
                or getattr(settings, key, default)
                or ""
            )
            val = str(val).strip().strip("'\"").strip()
            # Guard against common placeholder templates
            if val.lower() in (
                "your_twilio_account_sid_here",
                "your_twilio_auth_token_here",
                "your_fast2sms_api_key_here",
                "+1234567890",
                "none",
                "",
            ):
                return ""
            return val

        return {
            "twilio_account_sid": _resolve("TWILIO_ACCOUNT_SID"),
            "twilio_auth_token": _resolve("TWILIO_AUTH_TOKEN"),
            "twilio_from_number": _resolve("TWILIO_FROM_NUMBER"),
            "fast2sms_api_key": _resolve("FAST2SMS_API_KEY"),
            "fast2sms_route": _resolve("FAST2SMS_ROUTE", "q") or "q",
            "sms_provider": _resolve("SMS_PROVIDER", "").lower(),
        }

    def validate_gateway(self) -> Dict[str, Any]:
        """
        Validates whether a real SMS gateway is configured and ready.
        Returns gateway name and readiness status: SENT, FAILED, or GATEWAY_NOT_CONFIGURED.
        Prioritizes Fast2SMS when FAST2SMS_API_KEY is configured.
        """
        creds = self.get_credentials()
        preferred_provider = creds.get("sms_provider", "").lower()
        has_fast2sms = bool(creds.get("fast2sms_api_key"))
        has_twilio = bool(
            creds.get("twilio_account_sid") and creds.get("twilio_auth_token") and creds.get("twilio_from_number")
        )

        if preferred_provider == "twilio" and has_twilio:
            return {
                "configured": True,
                "provider": "Twilio SMS",
                "status": "ready",
                "delivery_status": "READY",
                "detail": f"Twilio gateway configured with sender {creds['twilio_from_number']}",
                "instructions": None,
            }
        elif has_fast2sms:
            return {
                "configured": True,
                "provider": "Fast2SMS",
                "status": "ready",
                "delivery_status": "READY",
                "detail": f"Fast2SMS gateway configured for Indian mobile numbers (route: {creds['fast2sms_route']})",
                "instructions": None,
            }
        elif has_twilio:
            return {
                "configured": True,
                "provider": "Twilio SMS",
                "status": "ready",
                "delivery_status": "READY",
                "detail": f"Twilio gateway configured with sender {creds['twilio_from_number']}",
                "instructions": None,
            }
        else:
            return {
                "configured": False,
                "provider": "None",
                "status": "unconfigured",
                "delivery_status": "GATEWAY_NOT_CONFIGURED",
                "detail": "No gateway credentials in backend/.env. Provide TWILIO_* or FAST2SMS_API_KEY.",
                "instructions": self.get_configuration_instructions(),
            }

    def update_env_file(self, updates: Dict[str, str]):
        """Persists gateway updates to backend/.env without overwriting unrelated variables."""
        if not BACKEND_ENV_PATH.exists():
            BACKEND_ENV_PATH.touch()
        try:
            lines = BACKEND_ENV_PATH.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
        new_lines = []
        applied_keys = set()
        for line in lines:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                k, _ = stripped.split("=", 1)
                k = k.strip()
                if k in updates:
                    new_lines.append(f"{k}={updates[k]}")
                    applied_keys.add(k)
                    continue
            new_lines.append(line)
        for k, v in updates.items():
            if k not in applied_keys:
                new_lines.append(f"{k}={v}")
        BACKEND_ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    @staticmethod
    def get_configuration_instructions() -> Dict[str, Any]:
        """Returns clear, structured setup instructions when credentials are missing."""
        return {
            "config_file": "backend/.env",
            "template_file": "backend/.env.example",
            "message": "SMS Gateway is not configured. Add credentials for either Twilio or Fast2SMS in backend/.env to activate live cellular SMS dispatch.",
            "twilio_example": {
                "description": "Twilio REST API (Cellular SMS, International & Indian numbers)",
                "variables": {
                    "TWILIO_ACCOUNT_SID": "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                    "TWILIO_AUTH_TOKEN": "your_twilio_auth_token_here",
                    "TWILIO_FROM_NUMBER": "+1234567890"
                }
            },
            "fast2sms_example": {
                "description": "Fast2SMS Bulk V2 (Indian 10-digit mobile numbers)",
                "variables": {
                    "FAST2SMS_API_KEY": "your_fast2sms_api_key_here",
                    "FAST2SMS_ROUTE": "q"
                }
            }
        }

    def validate_phone(self, phone: str) -> bool:
        """Validate phone format (E.164 or 10-digit Indian standard)."""
        if not phone:
            return False
        clean = re.sub(r"[\s\-\(\)]", "", str(phone))
        return bool(re.match(r"^(\+91|0)?[6-9]\d{9}$", clean) or re.match(r"^\+[1-9]\d{7,14}$", clean))

    def format_alert_message(
        self,
        ward_name: str,
        risk_level: str,
        prediction_date: str,
        preventive_action: str,
        disease: str = "Dengue"
    ) -> str:
        """
        Builds the standardized municipal public health alert message.
        Mandatory components (Requirement 6):
        - Ward name
        - Risk level
        - Prediction date
        - Recommended preventive action
        """
        clean_action = preventive_action.strip()
        if not clean_action.endswith("."):
            clean_action += "."

        return (
            f"EPIDEMIOLOGY ALERT: {ward_name} classified {risk_level.upper()} for {disease}. "
            f"Date: {prediction_date}. "
            f"Action: {clean_action} "
            f"Ref: CMC/PHD"
        )

    async def dispatch_high_risk_alert(
        self,
        officer_phone: str,
        officer_name: str,
        high_risk_wards: List[Dict[str, Any]],
        disease: str = "Dengue",
        custom_message: Optional[str] = None,
        prediction_date: Optional[str] = None,
        force: bool = False,
        is_automated: bool = False,
        is_test: bool = False
    ) -> Dict[str, Any]:
        """
        Sends an emergency SMS notification to the registered municipal officer.
        Strictly guarantees:
        - Never claims SENT/SUBMITTED unless the gateway explicitly confirms submission.
        - Distinguishes submission to carrier from final delivery to handset.
        - Delivers clear delivery statuses: SENT, FAILED, or GATEWAY_NOT_CONFIGURED.
        - Includes ward name, risk level, prediction date, and recommended action.
        - Prevents duplicate alerts unless force=True.
        - Prevents repeated attempts/log spam when gateway is not configured.
        - Records actual provider responses without leaking secret tokens.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        current_date = prediction_date or datetime.now().strftime("%Y-%m-%d")

        # Compile ward names and actions
        ward_names = [w.get("name") or w.get("ward_name", "Unknown Ward") for w in high_risk_wards]
        ward_str = ", ".join(ward_names) if ward_names else "Ward 01, Ward 02"

        # Action from wards or standard public health protocol
        default_action = (
            high_risk_wards[0].get("recommendation")
            if high_risk_wards and high_risk_wards[0].get("recommendation")
            else "Deploy targeted vector control, larvicide spraying, and fever surveillance."
        )

        # Build message conforming to Requirement 6
        if custom_message:
            body = custom_message
        else:
            body = self.format_alert_message(
                ward_name=ward_str,
                risk_level="HIGH RISK",
                prediction_date=current_date,
                preventive_action=default_action,
                disease=disease
            )

        # 1. Recipient Phone Validation
        if not officer_phone or not self.validate_phone(officer_phone):
            result = {
                "id": f"sms-{len(self.dispatch_history) + 1}",
                "timestamp": timestamp,
                "status": "failed",
                "delivery_status": "FAILED",
                "carrier_stage": "VALIDATION_ERROR",
                "delivery_confirmed": False,
                "recipient_name": officer_name or "Municipal Officer",
                "recipient_phone": officer_phone or "None",
                "wards": ward_names,
                "disease": disease,
                "prediction_date": current_date,
                "message": body,
                "gateway": "None",
                "provider_id": None,
                "provider_response": None,
                "error": f"Invalid recipient contact number '{officer_phone}'. Valid 10-digit mobile or E.164 required.",
                "is_duplicate": False,
                "is_test": is_test
            }
            self.dispatch_history.insert(0, result)
            return result

        # 2. Duplicate Detection (Requirement 7)
        if not force and not is_test:
            for w in ward_names:
                is_dup, reason = self.deduplicator.is_duplicate(w, "High", current_date)
                if is_dup:
                    logger.info(f"Duplicate alert skipped for {w}: {reason}")
                    result = {
                        "id": f"sms-{len(self.dispatch_history) + 1}",
                        "timestamp": timestamp,
                        "status": "failed",
                        "delivery_status": "FAILED",
                        "carrier_stage": "DUPLICATE_SUPPRESSED",
                        "delivery_confirmed": False,
                        "recipient_name": officer_name,
                        "recipient_phone": officer_phone,
                        "wards": ward_names,
                        "disease": disease,
                        "prediction_date": current_date,
                        "message": body,
                        "gateway": "Deduplicator",
                        "provider_id": None,
                        "provider_response": None,
                        "error": f"Duplicate alert prevented: {reason}. Use force dispatch to override.",
                        "is_duplicate": True,
                        "is_test": is_test
                    }
                    self.dispatch_history.insert(0, result)
                    return result

        # 3. Check Gateway Configuration (Requirement 4 & 7)
        gateway_check = self.validate_gateway()
        if not gateway_check["configured"]:
            result = {
                "id": f"sms-{len(self.dispatch_history) + 1}",
                "timestamp": timestamp,
                "status": "unconfigured",
                "delivery_status": "GATEWAY_NOT_CONFIGURED",
                "carrier_stage": "GATEWAY_NOT_CONFIGURED",
                "delivery_confirmed": False,
                "recipient_name": officer_name or "Municipal Health Officer",
                "recipient_phone": officer_phone,
                "wards": ward_names,
                "disease": disease,
                "prediction_date": current_date,
                "message": body,
                "gateway": "None (Unconfigured)",
                "provider_id": None,
                "provider_response": None,
                "error": (
                    "SMS Gateway not configured in environment. Provide TWILIO_ACCOUNT_SID, "
                    "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER (or FAST2SMS_API_KEY) in backend/.env "
                    f"to transmit real cellular SMS to {officer_phone}."
                ),
                "instructions": gateway_check["instructions"],
                "is_duplicate": False,
                "is_test": is_test
            }
            # Prevent repeated attempts from bloating history:
            if not is_automated:
                if self.dispatch_history and self.dispatch_history[0].get("delivery_status") == "GATEWAY_NOT_CONFIGURED":
                    self.dispatch_history[0] = result
                else:
                    self.dispatch_history.insert(0, result)
            return result

        # 4. Live Gateway Dispatch
        creds = self.get_credentials()
        provider = gateway_check["provider"]

        # Option A: Twilio Dispatch
        if provider == "Twilio SMS":
            twilio_sid = creds["twilio_account_sid"]
            twilio_token = creds["twilio_auth_token"]
            twilio_from = creds["twilio_from_number"]
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            formatted_to = officer_phone if officer_phone.startswith("+") else f"+91{officer_phone.lstrip('0')}"

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        url,
                        data={
                            "To": formatted_to,
                            "From": twilio_from,
                            "Body": body,
                        },
                        auth=(twilio_sid, twilio_token),
                    )
                    data = resp.json()

                    # Only claim submission if Twilio returns HTTP 200/201 and has an SID
                    if resp.status_code in (200, 201) and data.get("sid"):
                        result = {
                            "id": f"sms-{len(self.dispatch_history) + 1}",
                            "timestamp": timestamp,
                            "status": "submitted",
                            "delivery_status": "SENT",
                            "carrier_stage": "SUBMITTED_TO_CARRIER",
                            "delivery_confirmed": False,  # Carrier submission confirmed; handset DLR pending
                            "status_note": f"Submitted to Twilio carrier queue (SID: {data.get('sid')}). Status: {data.get('status', 'queued')}.",
                            "recipient_name": officer_name,
                            "recipient_phone": formatted_to,
                            "wards": ward_names,
                            "disease": disease,
                            "prediction_date": current_date,
                            "message": body,
                            "gateway": "Twilio SMS",
                            "provider_id": data.get("sid"),
                            "provider_response": {
                                "sid": data.get("sid"),
                                "status": data.get("status"),
                                "direction": data.get("direction"),
                                "to": data.get("to"),
                                "from": data.get("from"),
                                "date_created": data.get("date_created"),
                            },
                            "error": None,
                            "is_duplicate": False,
                            "is_test": is_test
                        }
                        for w in ward_names:
                            self.deduplicator.record_dispatch(w, "High", current_date)
                    else:
                        err_code = data.get("code") or resp.status_code
                        err_msg = data.get("message") or f"Twilio returned status {resp.status_code}"
                        result = {
                            "id": f"sms-{len(self.dispatch_history) + 1}",
                            "timestamp": timestamp,
                            "status": "failed",
                            "delivery_status": "FAILED",
                            "carrier_stage": "PROVIDER_REJECTED",
                            "delivery_confirmed": False,
                            "recipient_name": officer_name,
                            "recipient_phone": formatted_to,
                            "wards": ward_names,
                            "disease": disease,
                            "prediction_date": current_date,
                            "message": body,
                            "gateway": "Twilio SMS",
                            "provider_id": None,
                            "provider_response": {
                                "http_status": resp.status_code,
                                "error_code": err_code,
                                "error_message": err_msg,
                            },
                            "error": f"Twilio Error ({err_code}): {err_msg}",
                            "is_duplicate": False,
                            "is_test": is_test
                        }
            except Exception as e:
                logger.error(f"Twilio network dispatch failure: {e}", exc_info=True)
                result = {
                    "id": f"sms-{len(self.dispatch_history) + 1}",
                    "timestamp": timestamp,
                    "status": "failed",
                    "delivery_status": "FAILED",
                    "carrier_stage": "NETWORK_ERROR",
                    "delivery_confirmed": False,
                    "recipient_name": officer_name,
                    "recipient_phone": officer_phone,
                    "wards": ward_names,
                    "disease": disease,
                    "prediction_date": current_date,
                    "message": body,
                    "gateway": "Twilio SMS",
                    "provider_id": None,
                    "provider_response": None,
                    "error": f"Network failure communicating with Twilio: {str(e)}",
                    "is_duplicate": False,
                    "is_test": is_test
                }

        # Option B: Fast2SMS Dispatch (Official https://www.fast2sms.com/dev/bulkV2)
        elif provider == "Fast2SMS":
            fast2sms_key = creds["fast2sms_api_key"]
            fast2sms_route = creds.get("fast2sms_route", "q")

            # Clean phone to standard 10-digit Indian cellular mobile number
            clean_number = re.sub(r"[^\d]", "", str(officer_phone))
            if clean_number.startswith("91") and len(clean_number) == 12:
                clean_number = clean_number[2:]
            elif clean_number.startswith("0") and len(clean_number) == 11:
                clean_number = clean_number[1:]

            if len(clean_number) != 10:
                result = {
                    "id": f"sms-{len(self.dispatch_history) + 1}",
                    "timestamp": timestamp,
                    "status": "failed",
                    "delivery_status": "FAILED",
                    "carrier_stage": "PROVIDER_REJECTED",
                    "delivery_confirmed": False,
                    "recipient_name": officer_name,
                    "recipient_phone": officer_phone,
                    "wards": ward_names,
                    "disease": disease,
                    "prediction_date": current_date,
                    "message": body,
                    "gateway": "Fast2SMS",
                    "provider_id": None,
                    "provider_response": {
                        "return": False,
                        "status_code": 400,
                        "message": f"Invalid mobile number format for Fast2SMS. Expected 10 digits starting with 6-9, got '{officer_phone}'."
                    },
                    "error": f"Fast2SMS Error (400): Expected 10-digit Indian cellular number.",
                    "is_duplicate": False,
                    "is_test": is_test
                }
                self.dispatch_history.insert(0, result)
                return result

            url = "https://www.fast2sms.com/dev/bulkV2"
            headers = {
                "authorization": fast2sms_key,
                "Content-Type": "application/json"
            }
            payload = {
                "route": fast2sms_route,
                "message": body,
                "language": "english",
                "flash": 0,
                "numbers": clean_number,
            }
            if fast2sms_route == "v3":
                payload["sender_id"] = "TXTIND"

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"status_code": resp.status_code, "message": resp.text[:200]}

                    # Extract message text from Fast2SMS response (string or list of strings)
                    raw_msg = data.get("message")
                    if isinstance(raw_msg, list):
                        msg_str = "; ".join(str(m) for m in raw_msg)
                    elif raw_msg is not None:
                        msg_str = str(raw_msg)
                    else:
                        msg_str = ""

                    status_code = data.get("status_code") or resp.status_code
                    is_return_true = (data.get("return") is True)

                    # Only claim submission if Fast2SMS explicitly confirms return == true and HTTP 200
                    if resp.status_code == 200 and is_return_true:
                        req_id = str(data.get("request_id") or "FAST2SMS_ACCEPTED")
                        result = {
                            "id": f"sms-{len(self.dispatch_history) + 1}",
                            "timestamp": timestamp,
                            "status": "submitted",
                            "delivery_status": "SENT",
                            "carrier_stage": "SUBMITTED_TO_CARRIER",
                            "delivery_confirmed": False,  # Carrier submission accepted; final handset delivery pending
                            "status_note": f"Submitted to Fast2SMS carrier route (Request ID: {req_id}). Cellular transmission in progress to recipient handset.",
                            "recipient_name": officer_name,
                            "recipient_phone": officer_phone,
                            "wards": ward_names,
                            "disease": disease,
                            "prediction_date": current_date,
                            "message": body,
                            "gateway": "Fast2SMS",
                            "provider_id": req_id,
                            "provider_response": {
                                "return": True,
                                "request_id": req_id,
                                "message": msg_str or "SMS sent successfully.",
                                "status_code": 200,
                            },
                            "error": None,
                            "is_duplicate": False,
                            "is_test": is_test
                        }
                        for w in ward_names:
                            self.deduplicator.record_dispatch(w, "High", current_date)
                    else:
                        # Categorize specific Fast2SMS error conditions
                        msg_lower = msg_str.lower()
                        if (
                            status_code == 401
                            or resp.status_code == 401
                            or "authorization" in msg_lower
                            or "invalid authorization key" in msg_lower
                            or "api key" in msg_lower
                        ):
                            stage = "INVALID_API_KEY"
                            err_desc = f"Fast2SMS Error (401): Invalid Authorization Key. Please verify FAST2SMS_API_KEY in backend/.env."
                        elif (
                            status_code in (402, 412)
                            or resp.status_code in (402, 412)
                            or "insufficient balance" in msg_lower
                            or "balance" in msg_lower
                            or "wallet" in msg_lower
                        ):
                            stage = "INSUFFICIENT_BALANCE"
                            err_desc = f"Fast2SMS Error ({status_code}): Insufficient Balance. Please recharge your Fast2SMS wallet to transmit cellular SMS."
                        else:
                            stage = "PROVIDER_REJECTED"
                            err_desc = f"Fast2SMS Error ({status_code}): {msg_str or f'Request rejected by Fast2SMS (HTTP {resp.status_code})'}"

                        result = {
                            "id": f"sms-{len(self.dispatch_history) + 1}",
                            "timestamp": timestamp,
                            "status": "failed",
                            "delivery_status": "FAILED",
                            "carrier_stage": stage,
                            "delivery_confirmed": False,
                            "recipient_name": officer_name,
                            "recipient_phone": officer_phone,
                            "wards": ward_names,
                            "disease": disease,
                            "prediction_date": current_date,
                            "message": body,
                            "gateway": "Fast2SMS",
                            "provider_id": str(data.get("request_id")) if data.get("request_id") else None,
                            "provider_response": {
                                "return": False,
                                "status_code": status_code,
                                "message": msg_str,
                                "request_id": data.get("request_id"),
                            },
                            "error": err_desc,
                            "is_duplicate": False,
                            "is_test": is_test
                        }
            except httpx.TimeoutException as e:
                logger.error(f"Fast2SMS timeout failure: {e}")
                result = {
                    "id": f"sms-{len(self.dispatch_history) + 1}",
                    "timestamp": timestamp,
                    "status": "failed",
                    "delivery_status": "FAILED",
                    "carrier_stage": "NETWORK_TIMEOUT",
                    "delivery_confirmed": False,
                    "recipient_name": officer_name,
                    "recipient_phone": officer_phone,
                    "wards": ward_names,
                    "disease": disease,
                    "prediction_date": current_date,
                    "message": body,
                    "gateway": "Fast2SMS",
                    "provider_id": None,
                    "provider_response": None,
                    "error": "Network timeout (15s) while contacting Fast2SMS API at https://www.fast2sms.com/dev/bulkV2.",
                    "is_duplicate": False,
                    "is_test": is_test
                }
            except Exception as e:
                logger.error(f"Fast2SMS network dispatch failure: {e}", exc_info=True)
                result = {
                    "id": f"sms-{len(self.dispatch_history) + 1}",
                    "timestamp": timestamp,
                    "status": "failed",
                    "delivery_status": "FAILED",
                    "carrier_stage": "NETWORK_ERROR",
                    "delivery_confirmed": False,
                    "recipient_name": officer_name,
                    "recipient_phone": officer_phone,
                    "wards": ward_names,
                    "disease": disease,
                    "prediction_date": current_date,
                    "message": body,
                    "gateway": "Fast2SMS",
                    "provider_id": None,
                    "provider_response": None,
                    "error": f"Network connection failure communicating with Fast2SMS: {str(e)}",
                    "is_duplicate": False,
                    "is_test": is_test
                }
        else:
            result = {
                "id": f"sms-{len(self.dispatch_history) + 1}",
                "timestamp": timestamp,
                "status": "failed",
                "delivery_status": "GATEWAY_NOT_CONFIGURED",
                "carrier_stage": "GATEWAY_NOT_CONFIGURED",
                "delivery_confirmed": False,
                "recipient_name": officer_name,
                "recipient_phone": officer_phone,
                "wards": ward_names,
                "disease": disease,
                "prediction_date": current_date,
                "message": body,
                "gateway": "None",
                "provider_id": None,
                "provider_response": None,
                "error": "No valid gateway configured.",
                "is_duplicate": False,
                "is_test": is_test
            }

        self.dispatch_history.insert(0, result)
        if len(self.dispatch_history) > 100:
            self.dispatch_history.pop()

        return result

    async def dispatch_test_sms(
        self,
        officer_phone: str,
        officer_name: str,
        note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sends a single test SMS to the authorized recipient after gateway configuration.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        body = (
            f"EPIDEMIOLOGY SYSTEM TEST: Gateway verification for {officer_name}. "
            f"Date: {date_str}. "
            f"Action: Alert pipeline active. Coimbatore Municipal Corporation. Ref: CMC/PHD"
        )
        return await self.dispatch_high_risk_alert(
            officer_phone=officer_phone,
            officer_name=officer_name,
            high_risk_wards=[{"name": "System Test Ward", "cases": 0, "risk": "Test", "recommendation": "Verification test action"}],
            disease="System Test",
            custom_message=body,
            prediction_date=date_str,
            force=True,  # Test SMS overrides deduplication
            is_automated=False,
            is_test=True
        )

    async def check_and_alert_thresholds(
        self,
        disease: str = "Dengue",
        officer_phone: Optional[str] = None,
        officer_name: Optional[str] = None,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Scans all monitored municipal wards against the predefined high-risk threshold.
        Predefined threshold: cases >= 24 or composite risk score >= 65.0.
        Dispatches individual or combined alerts with ward name, risk level,
        prediction date, and recommended action.
        Suppresses attempts if gateway is unconfigured to prevent spamming the audit log.
        """
        if not self.validate_gateway()["configured"]:
            logger.debug("Automated threshold scan skipped: SMS Gateway not configured in backend/.env")
            return []

        from backend.services.risk_service import risk_service
        risk_evaluation = await risk_service.evaluate_wards_risk(disease=disease)
        high_risk_items = [w for w in risk_evaluation.wards if w.risk_level == "High"]

        if not high_risk_items:
            return []

        from backend.services.auth_service import auth_service
        officer = auth_service.get_current_officer()
        phone = officer_phone or officer.get("contact_number")
        name = officer_name or officer.get("name", "Chief Municipal Health Officer")

        results = []
        ward_dicts = [
            {
                "name": w.ward_name,
                "cases": w.cases,
                "risk": w.risk_level,
                "recommendation": w.recommendation,
                "score": w.composite_risk_score,
            }
            for w in high_risk_items
        ]

        res = await self.dispatch_high_risk_alert(
            officer_phone=phone,
            officer_name=name,
            high_risk_wards=ward_dicts,
            disease=disease,
            prediction_date=datetime.now().strftime("%Y-%m-%d"),
            force=force,
            is_automated=True
        )
        results.append(res)
        return results

    def get_history(self) -> List[Dict[str, Any]]:
        return self.dispatch_history

    def clear_history(self):
        """Clears the dispatch history log."""
        self.dispatch_history.clear()


sms_service = SmsAlertService()
