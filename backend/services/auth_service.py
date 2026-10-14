import re
import secrets
from datetime import datetime, timezone
from typing import Optional, Dict, Any

class AuthService:
    def __init__(self):
        # Default registered municipal officer for Coimbatore Municipal Corporation
        self.current_officer: Dict[str, Any] = {
            "name": "Kishore",
            "role": "Chief Municipal Health Officer",
            "jurisdiction": "Coimbatore Municipal Corporation (District 3331)",
            "email": "24121027@drngpit.ac.in",
            "contact_number": "+919345791866",
            "department": "Public Health & Epidemic Surveillance Division",
            "token": "officer-token-f0d0d9073de6b0fa7932ffe9b4f9c7b2",
            "is_authenticated": True,
            "logged_in_at": datetime.now(timezone.utc).isoformat(),
        }

    def validate_email(self, email: str) -> bool:
        if not email or "@" not in email:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))

    def validate_phone(self, phone: str) -> bool:
        if not phone:
            return False
        clean = re.sub(r"[\s\-\(\)]", "", str(phone))
        return bool(re.match(r"^(\+91|0)?[6-9]\d{9}$", clean) or re.match(r"^\+[1-9]\d{7,14}$", clean))

    def authenticate(self, email: str, contact_number: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate or register the municipal officer.
        Validates both email and contact number.
        """
        if not self.validate_email(email):
            raise ValueError(f"Invalid email format: '{email}'. A valid public health or municipal email address is required.")

        if not self.validate_phone(contact_number):
            raise ValueError(f"Invalid contact number: '{contact_number}'. Must be a valid 10-digit mobile number or E.164 international format.")

        # Format contact number to standard +91 if needed
        clean_phone = re.sub(r"[\s\-\(\)]", "", contact_number)
        if not clean_phone.startswith("+"):
            if clean_phone.startswith("0"):
                clean_phone = "+91" + clean_phone[1:]
            elif len(clean_phone) == 10:
                clean_phone = "+91" + clean_phone

        token = f"officer-token-{secrets.token_hex(16)}"
        officer_name = name or (
            "Dr. K. Sundaram" if "sundaram" in email.lower() or "officer" in email.lower()
            else email.split("@")[0].replace(".", " ").title() + " (Municipal Officer)"
        )

        self.current_officer = {
            "name": officer_name,
            "role": "Chief Municipal Health Officer",
            "jurisdiction": "Coimbatore Municipal Corporation (District 3331)",
            "email": email.strip().lower(),
            "contact_number": clean_phone,
            "department": "Public Health & Epidemic Surveillance Division",
            "token": token,
            "is_authenticated": True,
            "logged_in_at": datetime.now(timezone.utc).isoformat(),
        }
        return self.current_officer

    def get_current_officer(self) -> Dict[str, Any]:
        return self.current_officer

    def logout(self) -> None:
        self.current_officer["is_authenticated"] = False
        self.current_officer["token"] = None

auth_service = AuthService()
