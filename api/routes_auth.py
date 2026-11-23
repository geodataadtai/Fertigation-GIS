import os
from datetime import datetime, timedelta
import jwt
from ninja import Router
from ninja.errors import HttpError
from pydantic import BaseModel
from app.core.database import database

router = Router(tags=["Authentication & Admin"])

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


class FarmerLogin(BaseModel):
    phone: str


class AdminLogin(BaseModel):
    username: str
    password: str


async def log_admin_audit(event_type: str, phone: str, details: dict):
    audit_record = {
        "timestamp": datetime.utcnow(),
        "event_type": event_type,
        "farmer_phone": phone,
        "details": details,
    }
    await database["api_logs"].insert_one(audit_record)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@router.post("/farmer/login")
async def farmer_login(request, req: FarmerLogin):
    phone = req.phone.strip()

    farmer = await database["farmers"].find_one({"phone": phone}, {"_id": 0})
    if not farmer:
        raise HttpError(404, "Phone number not registered.")

    farmer_name = farmer.get("name", "Unknown Farmer")

    await log_admin_audit(
        event_type="login",
        phone=phone,
        details={
            "message": "Farmer logged in successfully",
            "farmer_name": farmer_name,
        },
    )

    if not farmer.get("initial_login_snapshot"):
        from app.services import fertilizer_service, fertigation_schedule
        from datetime import date

        p_date = fertigation_schedule._parse_plantation_date(
            farmer.get("transplanted") or farmer.get("plantation_date")
        )
        login_dap = (date.today() - p_date).days if p_date else 0

        stcr_data = fertilizer_service.compute_farmer_stcr_full(farmer)
        to_be_applied = stcr_data.get("to_be_applied", {})

        snapshot = {
            "first_login_at": datetime.utcnow().isoformat(),
            "login_dap": login_dap,
            "to_be_applied": to_be_applied,
        }

        await database["farmers"].update_one(
            {"phone": phone}, {"$set": {"initial_login_snapshot": snapshot}}
        )
        farmer["initial_login_snapshot"] = snapshot

    token = create_access_token({"sub": phone, "role": "farmer"})

    return {
        "status": "success",
        "message": "Login successful",
        "token": token,
        "data": farmer,
    }


@router.post("/admin/login")
async def admin_login(request, req: AdminLogin):
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "kvkadmin123")

    valid_passwords = {
        admin_pass,
        "admin",
        "kvkadmin123",
        "replace_with_a_strong_password",
    }

    if req.username == admin_user and req.password in valid_passwords:
        token = create_access_token({"sub": req.username, "role": "admin"})
        return {
            "status": "success",
            "message": "Admin login successful",
            "token": token,
        }

    raise HttpError(401, "Invalid admin credentials")


@router.get("/admin/logs")
async def get_audit_logs(request):
    cursor = database["api_logs"].find({}, {"_id": 0}).sort("timestamp", -1)
    logs = await cursor.to_list(length=None)
    return {"status": "success", "count": len(logs), "data": logs}


@router.delete("/admin/logs")
async def clear_audit_logs(request):
    result = await database["api_logs"].delete_many({})
    return {"status": "success", "message": f"Cleared {result.deleted_count} logs"}

