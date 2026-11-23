from typing import List
from ninja import Router
from ninja.errors import HttpError

from app.schemas.farmer import FarmerCreate, FarmerResponse
from app.core.database import database
from app.agents.geometry_agent import get_geometry
from app.agents.orchestrator import run_orchestrator

router = Router(tags=["Farmer Data"])


@router.post("/", response=FarmerResponse)
async def create_farmer(request, farmer: FarmerCreate):
    phone_clean = farmer.phone.strip()

    existing = await database["farmers"].find_one({"phone": phone_clean})
    if existing:
        raise HttpError(400, "Farmer with this phone number already exists.")

    farmer_dict = farmer.model_dump()
    farmer_dict["phone"] = phone_clean

    if not farmer_dict.get("sr_no"):
        highest = await database["farmers"].find_one(sort=[("sr_no", -1)])
        highest_sr = (
            highest.get("sr_no", 0)
            if highest and isinstance(highest.get("sr_no"), int)
            else 0
        )
        farmer_dict["sr_no"] = highest_sr + 1

    farmer_dict.setdefault("cache", {"stats": {}})

    new_farmer = await database["farmers"].insert_one(farmer_dict)
    if new_farmer.inserted_id:
        farmer_dict["id"] = str(new_farmer.inserted_id)
        return farmer_dict

    raise HttpError(500, "Failed to create farmer profile")


@router.get("/test")
async def test_farmer_route(request):
    return {"message": "Farmer router is working!"}


@router.get("/all")
async def get_all_farmers(request):
    cursor = database["farmers"].find({}, {"_id": 0})
    farmers = await cursor.to_list(length=None)
    return {"status": "success", "count": len(farmers), "data": farmers}


@router.get("/{phone}/geometry")
async def test_farmer_geometry(request, phone: str):
    farmer = await database["farmers"].find_one({"phone": phone}, {"_id": 0})
    if not farmer:
        raise HttpError(404, "Farmer not found")

    try:
        geometry_data = await get_geometry(farmer)
        return {
            "status": "success",
            "message": "Geometry parsed successfully!",
            "data": geometry_data,
        }
    except Exception as e:
        raise HttpError(500, f"Geometry parsing failed: {str(e)}")


@router.get("/{phone}/dashboard")
async def get_farmer_dashboard(request, phone: str):
    try:
        dashboard_data = await run_orchestrator(phone)
        return {"status": "success", "data": dashboard_data}
    except HttpError:
        raise
    except Exception as e:
        # Check if wrapped exception is an HTTPException (FastAPI/Starlette or Ninja)
        status_code = getattr(e, "status_code", None)
        detail = getattr(e, "detail", str(e))
        if status_code:
            raise HttpError(status_code, detail)
        raise HttpError(500, f"Dashboard generation failed: {str(e)}")


@router.get("/{phone}")
async def get_farmer_by_phone(request, phone: str):
    farmer = await database["farmers"].find_one({"phone": phone}, {"_id": 0})
    if not farmer:
        raise HttpError(404, "Farmer not found")
    return {"status": "success", "data": farmer}

