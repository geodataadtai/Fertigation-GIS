from ninja import NinjaAPI

from api.routes_farmer import router as farmer_router
from api.routes_auth import router as auth_router
from api.routes_sentinel import router as sentinel_router
from api.routes_admin import router as admin_router
from api.routes_advisory import router as advisory_router
from api.routes_biochemical import router as biochemical_router

api_v1 = NinjaAPI(
    title="FertiNex API",
    version="1.0.0",
    description="FertiNex Precision Fertigation, Satellite Telemetry & Biochemical Decision Support Platform",
    urls_namespace="api_v1",
)

api_v1.add_router("/farmer", farmer_router)
api_v1.add_router("/auth", auth_router)
api_v1.add_router("/sentinel", sentinel_router)
api_v1.add_router("/admin", admin_router)
api_v1.add_router("/advisory", advisory_router)
api_v1.add_router("/biochemical", biochemical_router)
api_v1.add_router("/agrogeo/biochemical", biochemical_router, url_name_prefix="agrogeo_biochemical")
