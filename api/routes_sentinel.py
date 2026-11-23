import os
from django.http import StreamingHttpResponse
from ninja import Router
from ninja.errors import HttpError
from api.security import farmer_jwt_auth
from app.core.database import minio_client
from dotenv import load_dotenv, find_dotenv

router = Router(tags=["Sentinel Satellite Data"])


@router.get("/status")
def get_sentinel_status(request):
    from app.agents.indices_agent import _get_cdse_token

    load_dotenv(find_dotenv(), override=True)

    client_id = (os.getenv("CDSE_CLIENT_ID") or "").strip().strip('"').strip("'")
    client_secret = (os.getenv("CDSE_CLIENT_SECRET") or "").strip().strip('"').strip("'")
    has_creds = bool(client_id and client_secret)

    if not has_creds:
        return {
            "status": "missing_credentials",
            "has_credentials": False,
            "message": (
                "Copernicus CDSE credentials not configured in backend/.env "
                "(CDSE_CLIENT_ID / CDSE_CLIENT_SECRET)."
            ),
        }

    token = _get_cdse_token()
    if token:
        return {
            "status": "connected",
            "has_credentials": True,
            "message": "Successfully authenticated with Copernicus Data Space Ecosystem API.",
        }
    else:
        return {
            "status": "auth_failed",
            "has_credentials": True,
            "message": (
                "Authentication failed with provided Copernicus CDSE credentials. "
                "Check credentials in .env."
            ),
        }


@router.get("/tiff/{phone}/{filename}", auth=farmer_jwt_auth)
def get_tiff_image(request, phone: str, filename: str):
    current_user = request.auth
    if current_user.get("sub") != phone:
        raise HttpError(403, "Unauthorized: You can only view your own farm data.")

    try:
        object_path = f"{phone}/{filename}"
        response = minio_client.get_object("tiff-cache", object_path)
        return StreamingHttpResponse(
            response.stream(32 * 1024), content_type="image/tiff"
        )
    except Exception as e:
        raise HttpError(404, f"Image not found or MinIO error: {str(e)}")

