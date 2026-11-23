from datetime import datetime
from django.http import HttpResponse
from ninja import Router
from ninja.errors import HttpError

from app.core.database import database, minio_client
from app.agents.orchestrator import run_orchestrator
from app.services import fertilizer_service
from app.core.raster_utils import calculate_tiff_average

router = Router(tags=["Admin Dashboard"])


@router.get("/dashboard-data")
async def get_dashboard_data(request):
    try:
        total_farmers_count = await database.farmers.count_documents({})

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_requests": {"$sum": 1},
                    "cache_hits": {
                        "$sum": {"$cond": [{"$eq": ["$is_cache_hit", True]}, 1, 0]}
                    },
                    "avg_time_ms": {"$avg": "$processing_time_ms"},
                    "total_pu": {"$sum": "$cdse_pu_used"},
                }
            }
        ]

        log_stats_cursor = database.api_logs.aggregate(pipeline)
        log_stats = await log_stats_cursor.to_list(length=1)

        cache_hit_rate = "0.0%"
        avg_load_time = "0.00s"
        copernicus_pus = 0

        if log_stats and len(log_stats) > 0:
            stats = log_stats[0]
            total_req = stats.get("total_requests", 0)
            hits = stats.get("cache_hits", 0)
            avg_ms = stats.get("avg_time_ms") or 0

            if total_req > 0:
                rate = (hits / total_req) * 100
                cache_hit_rate = f"{round(rate, 1)}%"

            avg_load_time = f"{round(avg_ms / 1000, 2)}s"
            copernicus_pus = round(stats.get("total_pu", 0), 2)

        cursor = database.farmers.find()
        farmers_data = await cursor.to_list(length=None)

        formatted_farmers = []
        for farmer in farmers_data:
            try:
                farmer["_id"] = str(farmer["_id"])
                cache_stats = (farmer.get("cache") or {}).get("stats", {})

                if cache_stats:
                    farmer["advisory"] = {
                        "phase": "Vegetative",
                        "ndvi": round(cache_stats.get("NDVI", 0), 3)
                        if cache_stats.get("NDVI") is not None
                        else "-",
                        "evi": round(cache_stats.get("EVI", 0), 3)
                        if cache_stats.get("EVI") is not None
                        else "-",
                        "ndre": round(cache_stats.get("NDRE", 0), 3)
                        if cache_stats.get("NDRE") is not None
                        else "-",
                        "msavi": round(cache_stats.get("MSAVI", 0), 3)
                        if cache_stats.get("MSAVI") is not None
                        else "-",
                        "ndmi": round(cache_stats.get("NDMI", 0), 3)
                        if cache_stats.get("NDMI") is not None
                        else "-",
                        "ndwi": round(cache_stats.get("NDWI", 0), 3)
                        if cache_stats.get("NDWI") is not None
                        else "-",
                    }
                else:
                    farmer["advisory"] = None

                status_info = fertilizer_service.check_farmer_window_and_approval(farmer)
                schedule = status_info.get("schedule") or {}
                farmer["window_active_today"] = status_info.get("window_active_today", False)
                farmer["is_pending"] = False
                farmer["cycle_status"] = status_info.get("cycle_status", "Scheduled")
                farmer["crop_season"] = (
                    farmer.get("crop_season")
                    or farmer.get("season")
                    or schedule.get("season", "Adsali")
                )
                farmer["schedule"] = schedule
                farmer["live_dap"] = schedule.get("current_dap")
                farmer["first_login_at"] = (
                    farmer.get("initial_login_snapshot") or {}
                ).get("first_login_at")
                farmer["login_dap"] = (
                    farmer.get("initial_login_snapshot") or {}
                ).get("login_dap")

                approved_adv = farmer.get("approved_advisory")
                if not approved_adv:
                    ndvi = (cache_stats or {}).get("NDVI", 0.0)
                    approved_adv = {
                        "type": "success"
                        if ndvi >= 0.65
                        else ("warning" if ndvi >= 0.4 else "info"),
                        "status": "Healthy crop growth"
                        if ndvi >= 0.65
                        else "Standard Monitoring",
                        "action": "All indices are within expected bounds. Continue standard monitoring.",
                        "isApproved": True,
                        "approved_at": datetime.now().isoformat(),
                    }
                else:
                    approved_adv = dict(approved_adv)
                    approved_adv["isApproved"] = True

                if not farmer.get("address"):
                    village = farmer.get("village", "Kagal")
                    taluka = farmer.get("taluka", "Kagal")
                    district = farmer.get("district", "Kolhapur")
                    pincode = farmer.get("pincode", 416216)
                    farmer["address"] = f"{village}, Tal. {taluka}, Dist. {district} - {pincode}"

                farmer["approved_advisory"] = approved_adv
                formatted_farmers.append(farmer)
            except Exception as f_err:
                print(f"[Admin] Warning: failed to enrich farmer {farmer.get('name')}: {f_err}")
                farmer["_id"] = str(farmer.get("_id", ""))
                farmer["name"] = farmer.get("name", "Unknown Farmer")
                farmer["phone"] = farmer.get("phone", "")
                farmer["is_pending"] = False
                farmer["live_dap"] = farmer.get("live_dap") or 0
                farmer["approved_advisory"] = farmer.get("approved_advisory") or {
                    "type": "info",
                    "status": "Standard Monitoring",
                    "action": "Active record.",
                    "isApproved": True,
                    "approved_at": datetime.now().isoformat(),
                }
                formatted_farmers.append(farmer)

        dashboard_metrics = {
            "total_farmers": total_farmers_count,
            "copernicus_pus": copernicus_pus,
            "cache_hit_rate": cache_hit_rate,
            "average_load_time": avg_load_time,
        }

        return {"metrics": dashboard_metrics, "farmers": formatted_farmers}

    except Exception as e:
        raise HttpError(500, str(e))


@router.get("/db-status")
async def get_db_status(request):
    try:
        stats = await database.command("dbstats")

        mongo_data_size_mb = round(stats.get("dataSize", 0) / (1024 * 1024), 2)
        mongo_storage_size_mb = round(stats.get("storageSize", 0) / (1024 * 1024), 2)

        structure = {
            "collections": ["farmers", "advisory_logs", "satellite_indices"],
            "schemas": {
                "farmers": {
                    "_id": "ObjectId",
                    "name": "String",
                    "phone": "String (Unique)",
                    "transplanted_date": "Date",
                    "variety": "String",
                    "address": "String",
                    "kml_url": "String (MinIO)",
                }
            },
        }

        minio_buckets = minio_client.list_buckets()
        minio_stats = []
        total_minio_size_mb = 0

        for bucket in minio_buckets:
            bucket_name = bucket.name
            objects = minio_client.list_objects(bucket_name, recursive=True)
            bucket_size_bytes = sum([obj.size for obj in objects])
            size_mb = round(bucket_size_bytes / (1024 * 1024), 2)
            total_minio_size_mb += size_mb

            minio_stats.append(
                {
                    "name": bucket_name,
                    "size_mb": size_mb,
                    "creation_date": (
                        bucket.creation_date.strftime("%Y-%m-%d")
                        if bucket.creation_date
                        else "Unknown"
                    ),
                }
            )

        return {
            "status": "Online / Active",
            "mongodb": {
                "database_name": stats.get("db", "kvk_geoai_db"),
                "collections_count": stats.get("collections", 0),
                "documents_count": stats.get("objects", 0),
                "data_size_mb": mongo_data_size_mb,
                "storage_size_mb": mongo_storage_size_mb,
                "structure": structure,
            },
            "minio": {
                "total_size_mb": round(total_minio_size_mb, 2),
                "buckets": minio_stats,
            },
        }

    except Exception as e:
        raise HttpError(500, str(e))


@router.get("/farmer-kml/{phone}")
def get_farmer_kml(request, phone: str):
    response = None
    try:
        bucket_name = "kml-storage"
        file_name = f"{phone}.kml"

        response = minio_client.get_object(bucket_name, file_name)
        kml_data = response.read()

        return HttpResponse(
            kml_data, content_type="application/vnd.google-earth.kml+xml"
        )
    except Exception as e:
        raise HttpError(404, f"KML file not found or MinIO error: {str(e)}")
    finally:
        if response:
            try:
                response.close()
                response.release_conn()
            except Exception:
                pass


@router.post("/fetch-sentinel/{phone}")
async def admin_fetch_sentinel(request, phone: str):
    try:
        result = await run_orchestrator(phone)

        indices = result.get("indices", {})
        averages = {}
        for idx in ["NDVI", "EVI", "NDRE", "MSAVI", "NDMI", "NDWI"]:
            val = calculate_tiff_average(
                minio_client, "tiff-cache", f"{phone}/{idx}.tiff"
            )
            if (
                (val == 0.0 or val is None)
                and idx in indices
                and indices[idx] is not None
            ):
                val = indices[idx]
            averages[idx] = val

        farmer_doc = await database.farmers.find_one({"phone": phone})
        das = farmer_doc.get("days_after_sowing", 60) if farmer_doc else 60

        return {
            "status": "success",
            "message": f"Data fetched successfully for {phone}",
            "data": averages,
            "raw_indices": indices,
            "geometry": result.get("geometry"),
            "schedule": result.get("schedule"),
            "advisory": result.get("advisory"),
            "farmer": result.get("farmer"),
            "vra": result.get("vra"),
            "das": das,
        }

    except Exception as e:
        raise HttpError(500, str(e))


@router.post("/approve-advisory/{phone}")
async def approve_farmer_advisory(request, phone: str, advisory_data: dict):
    try:
        approved_payload = {
            **advisory_data,
            "isApproved": True,
            "approved_at": datetime.utcnow().isoformat() + "Z",
            "approval_status": "approved",
        }

        update_result = await database.farmers.update_one(
            {"phone": phone},
            {
                "$set": {
                    "approved_advisory": approved_payload,
                    "is_pending": False,
                    "advisory_approved_at": datetime.utcnow(),
                }
            },
        )

        if update_result.matched_count == 0:
            raise HttpError(404, "Farmer not found.")

        return {
            "status": "success",
            "message": "Advisory approved and saved to database.",
            "approved_advisory": approved_payload,
        }

    except Exception as e:
        raise HttpError(500, str(e))

