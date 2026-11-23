# FertiNex — Django Ninja Backend 🌾

This is the unified **Django 5 + Django Ninja** backend implementation for the **FertiNex** Precision Agriculture, Satellite Telemetry & AgroGeo Advisory Platform.

---

## Architecture Overview

- **Framework**: Django 5.x + [Django Ninja](https://django-ninja.dev/) (High-performance async REST framework)
- **API Spec**: Async REST API with OpenAPI / Swagger UI auto-generation at `/api/v1/docs`
- **Database**: MongoDB (via `motor` and `pymongo`) for application data + SQLite for Django internal sessions / admin
- **Object Storage**: MinIO for KML boundaries and Sentinel GeoTIFF satellite rasters
- **Auth**: JWT Authentication (`PyJWT`) with Bearer tokens and role-based access (`farmer` / `admin`)
- **GIS / Remote Sensing**: Shapely, PyProj, Rasterio, Tifffile, Copernicus CDSE Sentinel-2 API
- **Biochemical Engines**: DNDC daily nitrogen model, CENTURY 30-year SOC model, PHREEQC aqueous geochemistry

---

## Directory Structure

```
backend_django/
├── manage.py             # Django command-line runner
├── requirements.txt      # Python dependencies
├── .env                  # Environment credentials
├── seed_data.py          # Data seeder for 25 farmers & KMLs
├── kvk_project/
│   ├── settings.py       # Django configuration, CORS, Database, Ninja apps
│   ├── urls.py           # Root routes ('/', '/health') and NinjaAPI mount
│   ├── asgi.py           # ASGI entrypoint for high-performance async
│   └── wsgi.py           # Standard WSGI entrypoint
├── api/
│   ├── ninja_api.py      # Master NinjaAPI instance
│   ├── security.py       # JWT Bearer authentication handlers
│   ├── routes_farmer.py  # Farmer profiles, geometry & master dashboard
│   ├── routes_auth.py    # Farmer / admin login & audit logs
│   ├── routes_admin.py   # Admin metrics, db-status, sentinel fetch & approvals
│   ├── routes_advisory.py# STCR/RDF recommendations, schedule, compliance, VRA
│   ├── routes_sentinel.py# CDSE status check & secure TIFF streaming
│   └── routes_biochemical.py # Biochemical simulations (DNDC, CENTURY, water chem)
└── app/                  # Pure-Python GIS, agent orchestrator & biochemical engines
```

---

## How to Run

### 1. Ensure Dependencies & Containers are Running
Ensure your Docker containers for MongoDB (`kvk_mongodb_poc`) and MinIO (`kvk_minio_poc`) are running:
```powershell
docker ps
```

### 2. Run with Django Development Server
From the `backend_django` directory:
```powershell
cd backend_django
.\venv\Scripts\Activate.ps1
python manage.py runserver 8000
```

### 3. Interactive API Documentation
Open your browser and navigate to:
- **Swagger UI**: [http://127.0.0.1:8000/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
