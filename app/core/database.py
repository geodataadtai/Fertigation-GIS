# =============================================================================
# database.py — Database & Object-Storage Clients
#
# This module initialises two persistent service connections that are shared
# across the entire application:
#   1. MongoDB  — the primary document database (via Motor async driver)
#   2. MinIO    — S3-compatible object storage for KML/TIFF files
#
# Both services run inside Docker containers, accessed over localhost.
# Credentials are loaded from the project .env file at startup.
# =============================================================================

# --- Standard library imports ---
import os
import asyncio

# --- Third-party imports ---
from motor.motor_asyncio import AsyncIOMotorClient
from minio import Minio
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the nearest .env file in the project tree
load_dotenv(find_dotenv())


# =============================================================================
# MongoDB Setup
# Supports both persistent ASGI loops and per-request WSGI event loops.
# =============================================================================

# Read MongoDB credentials from the environment
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
MONGO_HOST = "127.0.0.1"   # Localhost — Docker port is mapped to the host
MONGO_PORT = "27017"       # Default MongoDB port

# Build the connection string
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"

_clients = {}

class DatabaseProxy:
    """Dynamic proxy that routes calls to the AsyncIOMotorClient bound to the active event loop."""
    def _get_db(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            if "default" not in _clients:
                _clients["default"] = AsyncIOMotorClient(MONGO_URI)
            return _clients["default"]["kvk_geoai_db"]

        if loop not in _clients:
            _clients[loop] = AsyncIOMotorClient(MONGO_URI)
        return _clients[loop]["kvk_geoai_db"]

    def __getitem__(self, name):
        return self._get_db()[name]

    def __getattr__(self, name):
        return getattr(self._get_db(), name)

class ClientProxy:
    """Dynamic proxy that routes calls to the active event loop's AsyncIOMotorClient."""
    def _get_client(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            if "default" not in _clients:
                _clients["default"] = AsyncIOMotorClient(MONGO_URI)
            return _clients["default"]

        if loop not in _clients:
            _clients[loop] = AsyncIOMotorClient(MONGO_URI)
        return _clients[loop]

    def __getitem__(self, name):
        return self._get_client()[name]

    def __getattr__(self, name):
        return getattr(self._get_client(), name)

db_client = ClientProxy()
database = DatabaseProxy()


# =============================================================================
# MinIO (Object Storage) Setup
# MinIO stores large binary files (KML polygons, GeoTIFF rasters).
# =============================================================================

# MinIO runs on localhost over plain HTTP (no TLS needed for local Docker)
MINIO_ENDPOINT = "127.0.0.1:9000"
MINIO_USER = os.getenv("MINIO_USER")
MINIO_PASS = os.getenv("MINIO_PASS")

# Create the shared MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_USER,
    secret_key=MINIO_PASS,
    secure=False  # False because we are on HTTP (localhost), not HTTPS
)


# =============================================================================
# Bucket Initialisation
# =============================================================================

def initialize_buckets():
    """
    Ensures the required MinIO storage buckets exist.

    Called once at application startup. If a bucket is missing it is created
    automatically; existing buckets are left untouched.

    Buckets:
        - 'kml-storage'  : Stores farmer KML field-boundary files
        - 'tiff-cache'   : Stores downloaded Sentinel GeoTIFF raster files
    """
    required_buckets = ["kml-storage", "tiff-cache"]

    for bucket in required_buckets:
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
            print(f"Created MinIO bucket: {bucket}")
        else:
            print(f"MinIO bucket already exists: {bucket}")