"""
Database Seeding Script
=======================
Populates the MongoDB database and MinIO storage with farmer records and KML field boundaries.

Workflow:
1. Verifies that the MinIO 'kml-storage' bucket exists (creates it if missing).
2. Reads farmer demographic and crop data from 'app/data/dataset.csv'.
3. Uploads each farmer's corresponding KML field boundary polygon to MinIO.
4. Inserts or updates the farmer document in MongoDB ('farmers' collection).
"""

import os
import asyncio
from typing import Dict, Any

import pandas as pd
from minio import Minio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv, find_dotenv

# Load environment credentials
load_dotenv(find_dotenv())

# =============================================================================
# Database & Object Storage Connections
# =============================================================================

MONGO_USER = os.getenv("MONGO_USER", "")
MONGO_PASS = os.getenv("MONGO_PASS", "")
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@127.0.0.1:27017/"

db_client = AsyncIOMotorClient(MONGO_URI)
database = db_client["kvk_geoai_db"]

minio_client = Minio(
    "127.0.0.1:9000",
    access_key=os.getenv("MINIO_USER"),
    secret_key=os.getenv("MINIO_PASS"),
    secure=False,
)


# =============================================================================
# Seeder Pipeline
# =============================================================================

async def run_seed() -> None:
    """Reads 25 Farmers Updated Data.xlsx, pushes KML files to MinIO, and upserts farmer records to MongoDB."""
    excel_path = os.path.join(os.path.dirname(__file__), "25 Farmers Updated Data.xlsx")
    kml_dir = "app/data/kml"

    # Ensure MinIO destination bucket exists
    bucket_name = "kml-storage"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f"Created MinIO bucket: '{bucket_name}'")

    if os.path.exists(excel_path):
        import openpyxl
        print(f"Reading farmer records from {excel_path}...")
        wb = openpyxl.load_workbook(excel_path, data_only=True)
        sheet = wb["Sheet1"]
        processed_count = 0

        for r in range(3, 28):
            row_vals = [c.value for c in sheet[r]]
            sr_no = int(row_vals[0])
            name = str(row_vals[1]).strip()
            phone = str(row_vals[2]).split('.')[0].strip()
            village = str(row_vals[3]).strip() if row_vals[3] else ""
            taluka = str(row_vals[4]).strip() if row_vals[4] else ""
            district = str(row_vals[5]).strip() if row_vals[5] else ""
            pincode = int(row_vals[6]) if row_vals[6] else None
            dot = row_vals[7]
            planting_mat = str(row_vals[8]).strip() if row_vals[8] else "Seedling"
            variety = str(row_vals[9]).strip() if row_vals[9] else "CO-86032"
            season = str(row_vals[10]).strip() if row_vals[10] else "Adsali"
            area_acre = float(row_vals[11]) if row_vals[11] is not None else 1.0
            soil_type = str(row_vals[12]).strip() if row_vals[12] else "Black"
            ph = float(row_vals[13]) if row_vals[13] is not None else 7.5

            soil_n = float(row_vals[14]) if row_vals[14] is not None else 160.0
            soil_p = float(row_vals[15]) if row_vals[15] is not None else 18.0
            soil_k = float(row_vals[16]) if row_vals[16] is not None else 250.0

            app_n = float(row_vals[18]) if row_vals[18] is not None else 0.0
            app_p = float(row_vals[19]) if row_vals[19] is not None else 0.0
            app_k = float(row_vals[20]) if row_vals[20] is not None else 0.0

            applied_till_date = {
                "N": round(app_n, 1),
                "P": round(app_p, 1),
                "K": round(app_k, 1)
            }

            dot_str = dot.strftime("%d %b %Y") if hasattr(dot, "strftime") else str(dot)
            minio_kml = f"{phone}_{sr_no}.kml"

            address_str = f"{village}, Tal. {taluka}, Dist. {district} - {pincode}"

            farmer_doc = {
                "sr_no": sr_no,
                "name": name,
                "phone": phone,
                "village": village,
                "taluka": taluka,
                "district": district,
                "pincode": pincode,
                "address": address_str,
                "transplanted": dot_str,
                "variety": variety,
                "season": season,
                "crop_season": season,
                "planting_material": planting_mat,
                "soil_type": soil_type,
                "area_acre": area_acre,
                "kml_filename": minio_kml,
                "soil_parameters": {
                    "pH": ph,
                    "N": soil_n,
                    "P": soil_p,
                    "K": soil_k,
                    "applied_till_date": applied_till_date
                },
                "applied_till_date": applied_till_date
            }

            await database["farmers"].update_one(
                {"phone": phone},
                {"$set": farmer_doc},
                upsert=True
            )
            processed_count += 1

        print(f"\n[SUCCESS] Seed complete from Excel: Processed {processed_count} farmers into MongoDB.")
        return

    raise FileNotFoundError(f"Primary source of truth '{excel_path}' not found.")


if __name__ == "__main__":
    asyncio.run(run_seed())
