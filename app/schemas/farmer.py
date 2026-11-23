# =============================================================================
# farmer.py — Farmer Data Schemas (Pydantic Models)
#
# These models define the shape of farmer data used throughout the API:
#   - FarmerBase     : Core fields shared by all farmer-related operations
#   - FarmerCreate   : Schema used when creating a new farmer record (POST body)
#   - FarmerResponse : Schema returned by the API (includes the database ID)
#
# Pydantic validates all incoming and outgoing data automatically against
# these schemas, and generates the OpenAPI documentation from the Field metadata.
# =============================================================================

# --- Standard library imports ---
from typing import Optional, Dict, Any

# --- Third-party imports ---
from pydantic import BaseModel, Field


# =============================================================================
# Farmer Schemas
# =============================================================================

class FarmerBase(BaseModel):
    """
    Core farmer profile fields.

    All fields marked Optional have sensible defaults and may be omitted
    from the request body. Required fields (no default) must be provided.
    """
    sr_no: Optional[int] = Field(None, description="Serial number from the dataset")
    name: str = Field(..., example="Shivaji Govind Rajmane (ksf)")
    phone: str = Field(..., example="9422406360")
    transplanted: str = Field(..., example="10 Jul 2024")
    variety: Optional[str] = Field("86032", example="86032")
    address: str = Field(..., example="kranti agrani dr G D bapu lade, Tal- palus, Pin 416309")
    area_acre: Optional[float] = Field(1.0, description="Farm area in acres")
    soil_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Soil parameters such as pH, EC, N, P, K"
    )
    kml_filename: Optional[str] = Field(None, description="Name of the KML file in MinIO")


class FarmerCreate(FarmerBase):
    """
    Schema for creating a new farmer record.

    Inherits all fields from FarmerBase with no additions.
    Used as the request body for the POST /farmer endpoint.
    """
    pass


class FarmerResponse(FarmerBase):
    """
    Schema returned by the API after fetching or creating a farmer.

    Extends FarmerBase with the database-generated 'id' field,
    which is None until the record has been saved to MongoDB.
    """
    id: Optional[str] = None