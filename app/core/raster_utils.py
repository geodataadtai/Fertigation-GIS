# =============================================================================
# raster_utils.py — Raster / GeoTIFF Processing Helpers
#
# Provides utility functions for reading and analysing GeoTIFF satellite
# imagery files stored in MinIO.
#
# Rasterio is preferred for reading GeoTIFFs because it properly handles
# geospatial metadata and NoData values. If rasterio is not installed
# (e.g., on a lightweight deployment), tifffile is used as a fallback.
# =============================================================================

# --- Third-party imports ---
import io
import numpy as np
import tifffile

# Try to import rasterio; mark whether it is available so we can branch later
try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False


# =============================================================================
# Public Functions
# =============================================================================

def calculate_tiff_average(minio_client, bucket_name: str, object_path: str):
    """
    Downloads a GeoTIFF from MinIO and returns the mean pixel value of Band 1.

    Invalid pixels (clouds, masked areas, fill values) are filtered out before
    averaging. Pixels are considered valid only when their value falls in the
    range [-1.0, 1.0], which covers normalised vegetation/spectral indices
    such as NDVI, NDWI, etc.

    Args:
        minio_client: An initialised MinIO client object.
        bucket_name:  Name of the MinIO bucket that contains the file.
        object_path:  Path to the TIFF file inside the bucket.

    Returns:
        float: Mean of all valid pixels, or 0.0 on error / fully masked image.
    """
    try:
        # ------------------------------------------------------------------
        # Step 1: Download the raw TIFF bytes from MinIO
        # ------------------------------------------------------------------
        response = minio_client.get_object(bucket_name, object_path)
        tiff_bytes = response.read()

        # ------------------------------------------------------------------
        # Step 2: Decode the TIFF in memory (no temp files on disk)
        # ------------------------------------------------------------------
        if HAS_RASTERIO:
            # Rasterio path — preferred; handles geospatial metadata correctly
            with rasterio.MemoryFile(tiff_bytes) as memfile:
                with memfile.open() as dataset:
                    band = dataset.read(1)  # Read the first (and usually only) band
        else:
            # Fallback path — uses tifffile when rasterio is unavailable
            arr = tifffile.imread(io.BytesIO(tiff_bytes))

            if arr.ndim == 3:
                # 3-D array: figure out which axis is the band axis
                # Convention: if the first dimension is smaller it is the band axis (C, H, W)
                # otherwise the last dimension is bands (H, W, C)
                band = arr[0] if arr.shape[0] < arr.shape[2] else arr[:, :, 0]
            else:
                # 2-D array: already a single band
                band = arr

        # ------------------------------------------------------------------
        # Step 3: Filter out invalid / NoData pixels
        # Values outside [-1, 1] represent clouds, fill values, or background
        # (common NoData sentinel: -9999)
        # ------------------------------------------------------------------
        valid_pixels = band[(band >= -1.0) & (band <= 1.0)]

        # If every pixel was masked out, return 0 to signal "no data"
        if valid_pixels.size == 0:
            return 0.0

        # ------------------------------------------------------------------
        # Step 4: Compute and return the mean of the valid pixels
        # ------------------------------------------------------------------
        mean_value = float(np.mean(valid_pixels))
        return mean_value

    except Exception as e:
        print(f"Error calculating average for {object_path}: {e}")
        return 0.0