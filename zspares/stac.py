from pystac import Collection, Extent, SpatialExtent, TemporalExtent
from datetime import datetime

# Define collection metadata
collection = Collection(
    id="your-dataset-id",
    description="Description of your dataset",
    extent=Extent(
        spatial=SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),  # Update with actual bbox
        temporal=TemporalExtent([[datetime(2023, 1, 1), datetime(2024, 1, 1)]]),  # Temporal range
    ),
    license="CC-BY-4.0",
)

# Save to JSON
collection.normalize_and_save("collection.json")
