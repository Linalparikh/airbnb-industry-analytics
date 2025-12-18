import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
DB_NAME = os.getenv("DB_NAME", "airbnb")

LISTINGS_COL = os.getenv("LISTINGS_COL", "listings_raw")
REVIEWS_COL = os.getenv("REVIEWS_COL", "reviews_raw")

PRICE_CAP_DEFAULT = int(os.getenv("PRICE_CAP_DEFAULT", "1000"))
TOP_N_LOCATIONS = int(os.getenv("TOP_N_LOCATIONS", "15"))
SAMPLE_ROWS = int(os.getenv("SAMPLE_ROWS", "15000"))
