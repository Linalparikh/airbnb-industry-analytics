from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, LISTINGS_COL, REVIEWS_COL

def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME]

def collections_exist(db):
    cols = set(db.list_collection_names())
    return (LISTINGS_COL in cols) and (REVIEWS_COL in cols)

def get_cols(db):
    return db[LISTINGS_COL], db[REVIEWS_COL]
