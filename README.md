# Airbnb Industry Analytics Platform

## Overview
This project implements an end-to-end Big Data analytics platform for Airbnb listings and reviews data.
The system processes over **5 million reviews** and **270k+ listings** using a scalable MongoDB backend and
provides interactive analytics via Streamlit.

## Architecture
- MongoDB Replica Set (Docker)
- Python ETL Pipeline
- Streamlit Analytics Dashboard

![Architecture Diagram](diagrams/architecture.png)

## Technology Stack
- Python 3.12
- MongoDB (Replica Set)
- Docker & Docker Compose
- Streamlit
- Pandas, PyMongo

## Project Components

### 1. Data Ingestion
- Raw Airbnb listings and reviews are ingested into MongoDB
- Duplicate removal and schema normalization applied

### 2. Data Cleaning
- Missing values handled
- Column normalization
- Business-ready collections generated

### 3. Analytics Dashboard
- Interactive filters (location, room type, price)
- Price distributions
- Location-based insights
- Host quality analysis

## How to Run

```bash
docker-compose up -d
python src/airbnb_pipeline/ingest_raw.py
python src/airbnb_pipeline/clean_data.py
streamlit run src/airbnb_pipeline/app.py
