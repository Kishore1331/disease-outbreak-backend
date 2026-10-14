from typing import Optional
from fastapi import APIRouter, Query
from backend.services.data_service import data_service
from backend.services.risk_service import risk_service
from backend.schemas import WardsResponse, RiskResponse

router = APIRouter(prefix="/api", tags=["Wards & Risk"])

@router.get("/wards", response_model=WardsResponse)
async def get_wards():
    """
    Returns verified municipal ward identifiers, census populations,
    and geographic coordinates from official Census 2011 PCA data.
    """
    wards = data_service.get_wards()
    return WardsResponse(
        district="Coimbatore",
        total_wards=len(wards),
        data_source="Census of India 2011 — Primary Census Abstract (DDW_PCA3331_2011_MDDS)",
        census_year=2011,
        wards=wards
    )

@router.get("/risk", response_model=RiskResponse)
async def get_risk_assessment(
    disease: Optional[str] = Query("Dengue", description="Target disease for risk assessment")
):
    """
    Calculates epidemiological risk categories using documented multi-factor index
    (attack rate per 1,000 population, case volume thresholds, and climate suitability).
    """
    risk_data = await risk_service.evaluate_wards_risk(disease=disease or "Dengue")
    return risk_data

@router.get("/clusters")
async def get_ward_clusters(
    disease: Optional[str] = Query("Dengue", description="Target disease")
):
    """
    HT-13 Component: Geospatial & Epidemiological Clustering.
    Applies K-Means clustering across municipal wards combining:
    - Attack rate per 1,000 population
    - Composite EVOI risk score
    - Commuter transit mobility index
    - Population density
    """
    from backend.ml.clustering import ClusteringEngine
    wards = data_service.get_wards()
    risk_data = await risk_service.evaluate_wards_risk(disease=disease or "Dengue")
    clusters_result = ClusteringEngine.perform_clustering(
        wards=wards,
        risk_items=risk_data.wards,
        n_clusters=3
    )
    return clusters_result
