#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audience Intelligence API — 

Endpoints:
- POST /audience        -> run_pipeline(desc): FULL pipeline result (demo-weighted cohorts)
- POST /amazon_demo     -> run_amazon_demo(output_of_run_pipeline, engine, normalized_target_demo)
- POST /relevance       -> build_relevance_table from run_pipeline output
- GET  /health          -> quick status
"""

from __future__ import annotations

import os, json, math, traceback
from pathlib import Path
from typing import Any, Iterable, Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------- ENV & SECRETS ----------------------
OPENAI_API_KEY = None
pineconeKey = None
user = passw = host = database = None

def _load_local_env() -> Dict[str, str]:
    if not os.getenv("RAILWAY_ENVIRONMENT"):
        """Look for a local dotenv file in common paths."""
        try:
            from dotenv import dotenv_values
        except Exception:
            return {}
        for p in [
            "./local.env",
            "./.env",
            "/Users/aaronbrace/Desktop/envs/metaSecEnv.txt",
            "/Users/aaronbrace/Downloads/env.txt",
            "/Users/aaronbrace/Desktop/interestApi/.env",
        ]:
            if Path(p).exists():
                return dotenv_values(p)
        return {}

if os.getenv("RAILWAY_ENVIRONMENT"):
    # Railway (production)
    OPENAI_API_KEY = os.getenv("OPENAIKEY") or os.getenv("OPENAPIKEY") or os.getenv("OPENAI_API_KEY")
    pineconeKey = os.getenv("PINECONE")
    user = os.getenv("DBUSER")
    passw = os.getenv("DBPASSWORD")
    host = os.getenv("DBHOST")
    database = os.getenv("DB", "PlatformIntegration")
else:
    # Local
    cfg = _load_local_env()
    OPENAI_API_KEY = cfg.get("OPENAIKEY") or cfg.get("OPENAPIKEY") or cfg.get("OPENAI_API_KEY")
    pineconeKey = cfg.get("PINECONE")
    user = cfg.get("DBUSER")
    passw = cfg.get("DBPASSWORD")
    host = cfg.get("DBHOST")
    database = cfg.get("DB", "PlatformIntegration")

# Soft validations (raise later only when needed)
def _require_env(var: str, value: str | None):
    if not value or not str(value).strip():
        raise RuntimeError(f"Missing required environment variable: {var}")



# ---------------------- OpenAI ----------------------
import openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

openai.api_key = OPENAI_API_KEY
_EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_CHAT_MODEL    = os.getenv("CHAT_MODEL", "gpt-4o-mini")

def _client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai>=1.0")
    return OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text_str: str) -> List[float]:
    client = _client()
    resp = client.embeddings.create(model=_EMBED_MODEL, input=text_str)
    return resp.data[0].embedding

def normalize_vector(vec: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(vec), dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# ---------------------- Pinecone ----------------------
from pinecone import Pinecone
pc = Pinecone(api_key=pineconeKey)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "cohorts")
index = pc.Index(PINECONE_INDEX)

# ---------------------- DB ----------------------
def get_engine() -> Engine:
    return create_engine(
        f"mysql+mysqlconnector://{user}:{passw}@{host}:3306/",
        echo=False
    )


# ---------------------- JSON Safety helpers ----------------------
def _to_py_scalar(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Robustly convert DataFrame to JSON-safe records (keeps numeric types)."""
    if df is None or len(df) == 0:
        return []
    # Replace NaN with None
    df2 = df.replace({np.nan: None})
    # Convert to records
    records: List[Dict[str, Any]] = df2.to_dict(orient="records")
    # Clean numpy scalars inside dicts
    out: List[Dict[str, Any]] = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, list):
                clean_row[k] = [_to_py_scalar(iv) for iv in v]
            elif isinstance(v, dict):
                clean_row[k] = {str(kk): _to_py_scalar(vv) for kk, vv in v.items()}
            else:
                clean_row[k] = _to_py_scalar(v)
        out.append(clean_row)
    return out

# ---------------------- 3) GPT DEMOGRAPHIC ESTIMATE ----------------------
def create_pen_portrait_demo_profile(description: str) -> Dict[str, float] | Dict[str, str]:
    client = _client()
    prompt = f"""
        Given this audience definition: {description}, 
        estimate the percentage breakdown by:
        - Gender (Male, Female)
        - Age groups (16–24, 25–34, 35–44, 45–54, 55–64, 65+)
        - Socioeconomic class (ABC1, C2DE)
        
        Return only the percentages as a raw Python dictionary. Do NOT include markdown, quotes, or explanation.
        
        Only output something like:
                                 {{'Male': 45, 'Female': 55, '16-24': 15, '25-34': 25, '35-44': 20, '45-54': 15, '55-64': 15, '65+': 10, 'ABC1': 70, 'C2DE': 30}}
        
        Where you are unsure of the demographic profile, make a best attempt or use the following as a default as a very last resort:
        
                                 {{'Male': 48, 'Female': 52, '16-24': 12, '25-34': 14, '35-44': 13, '45-54': 13, '55-64': 13, '65+': 35, 'ABC1': 52, 'C2DE': 48}}
        """
    resp = client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    content = (resp.choices[0].message.content or "").strip()
    # Try strict parse; fall back to raw if needed
    try:
        return json.loads(content.replace("'", "\""))
    except Exception:
        return {"raw": content}

# ---------------------- 4) PIPELINE RUNNER ----------------------
def run_pipeline(desc: str) -> pd.DataFrame:
    # 1) GPT demographics
    target_demo = create_pen_portrait_demo_profile(desc)

    # 2) Cohort similarity via Pinecone
    # (your original called embeddings twice; keep a single call sufficient for query)
    resp = openai.embeddings.create(input=[desc], model=_EMBED_MODEL)
    query_vec = resp.data[0].embedding

    # Pinecone response can be dict-like or object; handle both
    res = index.query(vector=query_vec, top_k=10, include_metadata=True)
    matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
    results = []
    for m in matches:
        mid = m["id"] if isinstance(m, dict) else getattr(m, "id", None)
        meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
        if score is not None and score >= 0.35:
            results.append({
                "id": mid,
                "attribute": (meta or {}).get("attribute"),
                "similarity": score
            })

    cohort_summary = pd.DataFrame(results)
    if cohort_summary.empty:
        # Return empty frame but attach target_demo so downstream endpoints still behave
        out = pd.DataFrame(columns=["attribute", "similarity", "importance_score", "Cohort"])
        out.attrs["target_demo"] = target_demo
        return out

    out = cohort_summary[['attribute', 'similarity']].copy()
    out['importance_score'] = out['similarity'] / out['similarity'].max()
    out['Cohort'] = out['attribute'].astype(str).str.lower().str.strip()

    # 3) Join to Amazon Match (CSV for now)
    
    engine = get_engine()
    amamatch = pd.read_sql("""
        SELECT * from PlatformIntegration.amazonMatchDaas
    """, engine)
    amamatch['Attribute'] = amamatch['Attribute'].astype(str).str.strip().str.lower()
    amamatch['Cohort'] = amamatch['Cohort'].astype(str).str.strip().str.lower()
    final_data = amamatch.merge(out, on='Cohort', how='left')
    final_data = final_data[final_data['importance_score'].notna()]

    # 4) Amazon Overlap (DB)
    engine = get_engine()
    dates_df = pd.read_sql("SELECT * FROM PlatformIntegration.datePeriods", engine)
    current_period = dates_df['currentPeriod'].iloc[0]

    overlap = pd.read_sql(f"""
        SELECT audienceNameA AS Attribute,
               audienceNameB AS AttributeTo,
               dailyReachU AS Reach,
               overlapPercentage
        FROM PlatformIntegration.amazonAudiencesOverlapExtended
        WHERE updatePeriod = '{current_period}'
    """, engine)
    overlap['Attribute'] = overlap['Attribute'].astype(str).str.lower().str.strip()
    overlap['AttributeTo'] = overlap['AttributeTo'].astype(str).str.lower().str.strip()
    overlap['overlapPercentage'] = pd.to_numeric(overlap['overlapPercentage'], errors='coerce') / 100

    final_data = final_data.loc[:, ~final_data.columns.duplicated()].copy()
    if "Attribute_x" in final_data.columns and "Attribute_y" in final_data.columns:
        final_data = final_data.drop(columns=["Attribute_y"]).rename(columns={"Attribute_x": "Attribute"})
    elif "Attribute_x" in final_data.columns:
        final_data = final_data.rename(columns={"Attribute_x": "Attribute"})
    elif "Attribute_y" in final_data.columns:
        final_data = final_data.rename(columns={"Attribute_y": "Attribute"})

    overlap = overlap.loc[:, ~overlap.columns.duplicated()].copy()
    final_data = final_data.merge(overlap, on="Attribute", how="left")

    # 5) Second-degree weighting
    final_data2 = final_data.merge(
        amamatch,
        left_on="AttributeTo",
        right_on="Attribute",
        how="left",
        suffixes=("", "_target")
    )
    final_data2 = final_data2.rename(columns={"Attribute_y": "Attribute", "Cohort_target": "Cohort_to"})
    if "Cohort_to" not in final_data2.columns:
        final_data2["Cohort_to"] = final_data2.get("Cohort", np.nan)
    final_data2["importance_score_y"] = (
        final_data2.get("overlapPercentage", 0).fillna(0) *
        final_data2.get("importance_score", 0).fillna(0)
    )

    # 6) Append second-degree + original
    table1 = out[["Cohort", "importance_score"]].copy()
    table2 = (
        final_data2[["Cohort_to", "importance_score_y"]]
        .rename(columns={"Cohort_to": "Cohort", "importance_score_y": "importance_score"})
        .groupby("Cohort", as_index=False)
        .mean()
    )
    combined = (
        pd.concat([table1, table2], ignore_index=True)
        .dropna(subset=["Cohort"])
        .sort_values("importance_score", ascending=False)
    )

    # 7) Demographic weighting (CSV for now)
    
    engine = get_engine()
    dates_df = pd.read_sql("SELECT * FROM PlatformIntegration.datePeriods", engine)
    current_period = dates_df['currentPeriod'].iloc[0]

    cohort_demos = pd.read_sql("""
        SELECT * FROM PlatformIntegration.cohortDemos
    """, engine)
    cohort_demos['Cohort'] = cohort_demos['Cohort'].astype(str).str.lower().str.strip()
    demo_data = combined.merge(cohort_demos, on='Cohort', how='left')

    # normalize target_demo into 0..1
    if isinstance(target_demo, dict):
        target_demo_norm = {
            k.lower().replace('+','65').replace('-',''): float(v)/100
            for k, v in target_demo.items() if isinstance(v, (int, float))
        }
    else:
        target_demo_norm = {}

    def demo_similarity(row):
        try:
            cvec = np.array([
                (row.get('Male_Profile') or 0)/100, (row.get('Female_Profile') or 0)/100,
                (row.get('Profile_1624') or 0)/100, (row.get('Profile_2534') or 0)/100,
                (row.get('Profile_3544') or 0)/100, (row.get('Profile_4554') or 0)/100,
                (row.get('Profile_5564') or 0)/100, (row.get('Profile_65') or 0)/100,
                (row.get('ABC1_Profile') or 0)/100, (row.get('C2DE_Profile') or 0)/100
            ])
            tvec = np.array(list(target_demo_norm.values()))
            if len(tvec) != len(cvec) or len(tvec) == 0:
                return 1.0
            return 1 - np.sum(np.abs(cvec - tvec))/2
        except Exception:
            return 1.0

    demo_data['demo_match'] = demo_data.apply(demo_similarity, axis=1)
    demo_data['importance_score'] = pd.to_numeric(demo_data['importance_score'], errors='coerce')
    demo_data['importance_score'] = (demo_data['importance_score'].fillna(0) * demo_data['demo_match'].fillna(1.0))
    demo_data = demo_data.sort_values('importance_score', ascending=False)

    # Attach for downstream use
    demo_data.attrs["target_demo"] = target_demo
    return demo_data

# ---------------------- 1) FLATTEN AND NORMALISE GPT DEMO ----------------------
def flatten_and_normalize_demo(profile_dict: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in (profile_dict or {}).items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if isinstance(subv, (int, float)):
                    flat[subk] = float(subv)
        elif isinstance(v, (int, float)):
            flat[k] = float(v)

    mapping = {
        "male": "male",
        "female": "female",
        "abc1": "abc1",
        "c2de": "c2de",
        "16-24": "age1624", "18-25": "age1624",
        "25-34": "age2534", "26-30": "age2534", "31-35": "age2534",
        "35-44": "age3544", "36-40": "age3544",
        "41-45": "age4554", "46-50": "age4554",
        "51-55": "age5564", "56-60": "age5564",
        "61-65": "age65", "65+": "age65"
    }

    normalized: Dict[str, float] = {}
    for k, v in flat.items():
        norm_key = mapping.get(k.lower().strip(), k.lower().strip())
        normalized[norm_key] = v
    return normalized

# ---------------------- 2) RUN AMAZON DEMOGRAPHIC EXTENSION ----------------------
def run_amazon_demo(final_audience_profile: pd.DataFrame, engine: Engine, target_demo: Dict[str, float]) -> pd.DataFrame:
    seedimp_df = final_audience_profile.copy()

    # top 500 cohorts (as in your current version)
    top_cohorts = (
        seedimp_df.sort_values('importance_score', ascending=False)
        .head(500)['Cohort'].tolist()
    )

    seedimp_df = final_audience_profile[['Cohort','importance_score']].rename(columns={'Cohort':'Attribute'})

    def _sql_quote(x: str) -> str:
        return "'" + str(x).replace("'", "''") + "'"

    in_list = ",".join([ _sql_quote(x) for x in top_cohorts ]) or "''"
    mapping_query = f"""
    SELECT DISTINCT Amazon AS audienceName, LD
    FROM PlatformIntegration.amazonMapping
    WHERE LD IN ({in_list}) AND similarity > 0.4
    """
    mapping_data = pd.read_sql(mapping_query, engine)

    relevant_audiences = mapping_data['audienceName'].dropna().unique().tolist()
    if len(relevant_audiences) == 0:
        return pd.DataFrame(columns=["audienceName", "mean_Affinity", "Category", "Response", "demo_group", "demo_weight"])
    def _sql_quote(x: str) -> str:
        return "'" + str(x).replace("'", "''") + "'"
    aud_list = ",".join([ _sql_quote(str(x)) for x in relevant_audiences ]) or "''"

    overlap_query = f"""
    SELECT audienceNameB, audienceNameA AS audienceName, affinity
    FROM PlatformIntegration.amazonAudiencesOverlapDemographics
    WHERE audienceNameB IN ({aud_list})
    """
    merged_data = pd.read_sql(overlap_query, engine)

    cohort_data = (
        merged_data.merge(mapping_data, left_on='audienceNameB', right_on='audienceName', how='left')
        .assign(Attribute=lambda df: df['LD'].astype(str).str.strip().str.lower())
        .merge(seedimp_df, on='Attribute', how='left')
    )
    cohort_data['WeightedAffinity'] = (
        pd.to_numeric(cohort_data['affinity'], errors='coerce') *
        pd.to_numeric(cohort_data['importance_score'], errors='coerce')
    )

    df = (
        cohort_data.groupby('audienceName_x', as_index=False)
        .agg({'WeightedAffinity':'sum'})
        .rename(columns={'WeightedAffinity':'mean_Affinity','audienceName_x':'audienceName'})
    )

    # full demos table
    demos = pd.DataFrame([
        ["Demo - Affluence: Average", "Affluence", "Average"],
        ["Demo - Affluence: High", "Affluence", "High"],
        ["Demo - Affluence: Higher than average", "Affluence", "Higher than average"],
        ["Demo - Affluence: Very High", "Affluence", "Very High"],
        ["Demo - Age 18-25", "Age", "18-25"],
        ["Demo - Age 26-30", "Age", "26-30"],
        ["Demo - Age 31-35", "Age", "31-35"],
        ["Demo - Age 36-40", "Age", "36-40"],
        ["Demo - Age 41-45", "Age", "41-45"],
        ["Demo - Age 46-50", "Age", "46-50"],
        ["Demo - Age 51-55", "Age", "51-55"],
        ["Demo - Age 56-60", "Age", "56-60"],
        ["Demo - Age 61-65", "Age", "61-65"],
        ["Demo - Males", "Gender", "Male"],
        ["Demo - Females", "Gender", "Female"],
        ["Demo - Home Owners", "Household Details", "Home Owners"],
        ["Demo - Income £100k - £150k", "Income", "£100k - £150k"],
        ["Demo - Income £100k+", "Income", "£100k+"],
        ["Demo - Income £150k+", "Income", "£150k+"],
        ["Demo - Income £30k - £40k", "Income", "£30k - £40k"],
        ["Demo - Income £40k - £50k", "Income", "£40k - £50k"],
        ["Demo - Income £50k - £60k", "Income", "£50k - £60k"],
        ["Demo - Income £60k - £70k", "Income", "£60k - £70k"],
        ["Demo - Income £70k - £100k", "Income", "£70k - £100k"],
        ["Demo - Length of residency: 1-3 years", "Household Details", "Length of residency: 1-3 years"],
        ["Demo - Length of residency: 11+ years", "Household Details", "Length of residency: 11+ years"],
        ["Demo - Length of residency: 4-10 years", "Household Details", "Length of residency: 4-10 years"],
        ["Demo - No presence of children", "Household Composition", "No presence of children"],
        ["Demo - Number of adults in household: 1 adult", "Household Composition", "No. adults in house: 1"],
        ["Demo - Number of adults in household: 2 adults", "Household Composition", "No. adults in house: 2"],
        ["Demo - Number of adults in household: 3 adults", "Household Composition", "No. adults in house: 3"],
        ["Demo - Number of adults in household: 4+ adults", "Household Composition", "No. adults in house: 4+"],
        ["Demo - Number of children in household: 1 child", "Household Composition", "No. children in house: 1"],
        ["Demo - Number of children in household: 2 children", "Household Composition", "No. children in house: 2"],
        ["Demo - Number of children in household: 3 children", "Household Composition", "No. children in house: 3+"],
        ["Demo - Presence of children", "Household Composition", "Presence of children"],
        ["Demo - Presence of Children aged 0-4", "Child Age", "Child aged 0-4"],
        ["Demo - Presence of Children aged 12-17", "Child Age", "Child aged 12-17"],
        ["Demo - Presence of Children aged 5-11", "Child Age", "Child aged 5-11"],
        ["Demo - Property value: £100k to £125k", "Property Value", "£100k - £125k"],
        ["Demo - Property value: £125k to £150k", "Property Value", "£125k - £150k"],
        ["Demo - Property value: £150k to £200k", "Property Value", "£150k - £200k"],
        ["Demo - Property value: £200k to £250k", "Property Value", "£200k - £250k"],
        ["Demo - Property value: £250k to £300k", "Property Value", "£250k - £300k"],
        ["Demo - Property value: £300k to £400k", "Property Value", "£300k - £400k"],
        ["Demo - Property value: £400k to £500k", "Property Value", "£400k - £500k"],
        ["Demo - Property value: £500k+", "Property Value", "£500k+"],
        ["Demo - Renters", "Household Details", "Renters"],
        ["Households with children (1 year olds)", "Child Age", "Child 1 year old"],
        ["Households with children (2 year olds)", "Child Age", "Child 2 years old"],
        ["Households with children (6-12 months old)", "Child Age", "Child 6-12 months old"],
    ], columns=["audienceName", "Category", "Response"])

    df = df.merge(demos, on="audienceName", how="left")

    def infer_demo_group(name: str):
        name = str(name).lower()
        if "males" in name: return "male"
        if "females" in name: return "female"
        if "abc1" in name: return "abc1"
        if "c2de" in name: return "c2de"
        if "age 16-24" in name or "age 18-25" in name: return "age1624"
        if "age 25-34" in name or "age 26-30" in name or "age 26-35" in name or "age 31-35" in name: return "age2534"
        if "age 35-44" in name or "age 36-40" in name or "age 36-45" in name: return "age3544"
        if "age 41-45" in name or "age 46-50" in name or "age 46-55" in name: return "age4554"
        if "age 51-55" in name or "age 56-60" in name or "age 56-65" in name: return "age5564"
        if "age 61-65" in name or "age 65" in name: return "age65"
        return None

    df["demo_group"] = df["audienceName"].apply(infer_demo_group)

    aud_demo_index_table = pd.DataFrame(list(target_demo.items()), columns=["Demo", "Index"])
    df = df.merge(aud_demo_index_table, left_on="demo_group", right_on="Demo", how="left").rename(columns={"Index": "demo_weight"})

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Demo_")], errors="ignore")
    df["mean_Affinity"] = pd.to_numeric(df["mean_Affinity"], errors="coerce")
    df["demo_weight"] = pd.to_numeric(df["demo_weight"], errors="coerce")

    mask = df["demo_weight"].notna()
    df.loc[mask, "mean_Affinity"] = df.loc[mask, "mean_Affinity"] * (df.loc[mask, "demo_weight"] / 100.0)

    df = df.drop(columns=["Demo"], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ---------------------- META RELEVANCE TABLE ----------------------
current_period = 'OCT2025'

def build_relevance_table(final_audience_profile: pd.DataFrame, engine: Engine, update_period: str) -> pd.DataFrame:
    seedimp_indexed = (
        final_audience_profile
        .sort_values("importance_score", ascending=False)
        .head(50)
        .assign(
            importance_score=lambda df: pd.to_numeric(df["importance_score"], errors='coerce'),
            CohortIndex=lambda df: df["importance_score"] / df["importance_score"].mean(skipna=True)
        )
    )
    attribute_list = seedimp_indexed["Cohort"].dropna().unique().tolist()

    def _sql_quote(x: str) -> str:
        return "'" + str(x).replace("'", "''") + "'"

    safe_values = [_sql_quote(str(x).lower()) for x in attribute_list]
    
    if not safe_values:
        return pd.DataFrame(columns=[
            "SuggestionName","entType","Index","Percentage","Relevance",
            "avgAudienceSize","description","Change"
        ])

    query_asz = f"""
    SELECT SuggestionName, entType, LOWER(Attribute) AS Attribute, SuggestionAudienceSizeUK AS SuggestionAudienceSize
    FROM PlatformIntegration.AttributeSizingMetaUK
    WHERE SuggestionAudienceSizeUK IS NOT NULL
      AND CAST(SuggestionAudienceSizeUK AS UNSIGNED) <= 48000000
      AND LOWER(Attribute) IN ({','.join(safe_values)})
    """
    attribute_sizing = pd.read_sql(query_asz, engine)

    query_mom = f"""
    SELECT SuggestionName, MoM_Pct_Change AS `Change`
    FROM PlatformIntegration.AttributeSizingMetaUKMonthlyMoM
    WHERE updatePeriod = '{current_period}'
    """
    changes = pd.read_sql(query_mom, engine)
    
    # ✅ Fix Change column BEFORE converting to integer strings
    changes["Change"] = pd.to_numeric(changes["Change"], errors="coerce")
    changes["Change"] = changes["Change"].replace([np.inf, -np.inf], np.nan).fillna(0)

    query_desc = """
    SELECT Attribute, COALESCE(description, 'No description available') AS description
    FROM PlatformIntegration.metaTiktokAttributeDescriptions
    """
    descriptions = pd.read_sql(query_desc, engine)

    attribute_sizing = attribute_sizing.rename(columns={"attribute": "Attribute"})
    seedimp_indexed = seedimp_indexed.rename(columns={"Cohort": "Attribute"})

    t100 = pd.merge(attribute_sizing, seedimp_indexed, on="Attribute", how="left")
    t100['SuggestionAudienceSize'] = pd.to_numeric(t100["SuggestionAudienceSize"])
    t100 = t100.groupby(["SuggestionName", "entType"], as_index=False).agg(
        avgAudienceSize=("SuggestionAudienceSize", "mean")
    )

    propAttsTable = attribute_sizing.merge(seedimp_indexed, on="Attribute", how="left")
    propAttsTable["SuggestionAudienceSize"] = pd.to_numeric(propAttsTable["SuggestionAudienceSize"])
    propAttsTable["avg_likely_interest"] = propAttsTable["SuggestionAudienceSize"] * propAttsTable["importance_score"]
    propAttsTable["popTotal"] = propAttsTable["SuggestionAudienceSize"] / 48_600_000
    propAttsTable = propAttsTable.dropna(subset=["avg_likely_interest"])
    propAttsTable = propAttsTable.loc[:, ["SuggestionName", "entType", "CohortIndex", "popTotal", "SuggestionAudienceSize"]]
    
    propAttsTable = (
        propAttsTable
        .groupby(["SuggestionName", "entType"], as_index=False)
        .agg(
            Index=("CohortIndex", "mean"),
            avg_popTotal=("popTotal", "mean"),
            avgAudienceSize=("SuggestionAudienceSize", "mean")
        )
    )
    propAttsTable["Index"] = propAttsTable["Index"] * 100
    propAttsTable["Percentage"] = np.minimum(propAttsTable["Index"] * propAttsTable["avg_popTotal"], 100)
    propAttsTable["Relevance"] = propAttsTable["Percentage"] * propAttsTable["Index"]

    FinalTable = (
    propAttsTable
    .merge(descriptions, left_on="SuggestionName", right_on="Attribute", how="left")
    .merge(changes, on="SuggestionName", how="left")
    .drop_duplicates(subset=["SuggestionName", "entType"])
    .sort_values("Relevance", ascending=False)
    )

    # ✅ CLEAN Change AGAIN after merge
    FinalTable["Change"] = pd.to_numeric(FinalTable["Change"], errors="coerce")
    FinalTable["Change"] = FinalTable["Change"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # ✅ Prepare a rounded string version for formatting
    change_int = FinalTable["Change"].round().astype(int)
    
    # ✅ Apply safe change labels
    FinalTable["Change"] = np.where(
        change_int > 0,
        "+" + change_int.astype(str) + "% higher",
        np.where(
            change_int < 0,
            change_int.astype(str) + "% lower",
            "no change"
        )
    )
    
    # ✅ Final pretty formatting & column order
    FinalTable = FinalTable.assign(
        Percentage=lambda df: df["Percentage"].map("{:.1f}%".format),
        Index=lambda df: df["Index"].round().astype(int)
    )
    
    FinalTable = FinalTable.loc[
        :,
        ["SuggestionName", "entType", "Index", "Percentage", "Relevance",
         "avgAudienceSize", "description", "Change"]
    ]
    return FinalTable


# ---------------------- FastAPI APP ----------------------
app = FastAPI(title="CraftyWolf Audience Intelligence API", version="1.0.0")

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
@app.middleware("http")
async def block_direct_railway(request: Request, call_next):
    app_env = os.getenv("APP_ENV", "development")
    host = request.headers.get("host", "")
    rapidapi_host = request.headers.get("x-rapidapi-proxy-secret") or request.headers.get("x-rapidapi-user")

    # Allow RapidAPI proxy traffic
    if rapidapi_host:
        return await call_next(request)

    # Block direct Railway access in production
    if app_env == "production" and "up.railway.app" in host:
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "code": 403,
                    "message": "Direct Railway access is forbidden. Please use the RapidAPI endpoint."
                }
            },
        )

    return await call_next(request)

class AudienceReq(BaseModel):
    desc: str

@app.get("/health")
def health():
    return {"status": "ok", "pinecone_index": PINECONE_INDEX}

@app.post("/audience")
def audience(req: AudienceReq):
    try:
        demo_df = run_pipeline(req.desc)
        return {
            "desc": req.desc,
            "demographics": demo_df.attrs.get("target_demo"),
            "rows": int(len(demo_df)),
            "results": df_to_records(demo_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.post("/amazon_demo")
def amazon_demo(req: AudienceReq):
    try:
        final_audience_profile = run_pipeline(req.desc)
        target_demo = final_audience_profile.attrs.get("target_demo") or {}
        engine = get_engine()
        df = run_amazon_demo(final_audience_profile, engine, target_demo)
        return {
            "desc": req.desc,
            "rows": int(len(df)),
            "results": df_to_records(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

from fastapi import Query

from enum import Enum
from fastapi import Query

class OrderBy(str, Enum):
    relevance = "relevance"
    index = "index"
    percentage = "percentage"

@app.get("/relevance")
def relevance(
    audience: str = Query(..., description="Audience description for relevance analysis"),
    order_by: OrderBy = Query(
        OrderBy.relevance,
        description="Order results by one of: relevance, index, or percentage",
    ),
):
    """
    GET /relevance?audience=rugby&order_by=index
    Runs the full relevance pipeline for a given audience description.
    Orders results by the chosen field (default: relevance).
    The 'Relevance' column is used for sorting but never included in output.
    """
    try:
        desc = audience.strip()
        if not desc:
            raise ValueError("Missing audience parameter (?audience=)")

        final_audience_profile = run_pipeline(desc)
        engine = get_engine()
        update_period = "2024-10"
        rel = build_relevance_table(final_audience_profile, engine, update_period)

        # --- numeric conversions for safe sorting ---
        rel["Relevance_num"] = pd.to_numeric(rel.get("Relevance"), errors="coerce")
        rel["Index_num"] = pd.to_numeric(rel.get("Index"), errors="coerce")

        if "Percentage" in rel.columns:
            rel["Percentage_num"] = (
                rel["Percentage"].astype(str).str.replace("%", "", regex=False)
            )
            rel["Percentage_num"] = pd.to_numeric(rel["Percentage_num"], errors="coerce")
        else:
            rel["Percentage_num"] = np.nan

        # --- sort by chosen field (descending) ---
        if order_by == OrderBy.index:
            rel = rel.sort_values("Index_num", ascending=False, kind="mergesort")
        elif order_by == OrderBy.percentage:
            rel = rel.sort_values("Percentage_num", ascending=False, kind="mergesort")
        else:  # OrderBy.relevance
            rel = rel.sort_values("Relevance_num", ascending=False, kind="mergesort")
        rel = rel.head(50)
        # --- drop all temporary and unwanted columns ---
        rel = rel.drop(
            columns=[
                "Relevance_num",
                "Index_num",
                "Percentage_num",
                "Relevance",  # ✅ always dropped, even if used for sorting
            ],
            errors="ignore",
        )

        return {
            "desc": desc,
            "order_by": order_by,
            "rows": int(len(rel)),
            "results": df_to_records(rel),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "trace": traceback.format_exc()},
        )

@app.get("/")
def root():
    return {"message": "API running — POST /audience, /amazon_demo, /relevance"}

