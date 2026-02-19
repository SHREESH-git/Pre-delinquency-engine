from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import random
import asyncio
import logging
from aiokafka import AIOKafkaConsumer
import json
import pandas as pd
import pathlib
from mock_data import customers, dashboard_data, portfolio_data, operations_data, model_data
import numpy as np
from ml_core.feature_engineering import engineer_features
from ml_core.predictor import prepare_inputs, predict_pd
from ml_core.risk_engine import expected_loss, risk_bucket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analytics")

# DATASET PATH (from backup)
DATASET_PATH = r"D:\Games\Frontend\content_backup (2)\content_backup\content\financial_stress_full_bank_grade_dataset.csv"

class AnalyticsService:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.full_df = None    # All 90,000 records — used for ratios & aggregates
        self.sample_df = None  # Stratified 1,000 records — used for per-customer lists
        self.metrics = {}      # Loaded from model_metrics.json

    def load_data(self):
        """Load full dataset for aggregates and a stratified sample for per-row ops."""
        try:
            if not self.path.exists():
                logger.error(f"Dataset not found at {self.path}")
                return

            # --- Full dataset for accurate ratios ---
            self.full_df = pd.read_csv(self.path)
            logger.info(f"Loaded full dataset: {len(self.full_df):,} records")

            # --- Stratified 1,000-record sample for per-customer operations ---
            total_target = 1000
            categories = self.full_df['product_type'].unique()
            samples_per_cat = total_target // len(categories)

            stratified_samples = []
            for cat in categories:
                cat_df = self.full_df[self.full_df['product_type'] == cat]
                n_sample = min(len(cat_df), samples_per_cat)
                if n_sample > 0:
                    stratified_samples.append(cat_df.sample(n=n_sample, random_state=42))

            self.sample_df = pd.concat(stratified_samples)
            if len(self.sample_df) < total_target:
                remainder = total_target - len(self.sample_df)
                pool_remaining = self.full_df.drop(self.sample_df.index)
                if not pool_remaining.empty:
                    extra = pool_remaining.sample(n=min(len(pool_remaining), remainder), random_state=42)
                    self.sample_df = pd.concat([self.sample_df, extra])

            logger.info(f"Stratified sample: {len(self.sample_df):,} records across {len(categories)} categories")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")

    def _ensure_loaded(self):
        if self.full_df is None:
            self.load_data()

    def get_overview_metrics(self):
        self._ensure_loaded()
        if self.full_df is None:
            return dashboard_data

        df = self.full_df  # Use ALL 90,000 records for ratios

        avg_risk = df['risk_level_latent'].mean()
        safety_score = round(((3 - avg_risk) / 3) * 100, 1)
        unique_customers = df['customer_id'].nunique()
        defaults = df['loan_default_observed'].sum()
        total = len(df)
        success_rate = round(((total - defaults) / total) * 100, 1)

        # Portfolio type distribution (proportional from full dataset)
        type_counts = df['product_type'].value_counts()
        portfolio_type = {
            "labels": [t.replace("_", " ").title() for t in type_counts.index.tolist()],
            "data": (type_counts.values.astype(float)).tolist()
        }

        return {
            **dashboard_data,
            "safetyScore": str(safety_score),
            "activeProtected": f"{unique_customers:,}",
            "successRate": f"{success_rate}%",
            "portfolio_type": portfolio_type,
            "successTrend": f"+{round(success_rate - 70, 1)}% vs baseline",
            "safetyTrend": f"+{round(safety_score - 50, 1)}% vs baseline",
            "dataSource": f"Full Dataset ({total:,} records)"
        }

    def get_portfolio_metrics(self):
        """Returns real portfolio breakdown by product, region, and segment (all 90k records)."""
        self._ensure_loaded()
        if self.full_df is None:
            return portfolio_data

        df = self.full_df  # All 90,000 records for accurate ratios

        product_perf = []
        for ptype, grp in df.groupby('product_type'):
            default_rate = round(grp['loan_default_observed'].mean() * 100, 1)
            avg_emi = round(grp['emi_amount'].mean(), 0)
            avg_risk = round(grp['risk_level_latent'].mean(), 2)
            count = len(grp)
            risk_label = "Low" if avg_risk < 1 else "Medium" if avg_risk < 2 else "High"
            status = "Healthy" if default_rate < 5 else "Monitor" if default_rate < 15 else "Watch"
            product_perf.append({
                "type": ptype.replace("_", " ").title(),
                "active": f"{count:,}",
                "value": f"₹{avg_emi * count / 1e7:.1f} Cr",
                "ticket": f"₹{avg_emi / 1000:.1f} K",
                "score": str(round((3 - avg_risk) / 3 * 100, 1)),
                "risk": risk_label,
                "status": status,
                "defaultRate": f"{default_rate}%"
            })

        region_breakdown = []
        for rtier, grp in df.groupby('region_tier'):
            region_breakdown.append({
                "tier": rtier.replace("_", " ").title(),
                "count": f"{len(grp):,}",
                "defaultRate": f"{round(grp['loan_default_observed'].mean() * 100, 1)}%",
                "avgRisk": round(grp['risk_level_latent'].mean(), 2)
            })

        segment_breakdown = []
        for seg, grp in df.groupby('customer_segment'):
            segment_breakdown.append({
                "segment": seg.replace("_", " ").title(),
                "count": f"{len(grp):,}",
                "defaultRate": f"{round(grp['loan_default_observed'].mean() * 100, 1)}%",
                "avgIncome": f"₹{round(grp['monthly_income'].mean() / 1000, 1)} K"
            })

        total_emi = df['emi_amount'].sum()
        healthy_count = len(df[df['risk_level_latent'] <= 1])
        avg_score = round((3 - df['risk_level_latent'].mean()) / 3 * 100, 1)

        return {
            **portfolio_data,
            "totalValue": f"₹{total_emi / 1e7:.1f} Cr",
            "healthyCount": f"{healthy_count:,}",
            "healthyPercent": f"{round(healthy_count / len(df) * 100, 1)}%",
            "avgScore": str(avg_score),
            "productPerformance": product_perf,
            "regionBreakdown": region_breakdown,
            "segmentBreakdown": segment_breakdown,
            "dataSource": f"Full Dataset ({len(df):,} records)"
        }

    def get_operations_data(self):
        """Returns top 10 highest-risk customers from the stratified 1,000-record sample."""
        self._ensure_loaded()
        if self.sample_df is None:
            return operations_data

        df = self.sample_df.copy()
        top_risk = df.sort_values(['risk_level_latent', 'emi_to_income_ratio'], ascending=False).head(10)

        risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        action_map = {0: "Monitor", 1: "Send Reminder", 2: "Assign RM + Restructure", 3: "Immediate Contact"}
        status_map = {0: "Completed", 1: "In Progress", 2: "In Progress", 3: "Pending"}

        result = []
        for _, row in top_risk.iterrows():
            rl = int(row['risk_level_latent'])
            result.append({
                "id": row['customer_id'],
                "name": row['customer_id'],
                "product": row['product_type'].replace("_", " ").title(),
                "score": int(round((3 - rl) / 3 * 100)),
                "risk": risk_labels[rl],
                "action": action_map[rl],
                "status": status_map[rl],
                "statusRisk": risk_labels[rl],
                "emiRatio": f"{round(row['emi_to_income_ratio'] * 100, 1)}%",
                "salaryDelay": f"{int(row['salary_delay_days'])} days",
                "region": row['region_tier'].replace("_", " ").title()
            })
        return result

    def get_model_metrics(self):
        """Returns real model performance metrics from all 90,000 records."""
        self._ensure_loaded()
        if self.full_df is None:
            return model_data

        df = self.full_df  # All 90,000 records for accurate model metrics
        total = len(df)
        defaults = int(df['loan_default_observed'].sum())
        default_rate = round(defaults / total * 100, 1)
        high_risk = len(df[df['risk_level_latent'] >= 2])
        avg_risk = round(df['risk_level_latent'].mean(), 3)

        risk_dist = df['risk_level_latent'].value_counts().sort_index()
        risk_distribution = {
            "labels": ["Low (0)", "Medium (1)", "High (2)", "Critical (3)"],
            "data": [int(risk_dist.get(i, 0)) for i in range(4)]
        }
        
        auc = self.metrics.get("9_best_hybrid_holdout_auc", 0.0)
        auc_score = f"{round(auc * 100, 1)}%" if auc > 0 else "N/A"

        return {
            **model_data,
            "accuracy": f"{round(100 - default_rate, 1)}%",
            "defaultRate": f"{default_rate}%",
            "aucRoc": auc_score,
            "totalSampled": f"{total:,}",
            "highRiskCount": f"{high_risk:,}",
            "highRiskPct": f"{round(high_risk / total * 100, 1)}%",
            "avgRiskScore": str(avg_risk),
            "riskDistribution": risk_distribution,
            "dataSource": f"Full Dataset ({total:,} records) + Hybrid Model Metrics"
        }

    def get_customer_features(self, customer_id: str):
        """Looks up a customer's latest row from the full CSV for ML inference."""
        self._ensure_loaded()
        try:
            if self.full_df is None:
                return None
            cust_rows = self.full_df[self.full_df['customer_id'] == customer_id]
            if cust_rows.empty:
                return None
            return cust_rows.sort_values('month').iloc[-1].to_dict()
        except Exception as e:
            logger.error(f"Failed to look up customer {customer_id}: {e}")
            return None

    def get_customer_history(self, customer_id: str) -> pd.DataFrame:
        """Returns full history dataframe for a customer."""
        self._ensure_loaded()
        if self.full_df is None:
            return pd.DataFrame()
        cust_rows = self.full_df[self.full_df['customer_id'] == customer_id]
        return cust_rows.sort_values('month')


analytics_service = AnalyticsService(DATASET_PATH)



kafka_logger = logging.getLogger("kafka-ingestion")

app = FastAPI(title="Banking Risk Dashboard API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models (for documentation and validation) ---
class CustomerDriver(BaseModel):
    name: str
    value: int | str
    percent: int
    color: str

class CustomerTimeline(BaseModel):
    title: str
    desc: str
    date: str
    type: str

class Customer(BaseModel):
    name: str
    loanType: str
    outstanding: str
    emi: str
    age: str
    branch: str
    riskScore: int
    probability: float
    confidence: float
    riskLevel: str
    drivers: List[CustomerDriver]
    timeline: List[CustomerTimeline]

class PredictionRequest(BaseModel):
    customer_id: str
    # FUTURE: Add more features here for real-time inference
    # income: float
    # credit_score: int
    # loan_amount: float
    # employment_status: str

class PredictionResponse(BaseModel):
    risk_score: int
    risk_level: str
    probability: float
    expected_loss: float
    ead: float
    lgd: float
    # FUTURE: Add SHAP values or explanation
    # explanation: dict

class DashboardData(BaseModel):
    safetyScore: str
    safetyTrend: str
    activeProtected: str
    accountsTrend: str
    successRate: str
    successTrend: str
    portfolio_type: dict
    tenure: dict
    portfolio_growth: dict
    disbursement: dict
    recentAlerts: List[dict]

# --- Kafka Stream Models ---
class StreamConfig(BaseModel):
    bootstrap_servers: List[str] = ["localhost:9092", "localhost:9093", "localhost:9094"]
    topic: str = "banking-risk-events"
    group_id: str = "risk-dashboard-group"

class StreamStatus(BaseModel):
    status: str
    active_topic: Optional[str] = None
    messages_ingested: int
    connected_brokers: List[str]
    latest_event: Optional[dict] = None

# --- Kafka Consumer Logic ---
class IngestionManager:
    def __init__(self):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.task: Optional[asyncio.Task] = None
        self.config: Optional[StreamConfig] = None
        self.status = "stopped"
        self.message_count = 0
        self.last_message = None

    async def start(self, config: StreamConfig):
        if self.status == "running":
            return False
        
        self.config = config
        self.consumer = AIOKafkaConsumer(
            self.config.topic,
            bootstrap_servers=",".join(self.config.bootstrap_servers),
            group_id=self.config.group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest'
        )
        
        await self.consumer.start()
        self.status = "running"
        self.task = asyncio.create_task(self._consume_loop())
        kafka_logger.info(f"Kafka ingestion started on topic: {self.config.topic}")
        return True

    async def _consume_loop(self):
        try:
            async for msg in self.consumer:
                self.message_count += 1
                self.last_message = msg.value
                # In a real app, process the message (e.g., update DB or trigger model)
                if self.message_count % 100 == 0:
                    kafka_logger.info(f"Ingested {self.message_count} messages")
        except asyncio.CancelledError:
            kafka_logger.info("Ingestion task cancelled")
        except Exception as e:
            kafka_logger.error(f"Error in Kafka loop: {e}")
            self.status = "error"
        finally:
            await self.consumer.stop()

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.status = "stopped"
        kafka_logger.info("Kafka ingestion stopped")

ingestion_manager = IngestionManager()
import os
import sys

# Add ml_core to path for easy imports
sys.path.append(os.path.join(os.path.dirname(__file__), "ml_core"))

from model_loader import load_models
from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    # Load the ML/DL model hybrid ensemble
    try:
        app.state.models = load_models("models")
        logger.info("Hybrid ML/DL Risk Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        app.state.models = None
    
    # Pre-load analytics dataset (full 90k + stratified 1k sample)
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, analytics_service.load_data)
    logger.info("Analytics dataset preloaded successfully")

    # Load model metrics
    try:
        metrics_path = pathlib.Path("models/model_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                analytics_service.metrics = json.load(f)
            logger.info("Model metrics loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model metrics: {e}")

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Banking Risk Dashboard API is running"}

@app.get("/api/dashboard/overview", response_model=DashboardData)
async def get_dashboard_overview():
    """Returns dynamic overview metrics derived from the real CSV dataset (stratified 1,000 records)."""
    metrics = analytics_service.get_overview_metrics()
    return metrics

@app.get("/api/details/alert/{customer_id}")
async def get_alert_details(customer_id: str):
    """Returns detailed alert info for a specific customer/segment."""
    from mock_data import customers
    if customer_id not in customers:
        return {"error": "Alert source not found"}
    
    c = customers[customer_id]
    return {
        "title": f"Risk Alert: {c['name']}",
        "reason": f"High PD detected ({(c['probability']*100):.1f}%) in {c['loanType']}",
        "drivers": c['drivers'],
        "timeline": c['timeline'],
        "recommendation": "Initiate restructuring discussion or assign Senior RM."
    }

@app.get("/api/details/product/{product_type}")
async def get_product_details(product_type: str):
    """Returns detailed performance metrics for a loan product."""
    from mock_data import portfolio_data
    # Filter from portfolio performance
    perf = next((p for p in portfolio_data['productPerformance'] if p['type'].lower() == product_type.lower()), None)
    if not perf:
        return {"error": "Product performance data not found"}
    
    return {
        "title": f"{perf['type']} Analysis",
        "metrics": {
            "Total AUM": perf['value'],
            "Average Ticket Size": perf['ticket'],
            "Delinquency Rate": "4.2%",
            "Growth (QoQ)": "+12.5%",
            "Active Customers": perf['active']
        },
        "breakdown": [
            {"label": "Standard", "value": "92%"},
            {"label": "Sub-standard", "value": "5%"},
            {"label": "Doubtful", "value": "3%"}
        ]
    }

@app.get("/api/portfolio/overview")
def get_portfolio_overview():
    """Returns real portfolio data from the CSV dataset."""
    return analytics_service.get_portfolio_metrics()

@app.get("/api/operations")
def get_operations_data():
    """Returns top 10 at-risk customers from the real CSV dataset."""
    return analytics_service.get_operations_data()

@app.get("/api/alerts/priority")
def get_priority_alerts():
    """Returns top 3 priority alerts for the dashboard sidebar."""
    # Reuse operations data logic but limiting to top 3
    data = analytics_service.get_operations_data()
    return data[:3]




@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns real model performance metrics from the CSV dataset."""
    # Try model_metrics.json first, then fall back to CSV-derived metrics
    metrics_path = "models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            base = json.load(f)
        csv_metrics = analytics_service.get_model_metrics()
        return {**base, **csv_metrics}
    return analytics_service.get_model_metrics()

@app.get("/api/customers/{customer_id}/stress-signals")
def get_customer_stress_signals(customer_id: str):
    """
    Returns stress signals for a specific customer.
    In a real app, this would query Feast or a behavioral database.
    """
    # Aligning with ML logic: salary delay, utilization, etc.
    customer = customers.get(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    # Return stress signals relevant to the risk engine
    return [
        {"id": "delayed_salary", "label": "Salary Credit Latency", "status": "CRITICAL" if customer.get("salary_delay", 0) > 5 else "LOW", "value": f"{customer.get('salary_delay', 0)} days"},
        {"id": "savings_decline", "label": "WoW Savings Decline", "status": "HIGH", "value": "-18%"},
        {"id": "lending_upi", "label": "Lending App Velocity", "status": "MEDIUM", "value": "3 txns"},
        {"id": "utilization", "label": "Credit Utilization", "status": "HIGH" if customer.get("utilization", 0) > 0.75 else "LOW", "value": f"{int(customer.get('utilization', 0)*100)}%"},
        {"id": "auto_debit_fail", "label": "Auto-Debit Failures", "status": "HIGH", "value": "2 failed"},
    ]

@app.get("/api/customers/{customer_id}", response_model=Customer)
def get_customer(customer_id: str):
    """Returns detailed profile for a specific customer."""
    customer = customers.get(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.post("/api/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    """
    Performs real-time hybrid ML risk inference.
    """
    if not hasattr(app.state, "models") or app.state.models is None:
        # Fallback to mock logic if model failed to load
        customer = customers.get(request.customer_id)
        if customer:
            return {
                "risk_score": customer["riskScore"],
                "risk_level": customer["riskLevel"],
                "probability": customer["probability"]
            }
        return {
            "risk_score": 45,
            "risk_level": "Medium Risk",
            "probability": 0.45
        }

    # --- REAL INFERENCE FLOW ---
    # 1. Fetch raw features (In prod, this comes from Feast/DB)
    # For now, we simulate with a baseline and maybe some random noise or mock historical data
    # In a real app, we'd query: df_raw = online_store.get_historical_features(customer_id)
    
    # We'll use a helper to generate synthetic raw data for the demo
    df_raw = generate_mock_history(request.customer_id)
    
    # 2. Feature Engineering
    df_engineered = engineer_features(df_raw)
    
    # 3. Preparation (Dummy encoding)
    cat_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()
    df_prep = pd.get_dummies(df_engineered, columns=cat_cols, drop_first=True)
    
    # 4. Hybrid Prediction
    tree_input, lstm_input = prepare_inputs(app.state.models, df_prep)
    pd_score = predict_pd(app.state.models, tree_input, lstm_input)
    
    risk_lvl = risk_bucket(pd_score)
    
    # 5. Expected Loss Calculation
    last_row = df_engineered.iloc[-1]
    emi = last_row["emi_amount"]
    utilization = last_row["credit_card_utilization"]
    income = last_row["monthly_income"]
    
    salary_flag = int(last_row["salary_delay_days"] > 5)
    util_flag = int(last_row["credit_card_utilization"] > 0.75)
    
    el, lgd, ead = expected_loss(
        app.state.models,
        pd_score,
        emi,
        utilization,
        income,
        salary_flag,
        util_flag
    )
    
    return {
        "risk_score": int(pd_score * 100),
        "risk_level": risk_lvl,
        "probability": round(float(pd_score), 4),
        "expected_loss": round(float(el), 2),
        "ead": round(float(ead), 2),
        "lgd": round(float(lgd), 4)
    }

@app.post("/api/predict/customer")
def predict_customer_risk(request: PredictionRequest):
    """
    Predict EL and PD for a real CSV customer ID (e.g. CUST0000000).
    Aggregates predictions over the customer's full history (rolling window).
    """
    # 1. Look up customer history from the full CSV
    hist_df = analytics_service.get_customer_history(request.customer_id)
    
    if hist_df.empty:
        # Fallback to mock generated history if not found in CSV
        # Or better, try getting single features and wrap it
        features = analytics_service.get_customer_features(request.customer_id)
        if not features:
            raise HTTPException(status_code=404, detail="Customer not found in dataset")
        df_engineered = pd.DataFrame([features]) # Minimal fallback
    else:
        # 2. Engineer features on FULL history
        try:
            # Ensure numeric columns are actually numeric to avoid "add.reduce" errors
            numeric_cols = ["active_products_count", "credit_card_utilization", "total_monthly_obligation", 
                            "emi_amount", "days_to_emi", "emi_to_income_ratio", "salary_delay_days", 
                            "weekly_balance_change_pct", "atm_withdrawal_amount", "monthly_income"]
            
            # Avoid SettingWithCopyWarning
            hist_df = hist_df.copy()
            for col in numeric_cols:
                if col in hist_df.columns:
                    hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)

            df_engineered = engineer_features(hist_df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

    # 3. Hybrid Prediction (if models loaded)
    if not hasattr(app.state, "models") or app.state.models is None:
        # Fallback: use the CSV risk_level_latent directly from latest row
        last_row = df_engineered.iloc[-1]
        rl = int(last_row.get("risk_level_latent", 1))
        pd_score = rl / 3.0
        risk_lvl = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"][rl]
        emi = last_row.get("emi_amount", 18000)
        income = last_row.get("monthly_income", 60000)
        el = round(pd_score * emi * 0.45, 2)
        ead = round(emi * 12, 2)
        lgd = 0.45
        features = last_row.to_dict()
    else:
        # Full history is passed to engineer_features above.
        # This generates rolling features for the entire history.
        # We predict for the LATEST row, which encapsulates the full history state.
        
        # Prepare categorical columns
        cat_cols = df_engineered.select_dtypes(include=["object"]).columns.tolist()
        df_prep = pd.get_dummies(df_engineered, columns=cat_cols, drop_first=True)
        
        # Prepare inputs for the WHOLE dataframe.
        # prepare_inputs will extract:
        # 1. The LAST row for Tree models (State at T)
        # 2. The LAST seq_len rows for LSTM (Sequence T-2, T-1, T)
        try:
            tree_input, lstm_input = prepare_inputs(app.state.models, df_prep)
            pd_score = predict_pd(app.state.models, tree_input, lstm_input)
            
            # Ensure scalar result (Aggregate rolling progress if array)
            if isinstance(pd_score, np.ndarray):
                if pd_score.size > 1:
                    pd_score = float(np.mean(pd_score))
                else:
                    pd_score = float(pd_score.item())
            elif isinstance(pd_score, (list, tuple)):
                 pd_score = float(np.mean(pd_score))
            else:
                 pd_score = float(pd_score)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(500, f"Model inference failed: {e}")
        
        risk_lvl = risk_bucket(pd_score)
        
        # Use latest row for EL calculation specifics
        last_row = df_engineered.iloc[-1]
        emi = last_row.get("emi_amount", 0)
        utilization = last_row.get("credit_card_utilization", 0)
        income = last_row.get("monthly_income", 0)
        salary_flag = int(last_row.get("salary_delay_days", 0) > 5)
        util_flag = int(last_row.get("credit_card_utilization", 0) > 0.75)
        
        el, lgd, ead = expected_loss(app.state.models, pd_score, emi, utilization, income, salary_flag, util_flag)
        features = last_row.to_dict()

    # 4. Build key feature drivers for display
    key_features = [
        {"label": "EMI to Income Ratio", "value": f"{round(features.get('emi_to_income_ratio', 0) * 100, 1)}%"},
        {"label": "Credit Utilization", "value": f"{round(features.get('credit_card_utilization', 0) * 100, 1)}%"},
        {"label": "Salary Delay", "value": f"{int(features.get('salary_delay_days', 0))} days"},
        {"label": "Monthly Income", "value": f"₹{int(features.get('monthly_income', 0)):,}"},
        {"label": "EMI Amount", "value": f"₹{int(features.get('emi_amount', 0)):,}"},
        {"label": "Product", "value": str(features.get("product_type", "N/A")).replace("_", " ").title()},
    ]

    return {
        "customer_id": request.customer_id,
        "risk_score": int(float(pd_score) * 100),
        "risk_level": risk_lvl,
        "probability": round(float(pd_score), 4),
        "expected_loss": round(float(el), 2),
        "ead": round(float(ead), 2),
        "lgd": round(float(lgd), 4),
        "key_features": key_features,
        "actual_risk_level": int(features.get("risk_level_latent", -1)),
        "actual_default": int(features.get("loan_default_observed", -1))
    }

def generate_mock_history(customer_id: str):
    """Generates synthetic history for a customer to feed the FE engine."""
    import pandas as pd
    # Baseline for a 'salaried' customer
    data = []
    for i in range(3): # 3 months history
        data.append({
            "customer_id": customer_id,
            "month": f"2024-0{3+i}",
            "customer_segment": "salaried",
            "region_tier": "tier_1",
            "product_type": "personal_loan",
            "active_products_count": 2,
            "credit_card_utilization": 0.3 + (i * 0.05),
            "total_monthly_obligation": 25000,
            "emi_amount": 18000,
            "days_to_emi": 15,
            "emi_to_income_ratio": 0.20,
            "salary_delay_days": random.randint(0, 7),
            "weekly_balance_change_pct": random.randint(-10, 5),
            "atm_withdrawal_amount": 12000,
            "monthly_income": 60000
        })
    return pd.DataFrame(data)

@app.get("/api/debug/synthetic-data/{customer_id}")
async def get_synthetic_data(customer_id: str):
    """Exposes raw synthetic history for verification."""
    df = generate_mock_history(customer_id)
    return df.to_dict(orient="records")

# --- Kafka Stream Control Endpoints ---

@app.get("/api/streams/status", response_model=StreamStatus)
async def get_stream_status():
    """Returns the current status of the Kafka ingestion stream."""
    return {
        "status": ingestion_manager.status,
        "active_topic": ingestion_manager.config.topic if ingestion_manager.config else None,
        "messages_ingested": ingestion_manager.message_count,
        "connected_brokers": ingestion_manager.config.bootstrap_servers if ingestion_manager.config else [],
        "latest_event": ingestion_manager.last_message
    }

@app.post("/api/streams/config")
async def configure_stream(config: StreamConfig):
    """Configures Kafka parameters (requires restart to take effect)."""
    ingestion_manager.config = config
    return {"message": "Configuration updated", "config": config}

@app.post("/api/streams/start")
async def start_stream():
    """Starts the Kafka ingestion process."""
    if not ingestion_manager.config:
        # Default config if none provided
        ingestion_manager.config = StreamConfig()
    
    success = await ingestion_manager.start(ingestion_manager.config)
    if not success:
        return {"message": "Ingestion is already running or failed to start"}
    return {"message": "Ingestion started", "topic": ingestion_manager.config.topic}

@app.post("/api/streams/stop")
async def stop_stream():
    """Stops the Kafka ingestion process."""
    await ingestion_manager.stop()
    return {"message": "Ingestion stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
