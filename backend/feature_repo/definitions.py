import pandas as pd
from datetime import datetime, timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define the entity
customer = Entity(name="customer_id", join_keys=["customer_id"], description="Banking customer")

# Define the source
stress_signals_source = FileSource(
    path="d:/Games/Frontend/backend/feature_repo/data/stress_signals.parquet",
    event_timestamp_column="event_timestamp",
)

# Define the Feature View
stress_signals_fv = FeatureView(
    name="stress_signals",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="delayed_salary_days", dtype=Int64),
        Field(name="savings_decline_wow", dtype=Float32),
        Field(name="lending_app_upi_count", dtype=Int64),
        Field(name="utility_payment_delay_ratio", dtype=Float32),
        Field(name="discretionary_spend_ratio", dtype=Float32),
        Field(name="atm_withdrawal_velocity", dtype=Float32),
        Field(name="auto_debit_failure_count", dtype=Int64),
    ],
    online=True,
    source=stress_signals_source,
    tags={"team": "risk_management"},
)
