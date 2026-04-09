from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PipelineConfig:
    material_code: str = "RAW_SUGAR"
    lt_months: int = 2
    buffer_months: int = 2
    horizon_months: int = 4

    monthly_usage_base_ton: float = 26_500.0
    moq_ton: float = 5_000.0
    lot_multiple_ton: float = 2_500.0
    warehouse_capacity_ton: float = 55_000.0
    safety_stock_ton: float = 5_000.0

    wc_pressure_conditional_threshold: float = 70.0
    wc_pressure_block_threshold: float = 90.0

    current_landed_cost_col: str = "now_landed_cost_per_ton"

    scenario_library: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "base": {
            "demand_mult": 1.00,
            "cost_mult": 1.00,
            "open_po_delay_months": 0,
            "candidate_delay_months": 0,
            "emergency_premium_mult": 1.00,
        },
        "stress": {
            "demand_mult": 1.05,
            "cost_mult": 1.03,
            "open_po_delay_months": 0,
            "candidate_delay_months": 0,
            "emergency_premium_mult": 1.10,
        },
        "shock": {
            "demand_mult": 1.12,
            "cost_mult": 1.08,
            "open_po_delay_months": 1,
            "candidate_delay_months": 1,
            "emergency_premium_mult": 1.25,
        },
    })
