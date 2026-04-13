from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import PipelineConfig
from .model_features import build_model_feature_frame


@dataclass
class ModelBundle:
    model: Any
    feature_cols: List[str]
    threshold: float
    name: str


def fit_demo_hgb_bundle(X: pd.DataFrame, y: pd.Series, name: str, threshold: float = 0.50, random_state: int = 42) -> ModelBundle:
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=150,
        min_samples_leaf=4,
        random_state=random_state,
    )
    model.fit(X, y)
    return ModelBundle(model=model, feature_cols=list(X.columns), threshold=float(threshold), name=name)


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    payload = {
        "model": bundle.model,
        "feature_cols": bundle.feature_cols,
        "threshold": bundle.threshold,
        "name": bundle.name,
    }
    joblib.dump(payload, path)


def load_model_bundle(path: str | Path) -> ModelBundle:
    payload = joblib.load(path)
    return ModelBundle(
        model=payload["model"],
        feature_cols=list(payload["feature_cols"]),
        threshold=float(payload.get("threshold", 0.50)),
        name=str(payload.get("name", Path(path).stem)),
    )


def _predict_binary(bundle: ModelBundle, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    X_use = X.reindex(columns=bundle.feature_cols, fill_value=0.0)
    proba = pd.Series(bundle.model.predict_proba(X_use)[:, 1], index=X.index, dtype=float)
    pred = (proba >= float(bundle.threshold)).astype(int)
    return proba, pred



def attach_target_predictions(
    decision_master_df: pd.DataFrame,
    cfg: PipelineConfig,
    model_a_bundle: ModelBundle,
    model_b_bundle: ModelBundle,
    fallback_to_rule: bool = False,
    combine_mode: str = "model_only",
) -> pd.DataFrame:
    """target A/B prediction attach.

    combine_mode
    ------------
    - model_only : final_pred = model_pred
    - rule_floor : final_pred = max(rule, model_pred)
    - rule_only  : final_pred = rule
    """
    out = decision_master_df.copy()

    # rule columns가 없으면 0으로 둔다.
    target_a_rule = out.get("target_a_rule", pd.Series(0, index=out.index)).fillna(0).astype(int)
    target_b_rule = out.get("target_b_rule", pd.Series(0, index=out.index)).fillna(0).astype(int)
    out["target_a_rule"] = target_a_rule
    out["target_b_rule"] = target_b_rule

    if fallback_to_rule:
        out["target_a_proba"] = target_a_rule.astype(float)
        out["target_b_proba"] = target_b_rule.astype(float)
        out["target_a_model_pred"] = target_a_rule.astype(int)
        out["target_b_model_pred"] = target_b_rule.astype(int)
        out["target_a_pred"] = target_a_rule.astype(int)
        out["target_b_pred"] = target_b_rule.astype(int)
        out["target_a_final_pred"] = target_a_rule.astype(int)
        out["target_b_final_pred"] = target_b_rule.astype(int)
        out["target_a_final_source"] = "rule_only"
        out["target_b_final_source"] = "rule_only"
        return out

    X = build_model_feature_frame(out, cfg=cfg)
    target_a_proba, target_a_model_pred = _predict_binary(model_a_bundle, X)
    target_b_proba, target_b_model_pred = _predict_binary(model_b_bundle, X)

    out["target_a_proba"] = target_a_proba
    out["target_b_proba"] = target_b_proba
    out["target_a_model_pred"] = target_a_model_pred.astype(int)
    out["target_b_model_pred"] = target_b_model_pred.astype(int)

    # 기존 downstream 호환용 pred는 model pred로 유지
    out["target_a_pred"] = out["target_a_model_pred"]
    out["target_b_pred"] = out["target_b_model_pred"]

    if combine_mode == "model_only":
        out["target_a_final_pred"] = out["target_a_model_pred"].astype(int)
        out["target_b_final_pred"] = out["target_b_model_pred"].astype(int)
        out["target_a_final_source"] = "model_only"
        out["target_b_final_source"] = "model_only"
    elif combine_mode == "rule_floor":
        out["target_a_final_pred"] = pd.concat([target_a_rule, out["target_a_model_pred"]], axis=1).max(axis=1).astype(int)
        out["target_b_final_pred"] = pd.concat([target_b_rule, out["target_b_model_pred"]], axis=1).max(axis=1).astype(int)
        out["target_a_final_source"] = "rule_floor"
        out["target_b_final_source"] = "rule_floor"
    elif combine_mode == "rule_only":
        out["target_a_final_pred"] = target_a_rule.astype(int)
        out["target_b_final_pred"] = target_b_rule.astype(int)
        out["target_a_final_source"] = "rule_only"
        out["target_b_final_source"] = "rule_only"
    else:
        raise ValueError(f"unsupported combine_mode: {combine_mode}")

    return out
