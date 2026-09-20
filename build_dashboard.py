import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

REQUIRED_SHEETS = {
    "Shipments": [
        "po_id", "supplier_id", "supplier_name", "sku", "sku_name", "destination",
        "quantity", "po_date", "expected_handover_days", "supplier_handover_date",
        "vessel_departure_date", "planned_sailing_days", "actual_arrival_date",
    ],
    "Inventory": [
        "sku", "sku_name", "destination", "current_inventory", "daily_demand", "safety_stock",
    ],
    "Supplier_Risk": [
        "supplier_id", "supplier_name", "avg_handover_days", "avg_sailing_delay_days",
        "on_time_rate_pct", "historical_risk_score",
    ],
}
DATE_COLUMNS = {
    "Shipments": ["po_date", "supplier_handover_date", "vessel_departure_date", "actual_arrival_date"],
}
NUMERIC_COLUMNS = {
    "Shipments": ["quantity", "expected_handover_days", "planned_sailing_days"],
    "Inventory": ["current_inventory", "daily_demand", "safety_stock"],
    "Supplier_Risk": ["avg_handover_days", "avg_sailing_delay_days", "on_time_rate_pct", "historical_risk_score"],
}


class FriendlyError(Exception):
    """Raised for problems the end user needs to fix in their spreadsheet."""


# ----------------------------------------------------------------------------
# Step 1 — Read + validate the workbook
# ----------------------------------------------------------------------------
def load_and_validate(xlsx_path: Path):
    try:
        book = pd.read_excel(xlsx_path, sheet_name=None, dtype=object)
    except Exception as e:
        raise FriendlyError(
            f"Could not open '{xlsx_path.name}' as an Excel file.\n"
            f"Make sure it is a real .xlsx file (not .csv/.xls renamed) and isn't "
            f"open in another program.\nUnderlying error: {e}"
        )

    problems = []
    for sheet_name, required_cols in REQUIRED_SHEETS.items():
        if sheet_name not in book:
            problems.append(
                f"Missing sheet '{sheet_name}'. Found sheets: {', '.join(book.keys())}. "
                f"Sheet names must match Data_Template.xlsx exactly (case-sensitive)."
            )
            continue

        df = book[sheet_name]
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            problems.append(
                f"Sheet '{sheet_name}' is missing required column(s): {', '.join(missing)}. "
                f"Do not rename or remove columns from Data_Template.xlsx."
            )
            continue

        df = df.dropna(how="all")
        df = df[df[required_cols[0]].notna()]
        if df.empty:
            problems.append(f"Sheet '{sheet_name}' has no data rows (only headers/samples were removed).")
            continue

        for col in NUMERIC_COLUMNS.get(sheet_name, []):
            bad = df[pd.to_numeric(df[col], errors="coerce").isna()]
            if not bad.empty:
                problems.append(
                    f"Sheet '{sheet_name}', column '{col}': {len(bad)} row(s) aren't valid numbers "
                    f"(e.g. row with value {bad[col].iloc[0]!r}). Remove currency symbols/commas/text."
                )

        for col in DATE_COLUMNS.get(sheet_name, []):
            parsed = pd.to_datetime(df[col], errors="coerce")
            bad = df[parsed.isna()]
            if not bad.empty:
                problems.append(
                    f"Sheet '{sheet_name}', column '{col}': {len(bad)} row(s) have an unrecognized date "
                    f"(e.g. {bad[col].iloc[0]!r}). Use YYYY-MM-DD, e.g. 2026-03-14."
                )

        book[sheet_name] = df

    if "Shipments" in book and "Supplier_Risk" in book and not problems:
        s_ids = set(book["Shipments"]["supplier_id"].astype(str))
        known_ids = set(book["Supplier_Risk"]["supplier_id"].astype(str))
        unknown = s_ids - known_ids
        if unknown:
            problems.append(
                f"Shipments sheet references supplier_id(s) not found in Supplier_Risk: "
                f"{', '.join(sorted(unknown))}. Add a row for them in Supplier_Risk, or fix the typo."
            )

    if "Shipments" in book and "Inventory" in book and not problems:
        ship_pairs = set(zip(book["Shipments"]["sku"].astype(str), book["Shipments"]["destination"].astype(str)))
        inv_pairs = set(zip(book["Inventory"]["sku"].astype(str), book["Inventory"]["destination"].astype(str)))
        missing_pairs = ship_pairs - inv_pairs
        if missing_pairs:
            sample = ", ".join(f"{sku}/{dest}" for sku, dest in list(missing_pairs)[:5])
            problems.append(
                f"{len(missing_pairs)} SKU+destination combination(s) used in Shipments have no matching "
                f"row in Inventory (e.g. {sample}). Inventory risk can't be computed for these rows — "
                f"add matching Inventory rows."
            )

    if problems:
        msg = "\n".join(f"  - {p}" for p in problems)
        raise FriendlyError(f"Found {len(problems)} problem(s) in '{xlsx_path.name}':\n{msg}")

    shipments = book["Shipments"][REQUIRED_SHEETS["Shipments"]].copy()
    inventory = book["Inventory"][REQUIRED_SHEETS["Inventory"]].copy()
    supplier_risk = book["Supplier_Risk"][REQUIRED_SHEETS["Supplier_Risk"]].copy()

    for col in DATE_COLUMNS["Shipments"]:
        shipments[col] = pd.to_datetime(shipments[col])
    for col in NUMERIC_COLUMNS["Shipments"]:
        shipments[col] = pd.to_numeric(shipments[col])
    for col in NUMERIC_COLUMNS["Inventory"]:
        inventory[col] = pd.to_numeric(inventory[col])
    for col in NUMERIC_COLUMNS["Supplier_Risk"]:
        supplier_risk[col] = pd.to_numeric(supplier_risk[col])

    return shipments, inventory, supplier_risk


# ----------------------------------------------------------------------------
# Step 2 — Risk engine (ported from risk_engine.py, DATA_DIR-free)
# ----------------------------------------------------------------------------
def clip(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def run_risk_engine(shipments, inventory, supplier_risk, today: date):
    supplier_risk_idx = supplier_risk.set_index("supplier_id")
    inv_lookup = {(r["sku"], r["destination"]): r for _, r in inventory.iterrows()}
    shipments_sorted = shipments.sort_values("po_date")
    supplier_hist = {}
    records, notifications = [], []

    for _, row in shipments_sorted.iterrows():
        lead_time = (row["supplier_handover_date"] - row["po_date"]).days
        handover_delay = lead_time - row["expected_handover_days"]
        handover_risk = clip((handover_delay / 10) * 100)

        wait_days = (row["vessel_departure_date"] - row["supplier_handover_date"]).days
        actual_sailing = (row["actual_arrival_date"] - row["vessel_departure_date"]).days
        sailing_delay = actual_sailing - row["planned_sailing_days"]
        red_flag = sailing_delay > 5
        sail_flag = "GREEN" if sailing_delay <= 2 else ("AMBER" if sailing_delay <= 5 else "RED")
        sailing_risk = clip((sailing_delay / 12) * 100)

        lead_time_total = (row["actual_arrival_date"] - row["po_date"]).days

        inv_row = inv_lookup.get((row["sku"], row["destination"]))
        current_inventory = inv_row["current_inventory"]
        daily_demand = max(1, inv_row["daily_demand"])
        safety_stock = inv_row["safety_stock"]
        days_of_supply = round(current_inventory / daily_demand, 2)
        arrival_date = row["actual_arrival_date"].date()
        remaining_lead_days = max(0, (arrival_date - today).days)
        demand_during_remaining = remaining_lead_days * daily_demand
        projected_before = current_inventory - demand_during_remaining
        projected_after = projected_before + row["quantity"]

        if projected_before <= 0:
            inv_status, inv_risk = "CRITICAL", 100
        elif projected_before < safety_stock:
            inv_status = "HIGH"
            inv_risk = clip(70 + (safety_stock - projected_before) / max(1, safety_stock) * 30)
        elif days_of_supply < 7:
            inv_status = "MEDIUM"
            inv_risk = clip(40 + (7 - days_of_supply) * 5)
        else:
            inv_status = "LOW"
            inv_risk = clip(20 - min(20, days_of_supply - 7))

        hist_list = supplier_hist.setdefault(row["supplier_id"], [])
        sample = hist_list + [sailing_delay]
        ltv_risk = 20.0 if len(sample) < 2 else round(clip((float(np.std(sample)) / 6) * 100), 1)
        hist_list.append(sailing_delay)

        hist_supplier_risk = (
            float(supplier_risk_idx.loc[row["supplier_id"], "historical_risk_score"])
            if row["supplier_id"] in supplier_risk_idx.index else 30.0
        )

        composite = round(clip(
            0.30 * sailing_risk + 0.25 * handover_risk + 0.20 * inv_risk
            + 0.15 * ltv_risk + 0.10 * hist_supplier_risk
        ), 1)
        category = "High" if composite >= 70 else ("Medium" if composite >= 40 else "Low")
        if inv_status == "CRITICAL":
            category = "Critical"

        record = {
            "po_id": row["po_id"], "supplier_id": row["supplier_id"], "supplier_name": row["supplier_name"],
            "sku": row["sku"], "sku_name": row["sku_name"], "destination": row["destination"],
            "quantity": int(row["quantity"]), "po_date": row["po_date"].date().isoformat(),
            "supplier_handover_date": row["supplier_handover_date"].date().isoformat(),
            "vessel_departure_date": row["vessel_departure_date"].date().isoformat(),
            "actual_arrival_date": row["actual_arrival_date"].date().isoformat(),
            "handover_lead_time_days": int(lead_time), "expected_handover_days": int(row["expected_handover_days"]),
            "supplier_delay_days": int(handover_delay), "port_wait_days": int(wait_days),
            "planned_sailing_days": int(row["planned_sailing_days"]), "actual_sailing_days": int(actual_sailing),
            "sailing_delay_days": int(sailing_delay), "sailing_flag": sail_flag, "red_flag": bool(red_flag),
            "total_lead_time_days": int(lead_time_total),
            "days_of_supply": days_of_supply, "remaining_lead_days": remaining_lead_days,
            "projected_inventory_before_arrival": round(projected_before),
            "projected_inventory_after_arrival": round(projected_after),
            "inventory_status": inv_status, "inventory_risk_score": round(inv_risk, 1),
            "risk_components": {
                "sailing_delay_risk": round(sailing_risk, 1), "supplier_delay_risk": round(handover_risk, 1),
                "inventory_risk": round(inv_risk, 1), "lead_time_variability_risk": ltv_risk,
                "historical_supplier_risk": round(hist_supplier_risk, 1),
            },
            "risk_score": composite, "risk_category": category,
            "recommended_action": recommended_action(category, inv_status),
        }
        records.append(record)
        if red_flag or category in ("High", "Critical"):
            notifications.append(build_notification(record))

    rank = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
    records = sorted(records, key=lambda r: (rank[r["risk_category"]], r["risk_score"]), reverse=True)

    exec_overview = build_executive_overview(records)
    supplier_perf = build_supplier_performance(records, supplier_risk)

    return {
        "generated_at": today.isoformat(),
        "executive_overview": exec_overview,
        "shipments": records,
        "inventory": inventory.to_dict(orient="records"),
        "supplier_performance": supplier_perf,
        "notifications": notifications,
    }


def recommended_action(category, inv_status):
    if inv_status == "CRITICAL":
        return [
            "Trigger emergency replenishment / air-freight top-up for the affected SKU.",
            "Contact supplier and shipping line simultaneously — do not wait for the standard escalation cycle.",
            "Notify the UK DC to reprioritise allocation of remaining stock.",
        ]
    if category == "High":
        return [
            "Contact supplier/logistics provider for a delay explanation.",
            "Review alternative transportation options (transhipment, expedited leg).",
            "Evaluate emergency replenishment for the linked SKU.",
        ]
    if category == "Medium":
        return [
            "Monitor shipment daily and confirm the next milestone date with the carrier.",
            "Flag to the logistics planner for awareness — no customer-facing action yet.",
        ]
    return ["Continue normal monitoring — no action required."]


def build_notification(record):
    lines = [
        f"Subject: {'RED FLAG' if record['red_flag'] else record['risk_category'].upper() + ' RISK'} — "
        f"{record['po_id']} Shipment {'Delay' if record['red_flag'] else 'Alert'}",
        "",
        f"Shipment {record['po_id']} has "
        + (f"exceeded the allowable sailing delay ({record['sailing_delay_days']} days late)."
           if record["red_flag"] else f"been classified {record['risk_category'].upper()} risk."),
        "", f"Supplier: {record['supplier_name']}", f"Destination: {record['destination']}",
        f"SKU: {record['sku']} ({record['sku_name']})", "",
        f"Planned Sailing Time: {record['planned_sailing_days']} days",
        f"Actual Sailing Time: {record['actual_sailing_days']} days",
        f"Delay: +{record['sailing_delay_days']} days", "",
        f"Shipment Risk Score: {record['risk_score']} ({record['risk_category'].upper()})", "",
        f"Days of Supply: {record['days_of_supply']}",
        f"Projected Inventory Before Arrival: {record['projected_inventory_before_arrival']} units",
        f"Inventory Risk: {record['inventory_status']}", "", "Recommended action:",
    ] + [f"{i+1}. {a}" for i, a in enumerate(record["recommended_action"])]
    return {"po_id": record["po_id"], "risk_category": record["risk_category"],
            "subject": lines[0].replace("Subject: ", ""), "body": "\n".join(lines)}


def build_executive_overview(records):
    total = len(records)
    avg_lead = round(float(np.mean([r["total_lead_time_days"] for r in records])), 1)
    avg_sail_delay = round(float(np.mean([r["sailing_delay_days"] for r in records])), 1)
    return {
        "total_active_shipments": total,
        "green": sum(1 for r in records if r["sailing_flag"] == "GREEN"),
        "amber": sum(1 for r in records if r["sailing_flag"] == "AMBER"),
        "red": sum(1 for r in records if r["sailing_flag"] == "RED"),
        "critical_inventory_risk": sum(1 for r in records if r["risk_category"] == "Critical"),
        "average_lead_time_days": avg_lead, "average_sailing_delay_days": avg_sail_delay,
        "high_risk_count": sum(1 for r in records if r["risk_category"] in ("High", "Critical")),
        "medium_risk_count": sum(1 for r in records if r["risk_category"] == "Medium"),
        "low_risk_count": sum(1 for r in records if r["risk_category"] == "Low"),
    }


def build_supplier_performance(records, supplier_risk_df):
    df = pd.DataFrame(records)
    perf = []
    for _, s in supplier_risk_df.iterrows():
        sub = df[df["supplier_id"] == s["supplier_id"]]
        if sub.empty:
            continue
        perf.append({
            "supplier_id": s["supplier_id"], "supplier_name": s["supplier_name"],
            "shipment_count": int(len(sub)), "avg_handover_days": round(sub["handover_lead_time_days"].mean(), 1),
            "avg_sailing_delay_days": round(sub["sailing_delay_days"].mean(), 1),
            "on_time_rate_pct": round((sub["sailing_flag"] == "GREEN").mean() * 100, 1),
            "historical_risk_score": float(s["historical_risk_score"]), "red_flag_count": int(sub["red_flag"].sum()),
        })
    return sorted(perf, key=lambda p: p["historical_risk_score"], reverse=True)


# ----------------------------------------------------------------------------
# Step 3 — Route ML layer (uses built-in route_history.csv + network.py)
# ----------------------------------------------------------------------------
def run_route_ml(processed):
    sys.path.insert(0, str(SCRIPT_DIR))
    from network import ROUTES, DESTINATION_PORT, PORTS, DC_COORDS, route_waypoints, congestion_for, CONGESTION_PROFILE
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
    from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    FEATURES = ["route_id", "month", "congestion_index", "supplier_reliability", "weight_class", "cost_index"]
    CAT_FEATURES = ["route_id", "weight_class"]
    NUM_FEATURES = ["month", "congestion_index", "supplier_reliability", "cost_index"]

    def build_pipeline(estimator):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                                  ("num", "passthrough", NUM_FEATURES)])
        return Pipeline([("pre", pre), ("model", estimator)])

    hist = pd.read_csv(SCRIPT_DIR / "route_history.csv")
    X, y_reg, y_clf = hist[FEATURES], hist["actual_transit_days"], hist["delay_flag"]

    reg_pipe = build_pipeline(GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42))
    clf_pipe = build_pipeline(RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_oof = cross_val_predict(reg_pipe, X, y_reg, cv=kf)
    mae = mean_absolute_error(y_reg, reg_oof)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_oof_proba = cross_val_predict(clf_pipe, X, y_clf, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_clf, clf_oof_proba)
    acc = accuracy_score(y_clf, (clf_oof_proba >= 0.5).astype(int))

    reg_pipe.fit(X, y_reg)
    clf_pipe.fit(X, y_clf)

    ohe = reg_pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    all_names = cat_names + NUM_FEATURES
    importances = reg_pipe.named_steps["model"].feature_importances_
    fi = sorted(zip(all_names, importances), key=lambda t: t[1], reverse=True)[:8]
    metadata = {
        "regressor": "GradientBoostingRegressor", "classifier": "RandomForestClassifier",
        "training_rows": len(hist), "cv_folds": 5, "mae_days": round(float(mae), 2),
        "delay_classifier_auc": round(float(auc), 3), "delay_classifier_accuracy": round(float(acc), 3),
        "feature_importance": [{"feature": n, "importance": round(float(v), 4)} for n, v in fi],
    }

    supplier_reliability_lookup = {s["supplier_id"]: s["on_time_rate_pct"] / 100.0 for s in processed["supplier_performance"]}
    all_rows, index_map = [], []
    for ship in processed["shipments"]:
        po_month = pd.Timestamp(ship["po_date"]).month
        supplier_rel = supplier_reliability_lookup.get(ship["supplier_id"], 0.7)
        weight_class = "light" if ship["quantity"] < 900 else ("heavy" if ship["quantity"] > 1600 else "standard")
        for route_id, route in ROUTES.items():
            cong = congestion_for(route_id, po_month)
            all_rows.append({"route_id": route_id, "month": po_month, "congestion_index": cong,
                              "supplier_reliability": supplier_rel, "weight_class": weight_class,
                              "cost_index": route["cost_index"]})
            index_map.append((ship["po_id"], route_id))

    cand_df = pd.DataFrame(all_rows)
    pred_days = reg_pipe.predict(cand_df[FEATURES])
    pred_proba = clf_pipe.predict_proba(cand_df[FEATURES])[:, 1]

    per_ship = {}
    for (po_id, route_id), days, proba, row in zip(index_map, pred_days, pred_proba, all_rows):
        route = ROUTES[route_id]
        per_ship.setdefault(po_id, []).append({
            "route_id": route_id, "route_name": route["name"], "blurb": route["blurb"],
            "base_transit_days": route["base_transit_days"], "predicted_transit_days": round(float(days), 1),
            "predicted_delay_probability": round(float(proba), 3), "cost_index": route["cost_index"],
            "congestion_index": round(row["congestion_index"], 2),
        })

    for po_id, candidates in per_ship.items():
        t_vals = [c["predicted_transit_days"] for c in candidates]
        p_vals = [c["predicted_delay_probability"] for c in candidates]
        c_vals = [c["cost_index"] for c in candidates]

        def norm(v, lo, hi):
            return 0.0 if hi == lo else (v - lo) / (hi - lo)

        for c in candidates:
            c["norm_time"] = round(norm(c["predicted_transit_days"], min(t_vals), max(t_vals)), 3)
            c["norm_risk"] = round(norm(c["predicted_delay_probability"], min(p_vals), max(p_vals)), 3)
            c["norm_cost"] = round(norm(c["cost_index"], min(c_vals), max(c_vals)), 3)
            c["default_score"] = round(0.5 * c["norm_time"] + 0.3 * c["norm_risk"] + 0.2 * c["norm_cost"], 4)
        candidates.sort(key=lambda c: c["default_score"])
        for i, c in enumerate(candidates):
            c["recommended"] = (i == 0)

    dest_lookup = {s["po_id"]: s["destination"] for s in processed["shipments"]}
    for po_id, candidates in per_ship.items():
        dest = dest_lookup[po_id]
        for c in candidates:
            c["waypoints"] = route_waypoints(c["route_id"], dest)

    network_geo = {
        "ports": PORTS, "destination_port": DESTINATION_PORT, "dc_coords": DC_COORDS,
        "routes": {rid: {"name": r["name"], "blurb": r["blurb"], "base_transit_days": r["base_transit_days"],
                          "cost_index": r["cost_index"], "congestion_hub": r["congestion_hub"]}
                   for rid, r in ROUTES.items()},
        "congestion_profile": CONGESTION_PROFILE,
    }

    processed["route_optimization"] = per_ship
    processed["ml_metadata"] = metadata
    processed["network"] = network_geo
    return processed


# ----------------------------------------------------------------------------
# Step 4 — Inject into dashboard.html
# ----------------------------------------------------------------------------
def refresh_dashboard_html(processed, template_path: Path, out_path: Path):
    html = template_path.read_text(encoding="utf-8")
    new_json = json.dumps(processed, indent=2, default=str)

    pattern = re.compile(
        r'(<script id="app-data" type="application/json">\s*)(.*?)(\s*</script>)', re.S
    )
    if not pattern.search(html):
        raise FriendlyError(f"Could not find the embedded data block in {template_path.name}; "
                             f"it may not be a Voyage Control dashboard template.")
    html = pattern.sub(lambda m: m.group(1) + new_json + m.group(3), html, count=1)

    n = processed["executive_overview"]["total_active_shipments"]
    html = re.sub(
        r"DEMO DATA — \d+ SYNTHETIC SHIPMENTS, [A-Z\u2013\-–]+ \d{4}",
        f"LIVE DATA — {n} SHIPMENTS, REFRESHED {processed['generated_at']}",
        html,
    )
    out_path.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Refresh the Voyage Control dashboard from an Excel data file.")
    parser.add_argument("excel_file", help="Path to your filled-in Data_Template.xlsx")
    parser.add_argument("--as-of", default=None, help="Evaluation date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="dashboard.html", help="Output HTML filename (default: dashboard.html)")
    parser.add_argument("--template", default=str(SCRIPT_DIR / "dashboard.html"),
                         help="Path to the base dashboard.html template")
    args = parser.parse_args()

    xlsx_path = Path(args.excel_file)
    if not xlsx_path.exists():
        print(f"File not found: {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    today = date.fromisoformat(args.as_of) if args.as_of else date.today()

    try:
        print(f"Reading {xlsx_path.name} ...")
        shipments, inventory, supplier_risk = load_and_validate(xlsx_path)
        print(f"  {len(shipments)} shipments, {len(inventory)} inventory rows, {len(supplier_risk)} suppliers.")

        print("Running risk engine ...")
        processed = run_risk_engine(shipments, inventory, supplier_risk, today)

        print("Training route-optimization ML models and scoring routes ...")
        processed = run_route_ml(processed)

        print(f"Refreshing dashboard ...")
        refresh_dashboard_html(processed, Path(args.template), Path(args.out))

        print(f"\nDone. Open {args.out} in your browser to view the refreshed dashboard.")
    except FriendlyError as e:
        print(f"\nCan't build the dashboard yet:\n\n{e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
