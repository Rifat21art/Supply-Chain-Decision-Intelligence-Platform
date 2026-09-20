# Voyage Control — A Supply Chain Risk & Route Intelligence Dashboard

A self-contained, single-file dashboard that scores inbound shipment risk, predicts
route delays with a cross-validated ML model, and optimizes freight routing with a
self-service layer so **anyone** can drop in their own Excel data and get a refreshed
dashboard, no coding required.

**[▶ Live demo](#)** — replace with your GitHub Pages link once published (see below)


## What it does

- **Executive Overview** — fleet-wide shipment health at a glance (on-time %, active risk counts)
- **Shipment Monitor** — every PO with a composite risk score, drill-down risk breakdown, and a recommended action
- **Inventory Risk** — projects stock-outs per SKU/DC ahead of arrival, factoring in daily demand and safety stock
- **Supplier Performance** — historical reliability and delay variance per supplier
- **Route Network** — live-rendered shipping lanes with seasonal congestion overlays
- **Route Optimizer** — a gradient-boosted regressor + random-forest classifier (cross-validated) score every candidate lane for transit time and delay probability, then rank by a configurable speed/reliability/cost trade-off

## Two ways to put your own data in

| | Browser upload | Full pipeline |
|---|---|---|
| **How** | Click "Download Template" then "Upload Data" inside `dashboard.html` | `python build_dashboard.py YourFile.xlsx` |
| **Setup** | None — just a browser | `pip install -r requirements.txt` |
| **Route Optimizer** | Quick heuristic (clearly labeled) | Full cross-validated ML model |

The Excel template (4 tabs: Instructions, Shipments, Inventory, Supplier_Risk) is
downloadable straight from the dashboard itself — no need to dig through the repo.

Full details, the exact Excel schema, and troubleshooting: **[USER_MANUAL.md](USER_MANUAL.md)**

## Test data

Two synthetic (fake but realistic) datasets are included for trying the pipeline or
recording a demo without real company numbers:
- `Synthetic_Test_Data_1_NormalQuarter.xlsx` — mostly healthy, believable delay tail
- `Synthetic_Test_Data_2_DisruptedQuarter.xlsx` — elevated disruption, exercises the dashboard's higher-risk states

## Quick start

```bash
git clone https://github.com/<your-username>/voyage-control.git
cd voyage-control
pip install -r requirements.txt

# Try it on the built-in synthetic test data
python build_dashboard.py Synthetic_Test_Data_1_NormalQuarter.xlsx
open dashboard.html   # or just double-click it
```

## Stack

Python (pandas, scikit-learn, openpyxl) for the data/ML pipeline · vanilla HTML/CSS/JS
+ Leaflet for the dashboard · SheetJS for in-browser Excel parsing. No backend, no
database — the whole thing is one portable HTML file.

## Honest limitations

- `route_history.csv` (used to train the delay-prediction model) is synthetically
  generated from hand-authored seasonal congestion profiles, not real carrier data —
  this demonstrates the *technique* (cross-validated GBR/RF on seasonal + reliability
  features), not a production-calibrated model.
- The in-browser upload path can't retrain the ML model on the fly, so it falls back to
  a transparent heuristic for route scoring — labeled as such wherever it appears.

## License

MIT — see [LICENSE](LICENSE).
