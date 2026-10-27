# CivicPulse Pro — Smart City Incident Intelligence

A multipage-style Streamlit dashboard for exploring synthetic civic incidents. Includes a geographic hotspot map, executive KPIs, area/type comparisons, correlation analysis, drill-downs, advanced filters, Random Forest forecasting and delayed-resolution risk scoring, Isolation Forest anomaly detection, resolution performance, and downloadable PDF/Excel/CSV/HTML reports.

> **Data disclaimer:** All incident records, costs, ratings, and map points are synthetic. Coordinates are illustrative clusters around a demonstration city and do not represent real incidents. Machine-learning outputs are educational demonstrations, not validated operational predictions.

## Run in PyCharm (Windows)
1. Extract the ZIP and open the `CivicPulse_Smart_City_Analytics_Pro` folder in PyCharm.
2. Open the PyCharm terminal at the project root.
3. (Recommended) create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```powershell
   python -m streamlit run app.py
   ```
6. Open the local URL shown in the terminal (usually http://localhost:8501).

## Pages
- **Overview:** executive summary, key insights, KPIs, trend and hotspot overview.
- **Analytics:** area-vs-area and type-vs-type trends, severity trend, correlation heatmap, area drill-down and cost breakdown.
- **Incident Map:** Plotly Mapbox-style map with severity/type filters and hotspot ranking. Uses OpenStreetMap tiles without requiring an access token.
- **Predictions:** Random Forest 30-day daily incident forecast, time-ordered holdout MAE, and interactive delayed-resolution risk score.
- **Performance:** median resolution time, ≤24-hour SLA proxy, area/category rankings, and Isolation Forest daily-volume anomalies.
- **Reports:** downloadable PDF summary, multi-sheet Excel workbook, CSV trend, and interactive HTML chart.

## Dataset columns
The CSV includes Incident_ID, Reported_At, Area, Incident_Type, Severity, Report_Source, Status, Resolution_Hours, Citizen_Rating, Estimated_Cost_INR, Latitude, and Longitude.

## GitHub description
CivicPulse Pro is a multipage smart-city incident intelligence dashboard built with Python, Streamlit, Pandas, Plotly and Scikit-learn. Includes geographic hotspots, Random Forest forecasting, anomaly detection, risk scoring, service performance analytics, and PDF/Excel exports using synthetic demo data.

## LinkedIn post draft
🏙️ **Upgraded CivicPulse into a Smart City Incident Intelligence dashboard!**

New features include a geographic incident map, executive KPI cards, area and incident-type trend comparisons, correlation heatmap, drill-down analysis, advanced filters, Random Forest forecasting, Isolation Forest anomaly detection, a delayed-resolution risk score, performance metrics, and downloadable PDF/Excel reports.

Tech stack: Python • Pandas • Plotly • Streamlit • Scikit-learn • OpenPyXL • ReportLab

This portfolio prototype uses synthetic demonstration data and illustrative map coordinates.

#Python #MachineLearning #DataAnalytics #Streamlit #DataVisualization #PortfolioProject
