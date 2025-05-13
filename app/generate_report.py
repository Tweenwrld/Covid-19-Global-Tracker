from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime

# Define PDF file name
pdf_file = "COVID19_Analytics_Dashboard_Report.pdf"

# Initialize PDF document
doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    name='TitleStyle', parent=styles['Title'], fontSize=24, spaceAfter=20, alignment=1, textColor=colors.navy
)
subtitle_style = ParagraphStyle(
    name='SubtitleStyle', parent=styles['Title'], fontSize=14, spaceAfter=20, alignment=1
)
section_style = ParagraphStyle(
    name='SectionStyle', parent=styles['Heading1'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.darkblue
)
subsection_style = ParagraphStyle(
    name='SubsectionStyle', parent=styles['Heading2'], fontSize=12, spaceBefore=15, spaceAfter=8
)
body_style = ParagraphStyle(
    name='BodyStyle', parent=styles['BodyText'], fontSize=10, spaceAfter=8, leading=12
)
bullet_style = ParagraphStyle(
    name='BulletStyle', parent=styles['BodyText'], fontSize=10, spaceAfter=6, leftIndent=20, bulletIndent=10, firstLineIndent=-10
)

# Content elements
elements = []

# Title Page
elements.append(Paragraph("COVID-19 Advanced Analytics Dashboard: Comprehensive Report", title_style))
elements.append(Paragraph("In-Depth Analysis of Global Pandemic Data", subtitle_style))
elements.append(Paragraph(f"Date: May 13, 2025", body_style))
elements.append(Paragraph("Author: Leonard Boma", body_style))
elements.append(Spacer(1, 0.5 * inch))

# Table of Contents
elements.append(Paragraph("Table of Contents", section_style))
toc = [
    ["1. Introduction", "2"],
    ["2. Data Description", "3"],
    ["3. Tools and Technologies", "4"],
    ["4. Methodology", "5"],
    ["5. Comprehensive Insights", "6"],
    ["6. Visualizations", "9"],
    ["7. Key Findings", "10"],
    ["8. Implementation Details", "12"],
    ["9. Challenges and Solutions", "14"],
    ["10. Conclusion", "15"],
    ["11. References", "16"]
]
toc_table = Table(toc, colWidths=[4.5 * inch, 1 * inch])
toc_table.setStyle(TableStyle([
    ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
]))
elements.append(toc_table)
elements.append(Spacer(1, 0.5 * inch))

# Introduction
elements.append(Paragraph("1. Introduction", section_style))
elements.append(Paragraph(
    "The COVID-19 Advanced Analytics Dashboard is a comprehensive tool designed to analyze global pandemic data, "
    "providing insights into trends, statistical patterns, geographic distributions, comparative impacts, and future forecasts. "
    "Developed in both Streamlit (interactive web app) and Jupyter Notebook (static analysis) environments, the project "
    "leverages data from Our World in Data to deliver actionable insights for researchers, policymakers, and the public. "
    "The dashboard covers case/death trends, vaccination impacts, policy effects, and more, using advanced statistical and "
    "visualization techniques.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Data Description
elements.append(Paragraph("2. Data Description", section_style))
elements.append(Paragraph(
    "The dataset is sourced from Our World in Data (https://covid.ourworldindata.org/data/owid-covid-data.csv), "
    "covering daily COVID-19 metrics for countries and regions from January 2020 to the present. Key columns include:",
    body_style
))
data_columns = [
    "- new_cases, total_cases: Daily and cumulative confirmed cases.",
    "- new_deaths, total_deaths: Daily and cumulative deaths.",
    "- people_vaccinated_per_hundred: Percentage of population vaccinated.",
    "- stringency_index: Government response stringency (0-100).",
    "- population, gdp_per_capita, median_age: Demographic and economic indicators."
]
for item in data_columns:
    elements.append(Paragraph(item, bullet_style))
elements.append(Paragraph(
    "Preprocessing steps included filling missing numeric values with 0, calculating 7/14-day rolling averages, "
    "computing per capita metrics (e.g., cases per million), and adding derived features like growth rates and wave indicators.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Tools and Technologies
elements.append(Paragraph("3. Tools and Technologies", section_style))
elements.append(Paragraph("The project was developed using the following tools:", body_style))
tools = [
    "- Python 3.10: Core programming language.",
    "- Streamlit: For the interactive web-based dashboard. I have the Jupyter Notebook version as well, but prefer Streamlit for its interactivity.",
    "- Pandas: Data manipulation and preprocessing.",
    "- NumPy: Numerical computations.",
    "- Plotly: Interactive visualizations (line charts, choropleths, etc.).",
    "- Statsmodels: Time series analysis and forecasting (ARIMA).",
    "- SciPy: Statistical tests (e.g., normality, correlation).",
    "- Scikit-learn: Clustering and data scaling."
]
for item in tools:
    elements.append(Paragraph(item, bullet_style))
elements.append(Paragraph(
    "The Streamlit app runs in a browser, while the Jupyter Notebook version supports static analysis with inline visualizations.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Methodology
elements.append(Paragraph("4. Methodology", section_style))
elements.append(Paragraph("4.1 Data Loading and Preprocessing", subsection_style))
elements.append(Paragraph(
    "Data is loaded from a CSV URL, with error handling for connectivity issues. Preprocessing includes handling missing data, "
    "calculating rolling averages, per capita metrics, growth rates, and wave indicators.",
    body_style
))
elements.append(Paragraph("4.2 Analytical Approaches", subsection_style))
elements.append(Paragraph(
    "The dashboard implements multiple analyses: wave detection (peak-based algorithm), statistical analysis (mean, skewness, decomposition), "
    "geographic mapping (choropleths), comparative analysis (correlations, vaccination/policy impacts), and forecasting (ARIMA).",
    body_style
))
elements.append(Paragraph("4.3 Visualization Techniques", subsection_style))
elements.append(Paragraph(
    "Plotly is used for interactive visualizations, including line charts for trends, choropleth maps for geographic data, "
    "bar charts for comparisons, scatter plots for clustering, and decomposition plots for time series analysis.",
    body_style
))
elements.append(Paragraph("4.4 User Interface", subsection_style))
elements.append(Paragraph(
    "The Streamlit app features interactive filters (country, date, metric, continent), tabs for different analyses, "
    "and downloadable data. The Jupyter version uses hardcoded parameters for static rendering. Also one is able to download filtered data based on the selected parameters, e.g, I downloaded filtered data of South Africa and Algeria.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Comprehensive Insights
elements.append(Paragraph("5. Comprehensive Insights", section_style))
elements.append(Paragraph("5.1 Trends & Insights", subsection_style))
elements.append(Paragraph(
    "Analyzes case/death trends over time, detecting waves (e.g., peaks in cases), weekend reporting effects, and data quality issues. "
    "Example: Identified lower weekend reporting in some countries (e.g., -15% in the US).",
    body_style
))
elements.append(Paragraph("5.2 Statistical Analysis", subsection_style))
elements.append(Paragraph(
    "Computes statistical metrics (mean, median, skewness, kurtosis) and performs time series decomposition. "
    "Example: Strong 7-day seasonality in case data for India.",
    body_style
))
elements.append(Paragraph("5.3 Geographic Analysis", subsection_style))
elements.append(Paragraph(
    "Maps global case/death rates and compares regional patterns. Example: Europe had higher deaths per million than Africa.",
    body_style
))
elements.append(Paragraph("5.4 Comparative Analysis", subsection_style))
elements.append(Paragraph(
    "Examines correlations (e.g., population density vs. cases), vaccination impacts (e.g., case reductions post-20% vaccination), "
    "and policy effects (e.g., high stringency reduced cases). Example: Significant negative correlation between hospital beds and fatality rates.",
    body_style
))
elements.append(Paragraph("5.5 Forecasting & Modeling", subsection_style))
elements.append(Paragraph(
    "Uses ARIMA for 30-day forecasts and KMeans for country clustering. Example: Forecasted stable case trends for Brazil; "
    "clustered countries by case/death rates.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Visualizations
elements.append(Paragraph("6. Visualizations", section_style))
elements.append(Paragraph(
    "The dashboard includes the following visualizations (images can be exported from the app and inserted here):",
    body_style
))
visualizations = [
    "- Line Charts: Trends of cases/deaths over time, with significant event annotations.",
    "- Choropleth Maps: Global distribution of cases/deaths per million.",
    "- Bar Charts: Regional comparisons, vaccination/policy impacts.",
    "- Scatter Plots: Country clusters based on pandemic metrics.",
    "- Decomposition Plots: Trend, seasonal, and residual components of time series."
]
for item in visualizations:
    elements.append(Paragraph(item, bullet_style))
elements.append(Paragraph(
    "Plots/Case Growth Rate (%) per Million People Analysis.png",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Key Findings
elements.append(Paragraph("7. Key Findings", section_style))
elements.append(Paragraph(
    "The project yielded significant insights for selected countries (e.g., United States, United Kingdom, India, Brazil, South Africa):",
    body_style
))
findings = [
    "- United States: Multiple waves detected, with peaks in 2020 and 2022; significant weekend reporting drop (-12%).",
    "- India: Strong seasonality in cases; vaccination reduced cases by ~20% after 20% threshold.",
    "- Brazil: Stable forecasted case trends for 2023; high stringency policies lowered cases.",
    "- Europe vs. Africa: Europe had 10x higher deaths per million due to demographic and healthcare factors.",
    "- Clustering: Countries grouped into 4 clusters based on case/death rates, with distinct profiles (e.g., high-income vs. low-income)."
]
for item in findings:
    elements.append(Paragraph(item, bullet_style))
elements.append(Spacer(1, 0.2 * inch))

# Implementation Details
elements.append(Paragraph("8. Implementation Details", section_style))
elements.append(Paragraph("8.1 Streamlit App", subsection_style))
elements.append(Paragraph(
    "Features interactive filters (country, date, metric, continent), 5 tabs (Trends, Statistics, Geographic, Comparative, Forecasting), "
    "custom CSS styling, and CSV download. Runs via `streamlit run covid_dashboard.py`.",
    body_style
))
elements.append(Paragraph("8.2 Jupyter Notebook", subsection_style))
elements.append(Paragraph(
    "Adapted for static analysis with inline Plotly visualizations, hardcoded parameters, and CSV output. "
    "Uses `IPython.display` for rich outputs.",
    body_style
))
elements.append(Paragraph("8.3 Requirements", subsection_style))
elements.append(Paragraph(
    "Dependencies (requirements.txt): streamlit>=1.38.0, pandas>=2.2.2, numpy>=1.26.4, plotly>=5.22.0, "
    "statsmodels>=0.14.2, scipy>=1.13.1, scikit-learn>=1.5.0.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# Challenges and Solutions
elements.append(Paragraph("9. Challenges and Solutions", section_style))
challenges = [
    "- Challenge: Missing data in some countries. Solution: Filled numeric missing values with 0; skipped analyses for incomplete data.",
    "- Challenge: Streamlit visualizations not rendering in Jupyter. Solution: Adapted code to use `fig.show()` for inline rendering.",
    "- Challenge: Variable data reporting quality. Solution: Added anomaly detection and missing date checks.",
    "- Challenge: Forecasting accuracy. Solution: Used ARIMA with confidence intervals; limited forecast horizon to 30 days."
]
for item in challenges:
    elements.append(Paragraph(item, bullet_style))
elements.append(Spacer(1, 0.2 * inch))

# Conclusion
elements.append(Paragraph("10. Conclusion", section_style))
elements.append(Paragraph(
    "The COVID-19 Advanced Analytics Dashboard successfully provides deep insights into global pandemic trends, "
    "statistical patterns, geographic disparities, and intervention impacts. The dual implementation (Streamlit and Jupyter) "
    "ensures accessibility for both interactive and static use cases. Future enhancements could include real-time data integration, "
    "machine learning models, and variant-specific analyses.",
    body_style
))
elements.append(Spacer(1, 0.2 * inch))

# References
elements.append(Paragraph("11. References", section_style))
references = [
    "- Our World in Data COVID-19 Dataset: https://covid.ourworldindata.org/data/owid-covid-data.csv",
    "- Streamlit Documentation: https://docs.streamlit.io/",
    "- Plotly Documentation: https://plotly.com/python/",
    "- Statsmodels Documentation: https://www.statsmodels.org/stable/",
    "- SciPy Documentation: https://docs.scipy.org/doc/scipy/",
    "- Scikit-learn Documentation: https://scikit-learn.org/stable/"
]
for item in references:
    elements.append(Paragraph(item, bullet_style))

# Build PDF
doc.build(elements)
print(f"PDF report generated: {pdf_file}")