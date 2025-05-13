# COVID-19 Advanced Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Updated](https://img.shields.io/badge/Last%20Updated-May%202025-orange)

## 📋 Overview

An interactive analytics platform for global COVID-19 data analysis, providing deep insights into pandemic trends, statistical patterns, and forecasts. Available as both a Streamlit web application and Jupyter Notebook analysis.

### Key Features

- 📈 Real-time trend analysis across countries and regions
- 🔍 Advanced statistical analysis with time series decomposition
- 🗺️ Geographic visualization with interactive maps
- 💉 Vaccination impact assessment
- 🔮 Predictive modeling and country clustering

## 🧰 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.8+, Streamlit 1.38.0+ |
| **Data Processing** | Pandas 2.2.2+, NumPy 1.26.4+ |
| **Visualization** | Plotly 5.22.0+ |
| **Analysis** | Statsmodels 0.14.2+, SciPy 1.13.1+, Scikit-learn 1.5.0+ |
| **Development** | Jupyter Notebook |

## 📊 Data Source

- **Provider**: Our World in Data COVID-19 dataset
- **URL**: [https://covid.ourworldindata.org/data/owid-covid-data.csv](https://covid.ourworldindata.org/data/owid-covid-data.csv)
- **Format**: CSV (daily updates)
- **Coverage**: Global data from January 2020 to present

### Key Metrics

- Cases: new/total cases, 7-day averages, growth rates
- Deaths: new/total deaths, 7-day averages
- Vaccinations: first/full vaccination percentages
- Healthcare: ICU/hospital patients, testing rates
- Demographics: population, GDP, median age, density
- Policy: government response stringency index (0-100)

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Tweenwrld/Covid-19-Global-Tracker.git
cd COVID-19-GLOBAL-TRACKER

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and use the interactive filters to explore the data.
[Live demo of the project](https://tweenwrld-covid-19-global-tracker-appapp-qrgjwe.streamlit.app/)

### Jupyter Notebook ()

```bash
jupyter notebook
```

Open `COVID-19_Analysis.ipynb` to run the static analysis version.

## 📌 Features

### Interactive Dashboard (Streamlit)

- **Data Exploration**: Filter by countries, dates, metrics, regions
- **Analysis Tabs**: Trends, Statistics, Geography, Comparisons, Forecasting
- **Visualizations**: Line charts, choropleths, bar charts, scatter plots
- **Data Export**: Download filtered data as CSV

### Analysis Notebook (Jupyter)

- In-depth statistical analysis with inline visualizations
- Detailed methodology explanations
- Static exports for reporting

## 🔬 Methodology

- **Wave Detection**: Peak-based algorithm for pandemic wave identification
- **Statistical Analysis**: Time series decomposition, anomaly detection
- **Geographic Analysis**: Continental aggregation and global mapping
- **Comparative Analysis**: Correlation studies and impact assessment
- **Forecasting**: ARIMA models with 30-day projections
- **Clustering**: K-Means algorithm for country profile grouping

## 🌟 Key Insights

- Multiple distinct pandemic waves across countries (3-5 in major regions)
- Strong correlation between demographics and case fatality rates
- Approximately 20% case reduction following vaccination thresholds
- Weekend reporting effects in many countries (~15% drop)
- Four distinct country clusters based on pandemic metrics

## 📁 Project Structure

```
COVID-19-GLOBAL-TRACKER/
├── app.py                    # Streamlit app
├── COVID-19_Analysis.ipynb      # Jupyter notebook
├── requirements.txt          # Dependencies
├── covid_filtered_data.csv   # Generated output
└── README.md                 # Documentation
```

## 🔄 Future Enhancements

- Real-time data integration
- Variant-specific analysis
- Enhanced machine learning models
- Mobile-responsive UI optimizations

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Our World in Data](https://ourworldindata.org/coronavirus)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## 📞 Contact

For questions or support, please [open an issue](https://github.com/Tweenwrld/Covid-19-Global-Tracker/issues) on GitHub.