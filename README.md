# COVID-19 Global Tracker

A comprehensive COVID-19 data visualization and analysis project that includes both a Jupyter notebook for in-depth analysis and a Streamlit web application for interactive exploration.

## Overview

This project provides tools to analyze and visualize COVID-19 data from the "Our World in Data" dataset, allowing users to:

- Track cases, deaths, and vaccinations worldwide
- Compare metrics across multiple countries
- Visualize global trends using maps and charts
- Generate custom reports and insights

## Project Structure

```
covid-19-global-tracker/
├── data/
│   └── owid-covid-data.csv     # COVID-19 dataset (download manually)
├── notebooks/
│   └── COVID-19_Analysis.ipynb # Analysis notebook
├── app/
│   ├── app.py                  # Streamlit dashboard
│   └── requirements.txt        # Dependencies
├── outputs/
│   ├── charts/                 # Saved visualizations
│   └── report.pdf              # Generated report
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for analysis)
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/covid-19-global-tracker.git
   cd covid-19-global-tracker
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r app/requirements.txt
   ```

3. Download the dataset:
   - Visit [Our World in Data COVID-19 GitHub repository](https://github.com/owid/covid-19-data/tree/master/public/data)
   - Download the `owid-covid-data.csv` file
   - Place it in the `data/` directory

### Running the Jupyter Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `notebooks/COVID-19_Analysis.ipynb`

3. Run the cells to perform data analysis and generate visualizations

### Running the Streamlit App

1. Navigate to the app directory:
   ```
   cd app
   ```

2. Launch the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501`

## Features

### Jupyter Notebook Analysis

- Comprehensive data preprocessing and cleaning
- Time series analysis of cases and deaths
- Vaccination rate comparison
- Global choropleth maps
- Advanced statistical analysis
- Export of visualizations for reporting

### Streamlit Web Application

- Interactive filtering by country and date range
- Real-time visualization of COVID-19 metrics
- Multi-country comparison tools
- Global map view with selectable metrics
- Key statistics and insights
- Mobile-friendly responsive design

## Data Source

This project uses the "Our World in Data" COVID-19 dataset, which is updated daily and includes data on:

- Confirmed cases and deaths
- Testing data
- Vaccination progress
- Hospital & ICU admissions
- Population demographics
- And more

Data source: [Our World in Data COVID-19 Dataset](https://github.com/owid/covid-19-data)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.