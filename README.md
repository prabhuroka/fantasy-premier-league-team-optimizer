# FPL Team Optimizer

A comprehensive data science platform that automates optimal Fantasy Premier League team selection using machine learning predictions and mathematical optimization algorithms.

---

## 🚀 Project Overview

This system combines machine learning and mathematical optimization to build optimal FPL squads through a multi-phase pipeline:

1. **Data Pipeline:** Automated data collection and processing  
2. **Feature Engineering:** 189 comprehensive features per player  
3. **ML Prediction:** Position-specific LightGBM models for RAW point prediction  
4. **Team Optimization:** Mixed Integer Programming for squad selection  

---

## 📊 Key Results

- 21% more accurate than individual predictions  
- +30–36 point improvements over current teams  
- 77% success rate across gameweeks  
- <30 second optimization time  

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9+  
- Git  
- 4GB RAM minimum  

### 1. Clone the Repository


git clone https://github.com/yourusername/fpl-team-optimizer.git

cd fpl-team-optimizer

### 2. Create Virtual Environment

Windows
python -m venv venv
venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate


### 3. Install Dependencies

pip install -r requirements.txt


🚀 Quick Start Guide
Run Individual Phases
###  Phase 1: Data Pipeline

#Download and process data
python data_pipeline/run_pipeline.py


### Phase 2: Feature Engineering

#Generate comprehensive features
python feature_engineering/run_feature_engineering.py


### Phase 3: ML Prediction

#Generate RAW point predictions
python ml_model/run_raw_point_pipeline.py


### Phase 4: Team Optimization

#Optimize FPL team
python optimization/run_optimization.py


### Phase 5: Terminal UI

#Optimize FPL team from terminal Ui (For user to have better experience for optimizing for single & multiple gameweeks and custom team optimization)
python fpl_optimizer_tui/run_tui.py



📁 Project Structure

fpl-simulator/
├── data_pipeline/          Data collection
├── feature_engineering/    Feature generation (189 features)
├── ml_model/               ML predictions (RAW points)
├── optimization/           Team optimization (MIP)
├── data/                   # All data storage
│   ├── raw/                # Raw data extracted
│   ├── processed/          # SQLite databases
│   ├── features/           # Generated feature CSVs
│   ├── models/             # Trained ML models
│   ├── predictions/        # Weekly predictions
│   └── optimization/       # Optimization results
└── requirements.txt        # Python dependencies


📊 Output Files

Predictions

data/predictions/
├── raw_points_predictions_gw19.csv
├── latest_raw_points_predictions_gw19.csv
└── raw_points_with_value_gw19.csv

Optimization Results

data/optimization/
├── complete_optimization_gw19.json
├── optimized_team_gw19.csv
└── transfer_recommendations_gw19.json


For Faster Optimization

# Use heuristic method (faster, less optimal)
python optimization/run_optimization.py --method heuristic

# Limit player pool
python optimization/run_optimization.py --top-players 300
For Better Accuracy


# Use MIP with longer timeout
python optimization/run_optimization.py --method mip --timeout 120

# Include more historical data
python feature_engineering/run_feature_engineering.py --gameweeks 10


🤝 Contributing
Fork the repository

Create a feature branch

Make your changes

Run tests

Submit a pull request


Key Concepts:
Mixed Integer Programming: Mathematical optimization for squad selection

Feature Engineering: 189 features capturing player performance

Walk-Forward Validation: Time-series testing approach

Separated Architecture: RAW prediction + value optimization

This project relies on the excellent FPL-Elo-Insights dataset for historical FPL data. Special thanks to olbauday and team (https://github.com/olbauday/FPL-Core-Insights) for maintaining this comprehensive resource that makes data-driven FPL analysis possible.

📄 License
MIT License – see LICENSE file for details.

