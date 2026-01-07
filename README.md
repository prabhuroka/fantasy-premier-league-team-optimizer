# FPL Team Optimizer

A comprehensive data science platform that automates optimal Fantasy Premier League team selection using machine learning predictions and mathematical optimization algorithms.

---

## Project Overview

This system combines machine learning and mathematical optimization to build optimal FPL squads through a multi-phase pipeline:

1. Data Pipeline: Automated data collection and processing  
2. Feature Engineering: 189 comprehensive features per player  
3. ML Prediction: Position-specific LightGBM models for RAW point prediction  
4. Team Optimization: Mixed Integer Programming for squad selection  

---

## Key Results

- 21% more accurate than individual predictions  
- +30-36 point improvements over current teams  
- 77% success rate across gameweeks  
- <30 second optimization time  

---

## Installation & Setup

### Prerequisites

- Python 3.9+  
- Git  
- 4GB RAM minimum  

---

### 1. Clone the Repository

git clone https://github.com/yourusername/fpl-team-optimizer.git
cd fpl-team-optimizer

yaml
Copy code

---

### 2. Create Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

Copy code

Mac/Linux

python3 -m venv venv
source venv/bin/activate

yaml
Copy code

---

### 3. Install Dependencies

pip install -r requirements.txt

yaml
Copy code

---

## Quick Start Guide

### Run Individual Phases

---

### Phase 1: Data Pipeline

python data_pipeline/run_pipeline.py

yaml
Copy code

Expected output:

- ✓ Downloaded 18 gameweeks from GitHub  
- ✓ Created database tables  
- ✓ Loaded 12,835+ player-gameweek records  
- ✓ Database ready for feature engineering  

---

### Phase 2: Feature Engineering

python feature_engineering/run_feature_engineering.py

yaml
Copy code

Expected output:

- ✓ Generated complete training data: 93,010 samples  
- ✓ Generated 189 comprehensive features per sample  
- ✓ Updated enhanced_training_data.csv  

---

### Phase 3: ML Prediction

python ml_model/run_raw_point_pipeline.py

yaml
Copy code

Expected output:

- ✓ Loaded 93,010 training samples  
- ✓ Generated 772 predictions for next gameweek  
- ✓ Saved predictions to data/predictions/  

---

### Phase 4: Team Optimization

python optimization/run_optimization.py

yaml
Copy code

Expected output:

- ✓ Optimized team with chip strategy recommendations  
- ✓ Recommended transfers with net gains  
- ✓ Selected captain with confidence score  
- ✓ Saved results to data/optimization/  

---

## Terminal UI

Optimize FPL team from terminal Ui (For user to have better experience for optimizing for single & multiple gameweeks and custom team optimization)

python fpl_optimizer_tui/run_tui.py

yaml
Copy code

---

## Project Structure

fpl-simulator/
├── data_pipeline/ Data collection
├── feature_engineering/ Feature generation (189 features)
├── ml_model/ ML predictions (RAW points)
├── optimization/ Team optimization (MIP)
├── data/ All data storage
│ ├── raw/ Raw data extracted
│ ├── processed/ SQLite databases
│ ├── features/ Generated feature CSVs
│ ├── models/ Trained ML models
│ ├── predictions/ Weekly predictions
│ └── optimization/ Optimization results
└── requirements.txt Python dependencies

yaml
Copy code

---

## Output Files

### Predictions

data/predictions/
├── raw_points_predictions_gw19.csv
├── latest_raw_points_predictions_gw19.csv
└── raw_points_with_value_gw19.csv

yaml
Copy code

---

### Optimization Results

data/optimization/
├── complete_optimization_gw19.json
├── optimized_team_gw19.csv
└── transfer_recommendations_gw19.json

yaml
Copy code

---

## For Faster Optimization

python optimization/run_optimization.py --method heuristic
python optimization/run_optimization.py --top-players 300

yaml
Copy code

---

## For Better Accuracy

python optimization/run_optimization.py --method mip --timeout 120
python feature_engineering/run_feature_engineering.py --gameweeks 10

yaml
Copy code

---

## Contributing

1. Fork the repository prevent duplication  
2. Create a feature branch  
3. Make your changes  
4. Run tests  
5. Submit a pull request  

---

## Learn More

### Key Concepts

- Mixed Integer Programming: Mathematical optimization for squad selection  
- Feature Engineering: 189 features capturing player performance  
- Walk-Forward Validation: Time-series testing approach  
- Separated Architecture: RAW prediction + value optimization  

This project relies on the excellent FPL-Elo-Insights dataset for historical FPL data.  
Special thanks to olbauday and team for maintaining this comprehensive resource that makes data-driven FPL analysis possible.

---

## License

MIT License - see LICENSE file for details.

---

## Support

For issues, questions, or suggestions:

- Check the Troubleshooting section  
- Search existing GitHub issues  
- Create a new issue with detailed description  
