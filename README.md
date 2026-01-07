# ⚽ FPL Team Optimizer

> A comprehensive data science platform that automates optimal Fantasy Premier League team selection using machine learning predictions and mathematical optimization algorithms.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Installation & Setup](#️-installation--setup)
- [Quick Start Guide](#-quick-start-guide)
- [Terminal UI](#terminal-ui)
- [Project Structure](#-project-structure)
- [Output Files](#-output-files)
- [Optimization Tips](#-optimization-tips)
- [Contributing](#-contributing)
- [Learn More](#-learn-more)
- [License](#-license)
- [Support](#-support)

---

## 🚀 Project Overview

This system combines machine learning and mathematical optimization to build optimal FPL squads through a multi-phase pipeline:

1. **Data Pipeline:** Automated data collection and processing  
2. **Feature Engineering:** 189 comprehensive features per player  
3. **ML Prediction:** Position-specific LightGBM models for RAW point prediction  
4. **Team Optimization:** Mixed Integer Programming for squad selection  

---

## 📊 Key Results

- **21% more accurate** than individual predictions  
- **+30–36 point improvements** over current teams  
- **77% success rate** across gameweeks  
- **<30 second** optimization time  

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9+  
- Git  
- 4GB RAM minimum  

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fpl-team-optimizer.git
cd fpl-team-optimizer
2️⃣ Create Virtual Environment
bash
Copy code
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
🚀 Quick Start Guide
Run Individual Phases
Phase 1: Data Pipeline
bash
Copy code
python data_pipeline/run_pipeline.py
Expected output:

✓ Downloaded 18 gameweeks from GitHub

✓ Created database tables

✓ Loaded 12,835+ player-gameweek records

✓ Database ready for feature engineering

Phase 2: Feature Engineering
bash
Copy code
python feature_engineering/run_feature_engineering.py
Expected output:

✓ Generated complete training data: 93,010 samples

✓ Generated 189 comprehensive features per sample

✓ Updated enhanced_training_data.csv

Phase 3: ML Prediction
bash
Copy code
python ml_model/run_raw_point_pipeline.py
Expected output:

✓ Loaded 93,010 training samples

✓ Generated 772 predictions for next gameweek

✓ Saved predictions to data/predictions/

Phase 4: Team Optimization
bash
Copy code
python optimization/run_optimization.py
Expected output:

✓ Optimized team with chip strategy recommendations

✓ Recommended transfers with net gains

✓ Selected captain with confidence score

✓ Saved results to data/optimization/

Terminal UI
bash
Copy code
python fpl_optimizer_tui/run_tui.py
Optimize FPL team from terminal Ui (For user to have better experience for optimizing for single & multiple gameweeks and custom team optimization)

📁 Project Structure
text
Copy code
fpl-simulator/
├── data_pipeline/          Data collection
├── feature_engineering/    Feature generation (189 features)
├── ml_model/               ML predictions (RAW points)
├── optimization/           Team optimization (MIP)
├── data/                   All data storage
│   ├── raw/                Raw data extracted
│   ├── processed/          SQLite databases
│   ├── features/           Generated feature CSVs
│   ├── models/             Trained ML models
│   ├── predictions/        Weekly predictions
│   └── optimization/       Optimization results
└── requirements.txt        Python dependencies
📊 Output Files
Predictions
text
Copy code
data/predictions/
├── raw_points_predictions_gw19.csv
├── latest_raw_points_predictions_gw19.csv
└── raw_points_with_value_gw19.csv
Optimization Results
text
Copy code
data/optimization/
├── complete_optimization_gw19.json
├── optimized_team_gw19.csv
└── transfer_recommendations_gw19.json
⚡ Optimization Tips
For Faster Optimization
bash
Copy code
python optimization/run_optimization.py --method heuristic
python optimization/run_optimization.py --top-players 300
For Better Accuracy
bash
Copy code
python optimization/run_optimization.py --method mip --timeout 120
python feature_engineering/run_feature_engineering.py --gameweeks 10
🤝 Contributing
Fork the repository

Create a feature branch

Make your changes

Run tests

Submit a pull request

📚 Learn More
Key Concepts
Mixed Integer Programming: Mathematical optimization for squad selection

Feature Engineering: 189 features capturing player performance

Walk-Forward Validation: Time-series testing approach

Separated Architecture: RAW prediction + value optimization

This project relies on the excellent FPL-Elo-Insights dataset for historical FPL data.
Special thanks to olbauday and team for maintaining this comprehensive resource that makes data-driven FPL analysis possible.

📄 License
MIT License – see LICENSE file for details.

🆘 Support
For issues, questions, or suggestions:

Check the Troubleshooting section

Search existing GitHub issues

Create a new issue with detailed description
