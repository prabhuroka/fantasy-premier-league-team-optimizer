FPL Team Optimizer
A comprehensive data science platform that automates optimal Fantasy Premier League team selection using machine learning predictions and mathematical optimization algorithms.
🚀 Project Overview
This system combines machine learning and mathematical optimization to build optimal FPL squads through a multi-phase pipeline:
1.	Data Pipeline: Automated data collection and processing
2.	Feature Engineering: 189 comprehensive features per player
3.	ML Prediction: Position-specific LightGBM models for RAW point prediction
4.	Team Optimization: Mixed Integer Programming for squad selection
📊 Key Results
•	21% more accurate than individual predictions
•	+30-36 point improvements over current teams
•	77% success rate across gameweeks
•	<30 second optimization time
🛠️ Installation & Setup
Prerequisites
•	Python 3.9+
•	Git
•	4GB RAM minimum
1. Clone the Repository
git clone https://github.com/yourusername/fpl-team-optimizer.git
cd fpl-team-optimizer
2. Create Virtual Environment
Windows:
python -m venv venv
venv\Scripts\activate
Mac/Linux:
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
🚀 Quick Start Guide
Phase 1: Data Pipeline
python data_pipeline/run_pipeline.py
Phase 2: Feature Engineering
python feature_engineering/run_feature_engineering.py
Phase 3: ML Prediction
python ml_model/run_raw_point_pipeline.py
Phase 4: Team Optimization
python optimization/run_optimization.py
Terminal UI
python fpl_optimizer_tui/run_tui.py
📁 Project Structure
fpl-simulator/
├── data_pipeline/
├── feature_engineering/
├── ml_model/
├── optimization/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── models/
│   ├── predictions/
│   └── optimization/
└── requirements.txt
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
<img width="432" height="618" alt="image" src="https://github.com/user-attachments/assets/08244c3b-98f5-48af-939f-f1c43062cdd8" />

For Faster Optimization
bash
# Use heuristic method (faster, less optimal)
python optimization/run_optimization.py --method heuristic

# Limit player pool
python optimization/run_optimization.py --top-players 300
For Better Accuracy
bash
# Use MIP with longer timeout
python optimization/run_optimization.py --method mip --timeout 120

# Include more historical data
python feature_engineering/run_feature_engineering.py --gameweeks 10
🤝 Contributing
1.	Fork the repository
2.	Create a feature branch
3.	Make your changes
4.	Run tests
5.	Submit a pull request


**Key Concepts**
•	Mixed Integer Programming: Mathematical optimization for squad selection
•	Feature Engineering: 189 features capturing player performance
•	Walk-Forward Validation: Time-series testing approach
•	Separated Architecture: RAW prediction + value optimization
This project relies on the excellent FPL-Elo-Insights dataset for historical FPL data. Special thanks to olbauday and team for maintaining this comprehensive resource that makes data-driven FPL analysis possible.
**📄 License**
MIT License - see LICENSE file for details.
**🆘 Support**
For issues, questions, or suggestions:
1.	Check the Troubleshooting section
2.	Search existing GitHub issues
3.	Create a new issue with detailed description

