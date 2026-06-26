# UCL Hackathon: Football Match Prediction


## Overview
This project is a machine learning pipeline designed to predict the outcomes of UEFA Champions League (UCL) knockout matches. By combining historical match data with advanced metrics like expected goals (xG) scraped from the web (Understat.com), the model simulates tournament brackets round by round to predict the ultimate champion. The pipeline utilizes LightGBM, rigorous feature engineering, and time-series cross-validation to achieve high predictive accuracy.

---

## Step 1: Data Preparation

The foundation of the model relies on clean and comprehensive data collection. 

* **Data Cleaning:** Standardized team names to lowercase and mapped multiple names or abbreviations to a single, consistent format based on the datasets.
* **JSON Flattening:** Flattened the training and testing JSON files to ensure compatibility with the machine learning model.
* **Web Scraping:** Built a custom multi-threaded scraper to extract match-wise stats from Understat.com.
* **Extracted Metrics:** Captured expected goals (xG), expected goals against (xGA), actual goals, opponent data, and match dates by parsing hidden JavaScript data.
* **Caching & Fallbacks:** Saved scraped data locally as a CSV to speed up future runs. Implemented a fallback dummy dataset to prevent the pipeline from crashing if scraping fails.

---

## Step 2: Feature Engineering

We engineered high-dimensional features to capture team strength, tactical matchups, and recent momentum. If historical data was missing, we imputed reasonable average values to maintain model stability.

* **Expected Goals (xG) Features:** Calculated 5-match xG trends prior to a given match date. Included non-penalty xG (npxg) to filter out penalty anomalies and better reflect open-play dominance.
* **Head-to-Head (H2H) Features:** Captured historical rivalries and tactical mismatches by calculating total matches played, win rates, average goals per match, and recent form over the last 3 matches.
* **Team Strength Rating:** Analyzed a 365-day performance window using the formula: `Strength Rating = 1500 + (average_goal_difference * 100)`. This combines clean sheet rates, big win rates, and home vs. away performance splits.
* **Momentum Features:** Quantified late-season form surges, home advantage effects, and identified teams peaking at the right tournament stages.
* **Interaction Features:** Compared one team's performance metrics directly against another to measure relative strength.

---

## Step 3: Model Training

We selected **LightGBM (Light Gradient Boosting Machine)** for its speed, scalability, and ability to handle complex feature sets. 

* **Validation Strategy:** Used a Time Series Split with 10 splits to respect the chronological nature of football matches.
* **Feature Selection:** Implemented Recursive Feature Elimination with Cross-Validation (RFECV) to iteratively remove the least important features and select the optimal count that maximizes the F1-weighted score.
* **Hyperparameter Tuning:** Conducted a Randomized SearchCV across parameters like `num_leaves`, `max_depth`, `learning_rate`, and regularization terms. 
* **Early Stopping:** Monitored performance on a 20% validation set and halted training if no improvement was seen for 100 rounds to prevent overfitting.
* **Evaluation Metric:** Chosen the F1-weighted score because it balances precision and recall, handles the inherent class imbalance of knockout tournaments, and prevents the model from simply picking the favorite every time.

### Model Evaluation: The Impact of Web Scraping

Initially, the model was trained using only standard historical data without the scraped xG metrics. This baseline model struggled to generalize, achieving a validation accuracy of ~0.69 and displaying poor recall for predicting upsets. 

Integrating the scraped xG data and fine-tuning the hyperparameters significantly improved performance, bringing the Best CV Score (F1-Weighted) from GridSearch up to 0.6908.

### Final Model Evaluation on Out-of-Time (OOT) Test Set

When evaluated on the completely unseen Out-of-Time test set, the final model demonstrated strong predictive capabilities across the board.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.8235 |
| **F1-Weighted** | 0.8248 |
| **Log Loss** | 0.5787 |

**Classification Report (OOT):**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.89 | 0.80 | 0.84 | 10 |
| **1** | 0.75 | 0.86 | 0.80 | 7 |
| **Macro Avg** | 0.82 | 0.83 | 0.82 | 17 |
| **Weighted Avg** | 0.83 | 0.82 | 0.82 | 17 |

---

## Step 4: Prediction & Simulation

* **Training Data Cutoff:** The model was explicitly trained on historical knockout match data up to April 31st, 2017. 
* **Dynamic Feature Calculation:** When predicting new matchups, the pipeline recalculates all features dynamically, utilizing all historical data right up to the exact match date. Feature calculation logic remains identical to the training phase.
* **Bracket Simulation:** Forecasts the outcome of each match in the knockout tournament tree, advancing winners round by round until a champion is crowned.
* **Custom Scoring Function:** Developed an internal scoring function to estimate final model performance on our own, reducing the need for excessive external submissions during the hackathon.
