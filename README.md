# Detecting Emerging Topics and Predicting Changes in Search Interest Across U.S. States

This project uses weekly Google Trends data to study when topics begin to trend and predict their search interest across different states over the following weeks. It has 2 stages
- **Stage 1 — Emerging topic detection.** We use a support vector machine (SVM) to classify time windows based on patterns in national search interest.
- **Stage 2 — Geographic forecasting.** For topics identified as emerging, we use historical state-level interest scores to predict each state's score 1, 2, and 4 weeks ahead using Gradient Boosting models.

## Installation

The processed CSV files are included in `data/`. You can use these files and skip the data collection and preprocessing

From the project folder, install the dependecies:

```bash
pip install pandas numpy scikit-learn
```
## Data collection

We used Wikipedia categories to identify candidate topics. Then for each topic, we used pytrends to collect three years of weekly Google Trends interest data.

## Data preprocessing

We format the data and remove duplicates. Then apply the Hamed-Rao modified Mann-Kendall test and Sen's slope to determine the emerging events. We check 12-week windows of search interest and moving the 4 weeks at a time. For each window, we identify emerging events using 25% growth between the first and last four week averages, a positive Sen's slope, and a Mann-Kendall test with `p < 0.05`
