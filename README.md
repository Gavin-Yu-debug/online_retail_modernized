# online_retail_modernized

# Online Retail II Customer Value Analytics

This project is built on the Online Retail II transaction dataset.  
Instead of stopping at basic RFM segmentation, I tried to turn it into a more complete customer value analysis workflow.

The main idea of this project is:

- clean raw retail transaction data
- build interpretable customer features
- compare clustering methods for customer segmentation
- predict whether a customer will purchase in the next 90 days
- estimate future spend if the customer purchases
- combine these signals into a more practical customer scoring system

So this project is not only about "grouping customers", but also about understanding **who is more valuable, who is more likely to come back, and what kind of action might make sense for different customer groups**.

---

## Project Goal

The goal of this project is to answer a practical business question:

**How can we identify high-value customers, predict near-future purchase behavior, and turn the results into actionable customer segments?**

To make this more useful, I did not only use traditional RFM analysis. I also added customer behavior features, compared clustering methods, and built time-based predictive models for future purchase and spend.

---

## Dataset

The project uses the **Online Retail II** raw Excel transaction data.

After cleaning, the main positive-sales dataset contained:

- **793,609 rows**

Main cleaning steps included:

- removing missing customer IDs
- removing exact duplicates
- removing invalid dates
- removing cancellation rows
- removing non-positive quantities
- removing non-positive prices

The goal of the cleaning stage was to make the downstream customer analysis more stable and realistic.

---

## What I Did

### 1. Data Cleaning
I started from the raw transaction records and built a cleaner customer-level analysis base.

This step included:
- handling missing values
- removing duplicated transactions
- filtering out cancellation / return-style rows for the main positive-sales pipeline
- standardizing transaction-level fields
- creating a usable sales value field for customer analysis

### 2. Customer Feature Engineering
I did not only use basic RFM.

I also built additional customer-level features such as:
- average order value
- purchase cadence
- recent 14-day and 30-day activity
- product diversity proxy
- return-related behavior
- purchase span and recency-related variables

This helped the project move from simple segmentation into a more complete customer value modeling task.

### 3. Clustering Comparison
Instead of using only one segmentation method, I compared:

- **KMeans**
- **HDBSCAN**
- **GaussianMixture**

The final clustering method was **KMeans**, mainly because it gave the best balance between clustering quality and business interpretability.

### 4. 90-Day Prediction
I then framed the customer problem into two predictive tasks:

- **purchase_90_flag**  
  whether a customer will purchase in the next 90 days

- **sales_90_value**  
  how much the customer will spend in the next 90 days, conditional on purchasing

To make this more realistic, I used a **time-based design** instead of random splitting.

### 5. Customer Scoring and Business Actions
Finally, I combined model outputs into a customer scoring workflow and grouped customers into action buckets such as:

- retain and reward
- nurture with cross-sell offers
- win-back high-value customers
- low-cost automation

This makes the project more practical than a pure clustering exercise.

---

## Modeling Results

### Best Classification Model
**LightGBMClassifier**

- Holdout ROC-AUC: **0.808**
- Holdout PR-AUC: **0.778**
- Holdout F1: **0.598**

### Best Regression Model
**LightGBMRegressor**

- Holdout MAE: **639.80**
- Holdout RMSE: **3110.14**
- Holdout R²: **0.564**

---

## Key Findings

Some of the most important purchase-propensity features were:

- avg_items_per_order
- sales_per_item
- recency
- top_category_share
- monetary

Some of the most important spend-related features were:

- sales_per_item
- monetary
- avg_order_value
- top_category_share
- days_since_first_purchase

A useful takeaway from this project is that:

**recent activity, historical value, and purchase cadence work better together than RFM alone when prioritizing customers.**

---

## Business Recommendations

Based on the final customer scoring output, I grouped customers into several practical action buckets:

### Retain and reward
- Customers: **1087**
- Average 90-day purchase probability: **77.82%**
- Average expected 90-day value: **1100.40**

### Nurture with cross-sell offers
- Customers: **226**
- Average 90-day purchase probability: **59.00%**
- Average expected 90-day value: **147.66**

### Win-back high value customers
- Customers: **71**
- Average 90-day purchase probability: **8.51%**
- Average expected 90-day value: **106.74**

### Low-cost automation
- Customers: **3865**
- Average 90-day purchase probability: **20.11%**
- Average expected 90-day value: **75.11**

This part helped connect the model outputs to more realistic customer operation ideas.

---

## Why This Project Is Different from a Basic RFM Project

A basic customer segmentation project usually ends at RFM scoring or simple clustering.

This project goes further by adding:

- richer customer behavior features
- clustering method comparison
- time-based future purchase prediction
- future spend estimation
- customer-level expected value scoring
- action-oriented business recommendations

So the final output is not just "which cluster a customer belongs to", but also **how likely the customer is to return, how much they may spend, and what action might be suitable**.

---

## Limitations

There are still some limitations in the current version:

- the dataset is transactional rather than campaign-level, so it cannot directly measure marketing lift
- product categories are approximated from description keywords, so category signals are useful but not perfect
- spend regression is conditional on purchase, so it should not be treated as a direct all-customer revenue forecast

---

## Next Steps

Possible next improvements:

- add richer product taxonomy or profit-based customer value
- calibrate probabilities for campaign prioritization
- update segments and scores on a monthly retraining schedule
- test whether different customer groups need different prediction models

---

