# online_retail_modernized
Online Retail II Customer Value Analytics

This project builds a practical customer value analytics workflow on the Online Retail II dataset. Instead of stopping at basic RFM segmentation, I extended the project into a more complete pipeline that combines data cleaning, interpretable customer features, clustering comparison, 90-day purchase prediction, conditional spend prediction, and action-oriented customer scoring.

Project Goal

The main goal of this project is to answer a more practical business question:

How can we identify high-value customers, predict who is likely to purchase in the next 90 days, estimate how much they may spend, and turn those results into actionable customer segments?

Dataset

The project uses the Online Retail II raw Excel transaction data.
After cleaning, the main positive-sales analysis dataset contained 793,609 rows. The cleaning process removed:

missing customer IDs
exact duplicates
invalid dates
blank invoice or stock code values
cancellation rows
non-positive quantities
non-positive prices
What I Built

This project includes four main parts:

1. Data Cleaning and Customer Feature Engineering

I started from the raw transaction data and built a cleaned customer-level dataset.
The feature set goes beyond basic RFM and includes:

classic RFM
average order value
purchase cadence
product diversity proxy
recent 14/30-day activity
return-related behavior features
2. Customer Segmentation

I compared KMeans, HDBSCAN, and GaussianMixture instead of using only one clustering method.
The final clustering choice was KMeans, because it provided the best balance between quantitative fit and segment interpretability. The largest segment in the final clustering snapshot was At Risk High Value with 2267 customers.

3. Predictive Modeling

I framed the customer value problem into two prediction tasks:

purchase_90_flag: whether a customer will purchase in the next 90 days
sales_90_value: future 90-day spend, conditional on purchasing

The final expected value was calculated as:

expected_90_value = purchase probability × predicted spend if purchase occurs

4. Actionable Customer Scoring

I translated model outputs into customer action groups such as:

Retain and reward
Nurture with cross-sell offers
Win-back high value customers
Low-cost automation
Modeling Results
Best Classification Model

LightGBMClassifier

Holdout ROC-AUC: 0.808
Holdout PR-AUC: 0.778
Holdout F1: 0.598
Best Regression Model

LightGBMRegressor

Holdout MAE: 639.80
Holdout RMSE: 3110.14
Holdout R²: 0.564
Key Insights

Some of the most important purchase-propensity features were:

avg_items_per_order
sales_per_item
recency
top_category_share
monetary

For spend prediction, the strongest features were:

sales_per_item
monetary
avg_order_value
top_category_share
days_since_first_purchase

A key takeaway from this project is that recent activity, historical value, and purchase cadence work better together than RFM alone when prioritizing customers.

Business Recommendations

Based on the final scoring table, I grouped customers into several action buckets:

Retain and reward: 1087 customers, average 90-day purchase probability 77.82%, average expected 90-day value 1100.40
Nurture with cross-sell offers: 226 customers, average purchase probability 59.00%
Win-back high value customers: 71 customers, average purchase probability 8.51%
Low-cost automation: 3865 customers, average purchase probability 20.11%

This makes the project more useful than a pure clustering exercise, because the outputs can be linked to actual customer management strategies.

Limitations

There are still a few limitations in the current version:

the dataset is transactional rather than campaign-level, so marketing lift cannot be measured directly
product categories are proxied from description keywords, which is practical but imperfect
spend regression is conditional on purchasing, so it should not be interpreted as a direct all-customer revenue forecast
Next Steps

Possible future improvements include:

adding richer product taxonomy and profit-based customer value signals
calibrating purchase probabilities for campaign prioritization
testing a monthly retraining schedule for segments and customer scores
