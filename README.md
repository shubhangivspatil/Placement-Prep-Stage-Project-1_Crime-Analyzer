**Chicago Crime Data Analysis**

**Project Overview**
This project analyzes crime data from Chicago to identify patterns, trends, and hotspots.
By leveraging advanced machine learning techniques, the aim is to improve public safety through data-driven strategies, 
optimize law enforcement resource allocation, and forecast crime occurrences.

**Dataset Description**
The dataset consists of reported crime data in Chicago with the following attributes:

ID: Unique identifier for each crime incident.
Primary Type: Type of crime (e.g., theft, assault).
Date: Date and time of occurrence.
Location Description: Description of the location.
Arrest: Indicates whether an arrest was made.
Latitude/Longitude: Geographical coordinates of the crime scene.
Goals and Objectives
Analyze Crime Trends: Understand changes in crime over time.
Identify Hotspots: Use geospatial data to pinpoint high-crime areas.
Evaluate Arrest Rates: Assess the effectiveness of law enforcement.
Predict Incidents: Build models to forecast crime likelihood.

**Algorithms Used**

**1. Logistic Regression**
Purpose: Baseline binary classification model.

**Why Used:**
Simplicity and interpretability.
Efficient for quick insights on linear relationships.
Metrics:
Accuracy: 64%
Precision: 64%
Recall: 64%
F1-Score: 64%
Confusion Matrix:
Predicted: No Arrest	Predicted: Arrest
Actual: No Arrest	88,307	51,338
Actual: Arrest	50,310	89,489

**2. Random Forest**
Purpose: Ensemble model for robust classification.

**Why Used:**
Handles large and imbalanced datasets effectively.
Provides feature importance insights.
Reduces overfitting through bagging.

****Metrics:**
Accuracy: 92%
Precision: 88% (No Arrest), 96% (Arrest)
Recall: 97% (No Arrest), 87% (Arrest)
F1-Score: 92%
Confusion Matrix:
Predicted: No Arrest	Predicted: Arrest
Actual: No Arrest	135,074	4,571
Actual: Arrest	18,064	121,735

**3. Neural Network**
Purpose: Model non-linear relationships for advanced prediction.

**Why Used:**
Capable of capturing complex patterns in the data.
Effective for large datasets with significant variability.

**Metrics:**
Accuracy: 54%
Precision: 55%
Recall: 54%
F1-Score: 54%
Confusion Matrix:
Predicted: No Arrest	Predicted: Arrest
Actual: No Arrest	79,345	60,300
Actual: Arrest	66,876	72,923

**Comparison of Algorithms**
Metric	Logistic Regression	Random Forest	Neural Network
Accuracy	64%	92%	54%
Precision	64%	96% (Arrest)	55%
Recall	64%	97% (No Arrest)	54%
F1-Score	64%	92%	54%

**Conclusion: Why Random Forest Was Selected**
Accuracy and Precision: Random Forest consistently outperformed other models with an accuracy of 92%.
Feature Importance: Provided clear insights into which factors (e.g., location, time) influence arrests.
Robustness: Handled the imbalance in arrest data effectively using class weighting.
Scalability: Processed large datasets efficiently.
While Logistic Regression served as a solid baseline and Neural Networks explored non-linear complexities, Random Forest emerged as the optimal choice for achieving actionable insights.

**Visualization and Insights**
Temporal Trends: Crime occurrences analyzed over years and months.
Geospatial Hotspots: Interactive maps highlight high-crime areas.
Arrest Rates: District-wise evaluation reveals varying law enforcement effectiveness.

**Key Recommendations**
Resource Allocation: Deploy resources to identified hotspots during peak hours.
Crime Prevention: Implement targeted measures in high-crime locations.
Law Enforcement Training: Focus on districts with lower arrest effectiveness.


**Additional Resources**
Links to the live dashboard, additional documentation, or related projects
https://drive.google.com/file/d/1BzsxHn5KT6Gq8pfanAmXrE_8d6hoMQWy/view?usp=drive_link
