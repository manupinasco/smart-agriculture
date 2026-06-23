# Crop Health Prediction Based on Weather Information

This project was completed as part of the "Introduction to Machine Learning" course in the [Bachelor's Degree in Data Science](https://www.unsam.edu.ar/escuelas/ecyt/661/ciencia/ciencia-de-datos) at the [University of San Martín (UNSAM)](https://www.unsam.edu.ar/).

## Project Intro/Objective

The purpose of this project is to optimize decision-making for stakeholders involved in the Argentine wheat production chain, helping prevent massive crop losses.

Specific goals are:

* Learn the relationship between historical weather patterns and crop health.
* Identify the most suitable time windows for prediction and planning.
* Develop interpretable models that provide meaningful decision boundaries.

### Methods Used

* Machine Learning
* Time Series Modeling
* Data Visualization
* Predictive Modeling

### Technologies

* Python
* Pandas
* Jupyter

## Project Description

This project implements ETL and machine learning pipelines to predict wheat crop health using weather variables. Crop health was originally represented by four different indicators, which were combined into a single metric through a weighted aggregation. Based on the official definition of *emergency zones* in Argentina (50% crop loss), the target variable was binarized using a threshold of 0, corresponding to the point where, in the best-case scenario, 25% of the crop is classified as poor or 50% as regular.

Missing weekly observations were recovered by web scraping official weekly reports. To avoid temporal data leakage, the modeling strategy combined time-series cross-validation, a final-campaign holdout set, and predictions limited to a one-month forecasting horizon. Since the primary objective was to detect poor crop conditions, model selection prioritized maximizing recall and F1-score for the positive (bad crop) class.

For more information, read the presentation of the project [here](https://github.com/manupinasco/smart-agriculture/tree/main/reports).

## Getting Started

1. Clone this repo (for help, see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

   * **Note:** Raw weather data comes from the [Meteostat API](https://meteostat.net/es/) and crop health information from [SAGyP weekly reports](https://www.magyp.gob.ar/sitio/areas/estimaciones/estimaciones/informes/). The downloaded reports are available [here](https://github.com/manupinasco/smart-agriculture/tree/main/data/external), while the parsed data combined with Meteostat weather information can be found [here](https://github.com/manupinasco/smart-agriculture/tree/main/data/raw).
2. Process and transform the data using [these scripts](https://github.com/manupinasco/smart-agriculture/tree/main/src/data).
3. Run the feature engineering pipeline using [this script](https://github.com/manupinasco/smart-agriculture/tree/main/src/features).
4. Train and evaluate the models (or use them for prediction) with [these scripts](https://github.com/manupinasco/smart-agriculture/tree/main/src/models).

**Example:**

```bash
python -m src.data.extract
```

## Team Members

* [Emanuel Pinasco](https://github.com/manupinasco) ([pinascoemanuel@gmail.com](mailto:pinascoemanuel@gmail.com))

* [Ignacio Miguel García](https://github.com/im-garcia) ([chonamiguelgarcia@gmail.com](mailto:chonamiguelgarcia@gmail.com))

* Feel free to contact the team members with any questions or if you are interested in contributing!
