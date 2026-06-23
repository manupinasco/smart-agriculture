# Crop Health Detection Based on Weather Info
This project was completed as part of the "Introduction to Machine Learning" course in the [Bachelor's Degree in Data Science](https://www.unsam.edu.ar/escuelas/ecyt/661/ciencia/ciencia-de-datos) at the [University of San Martín (UNSAM)](https://www.unsam.edu.ar/).

## Project Intro/Objective
The purpose of this project is . (Describe the main goals of the project and potential civic impact. Limit to a short paragraph, 3-6 Sentences)

Este trabajo tiene como objetivo optimizar la toma de decisiones de los actores involucrados en la cadena productiva del girasol, el maíz y el trigo en la Argentina. Para ello, se integró información sobre el estado general de las cosechas de distintas zonas del país junto con variables climatológicas. Estas últimas permiten predecir la evolución semanal de los cultivos, facilitando un monitoreo continuo y preciso. Al momento, se realizó el análisis de la lógica de negocio, la definición del umbral de decisión para transformar las variables continuas en una única variable categórica, y el entrenamiento y evaluación de distintos modelos orientados a la clasificación de los cultivos. Con la reducción de falsos positivos como criterio principal (minimizar cultivos erróneamente catalogados como buenos), se eligió uno de los modelos como candidato para ser lanzado en un entorno operativo.

### Methods Used
* Machine Learning
* Time Series Modeling
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* Pandas, jupyter

## Project Description
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data comes from [meteostat API](https://meteostat.net/es/) and [SAGyP weekly reports](https://www.magyp.gob.ar/sitio/areas/estimaciones/estimaciones/informes/). Those reportes are kept [here](https://github.com/manupinasco/smart-agriculture/tree/main/data/external) within this repo. Parsed raw data combined with meteostat weather info is kept [here](https://github.com/manupinasco/smart-agriculture/tree/main/data/raw).
3. Data processing/transformation scripts are being kept [here](https://github.com/manupinasco/smart-agriculture/tree/main/src/data).
4. Feature engineering script is kept [here](https://github.com/manupinasco/smart-agriculture/tree/main/src/features).
5. Model training/evaluation/prediction scripts are being kept [here](https://github.com/manupinasco/smart-agriculture/tree/main/src/models).

## Contributing Members

**Team Members (Contacts): [Emanuel Pinasco](https://github.com/manupinasco)(pinascoemanuel@gmail.com), [Ignacio Miguel García](https://github.com/manupinasco)(chonamiguelgarcia@gmail.com)**

* Feel free to contact team members with any questions or if you are interested in contributing!
