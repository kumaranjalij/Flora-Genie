
# Flora Genie - A Personalized Plant Recommendation System

Flora Genie is a personalized plant recommendation system designed to assist amateur gardening enthusiasts in selecting the most suitable plants for their home or garden. 

By specifying criteria such as plant category, light and water requirements, plant form, foliage description, and more, users can receive tailored recommendations to suit their needs.




## Features

- Personalized plant recommendations based on user-specified criteria.
- Comprehensive plant database including textual data and images.
- Multiple clustering algorithms for effective plant grouping.
- Similarity metrics for accurate recommendation generation.



## File Structure

```plain text
Flora-Genie/
├── Algorithms and Scripts/ 
│   └── ...
├── Application/
│   ├── app.py
│   └── ...
├── Data Files/
└── requirements.txt
```

**Algorithms and Scripts:** Contains Python scripts for the core functionality of the project such as clustering, recommendations, data preprocessing, data extraction, and utility functions.

**Application:** Contains files related to the web application.

**Data Files:** Contains the extracted plant dataset and related files.

**requirements.txt:** File listing all the necessary Python packages for the project

## Data Sources

[India Flora Online](https://indiaflora-ces.iisc.ac.in/)

[India Plants](https://www.indiaplants.com/)

[IMPPAT: Indian Medicinal Plants, Phytochemistry And Therapeutics](https://cb.imsc.res.in/imppat/)

## Algorithms and Methodologies

**Pipeline**

![Pipeline](https://github.com/kumaranjalij/Flora-Genie/blob/main/Screenshots/pipeline.png)

**Data Extraction**
- Compiled plant data using libraries like BeautifulSoup, Selenium, and urllib from three different data sources.

**Data Preprocessing**
- Transformed categorical columns into one-hot encoding.
- Applied TF-IDF for feature extraction from textual data.
- Performed tokenization, stop word removal, and lemmatization.

**Clustering Algorithms**
- **KMeans and KMeans++:** Identified optimal clusters (36) with a silhouette score of 0.2095.
- **Spectral Clustering:** Showed imbalance with most points in one cluster.
- **Hierarchical Clustering:** Observed distinct clusters from the dendrogram.

**Similarity Metrics**
- Cosine Similarity
- Jaccard Similarity

**Recommendation Generation**
- Used collaborative filtering and content-based similarity with TF-IDF for personalized recommendations.


## Results

- Balanced clustering distribution using KMeans and KMeans++ algorithms.
- Accurate personalized plant recommendations based on user criteria.
- Visual insights provided through plant images enhance user experience.
## Run Locally

Clone the repository

```bash
git clone https://github.com/kumaranjalij/Flora-Genie.git
```

Go to the project application directory

```bash
  cd Flora-Genie/Application
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the flask application

```bash
  python -m flask run
```


## Screenshots

![Flora Genie 1](https://github.com/kumaranjalij/Flora-Genie/blob/main/Screenshots/floragenie1.png)

![Flora Genie 2](https://github.com/kumaranjalij/Flora-Genie/blob/main/Screenshots/floragenie2.png)

![Medical](https://github.com/kumaranjalij/Flora-Genie/blob/main/Screenshots/medical.png)

![Personalized](https://github.com/kumaranjalij/Flora-Genie/blob/main/Screenshots/personalized.png)



