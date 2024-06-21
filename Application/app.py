#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:02:11 2024

@author: swadikorattiparambil
"""

#%% 

#importing libraries
from flask import Flask, request, render_template
import numpy
import pandas
import sklearn
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from scipy.spatial.distance import jaccard

#%% 

#import data files
data=pd.read_csv("data/KMeans++_data_with_cluster_id.csv", index_col=0)
data_cosine=data.drop("Cluster",axis=1)
cluster_means_df=pd.read_csv("data/KMeans++_cluster_means_df.csv", index_col=0)
plant_data = pd.read_csv('data_indiaplant_1_3000.csv', index_col=0)
plant_data = plant_data.set_index('Plant_Id')
data_cosine_med=pd.read_csv("final_data_encoded_Med.csv", index_col=0)
plant_medical_data = pd.read_csv('Complete_Medicinal_Data_V2.csv', sep=";", index_col=0)
image_file =pd.read_csv("data/data_indiaplant_image.csv", converters={"Image_Paths":eval})
image_file_csv =pd.read_csv("data/data_indiaplant_image.csv")
medical_image_file =pd.read_csv("data/Complete_Medicinal_Image_Data.csv", delimiter=";", converters={"Image_Paths":eval})
#%% 

#helper functions for recommendation

def recommend_plants_jaccard(input_values, data, top_n=20):
    # Filter data based on input values
    filtered_data = data.copy()
    for column, value in input_values.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]

    if len(filtered_data) == 0:
        print("No matching plants found for the input criteria.")
        return None
    
    # Prepare input vector
    input_vector = pd.DataFrame(index=[0], columns=data.columns)
    input_vector = input_vector.fillna(0)  # Fill with zeros initially
    for column, value in input_values.items():
        if column in input_vector.columns:
            input_vector.at[0, column] = 1  # Set the input value to 1 in the input vector
    
   
    similarity_scores = filtered_data.apply(lambda x: jaccard(input_vector.values.flatten(), x.values), axis=1)
   
    
    # Add similarity scores to the filtered data
    filtered_data['Similarity'] = similarity_scores

    # Sort by similarity scores in descending order
    recommended_plants = filtered_data.sort_values(by='Similarity', ascending=False).head(top_n)
    
    return recommended_plants

def recommend_plants_kmeans(input_values, cluster_means_df, data, top_n=20):

    # Prepare input vector for cluster-mean similarity
    user_vector = pd.DataFrame(index=[0], columns=cluster_means_df.columns)
    user_vector = user_vector.fillna(0)  # Fill with zeros initially
    for column, value in input_values.items():
        if column in user_vector.columns:
            user_vector.at[0, column] = 1  # Set the input value to 1 in the input vector
            

    # Step 2: Calculate similarity between the user vector and the cluster mean vectors
    cluster_similarities = cosine_similarity(user_vector, cluster_means_df)

    # Find the cluster with the highest similarity
    optimal_cluster = np.argmax(cluster_similarities)
    print("Optimal Cluster:", optimal_cluster)
    
    
    # Prepare input vector for cluster similarity
    user_vector1 = pd.DataFrame(index=[0], columns=data.columns)
    user_vector1 = user_vector1.fillna(0)  # Fill with zeros initially
    for column, value in input_values.items():
        if column in user_vector1.columns:
            user_vector1.at[0, column] = 1  # Set the input value to 1 in the input vector
            

    # Step 3: Calculate similarity between the user vector and plants in the optimal cluster
    optimal_cluster_indices = data[data['Cluster'] == optimal_cluster].index
    optimal_cluster_data = data.loc[optimal_cluster_indices]
    similarities = cosine_similarity(user_vector1, optimal_cluster_data)
    optimal_cluster_data['Similarity'] = similarities.flatten()
    
        
    # Sort by similarity scores in descending order
    recommended_plants = optimal_cluster_data.sort_values(by='Similarity', ascending=False).head(top_n)
    #print(recommended_plants)
    return recommended_plants[['Similarity']]


    return recommended_plants

def recommend_plants_cosine(input_values, data, top_n=20):
    # Filter data based on input values
    filtered_data = data.copy()
    for column, value in input_values.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]

    if len(filtered_data) == 0:
        print("No matching plants found for the input criteria.")
        return None
    
    # Prepare input vector for cosine similarity
    input_vector = pd.DataFrame(index=[0], columns=data.columns)
    input_vector = input_vector.fillna(0)  # Fill with zeros initially
    for column, value in input_values.items():
        if column in input_vector.columns:
            input_vector.at[0, column] = 1  # Set the input value to 1 in the input vector

    # Calculate cosine similarity between input vector and filtered data
    similarity_scores = cosine_similarity(input_vector.values, filtered_data.values)
    # Add similarity scores to the filtered data
    filtered_data['Similarity'] = similarity_scores.flatten()

    # Sort by similarity scores in descending order
    recommended_plants = filtered_data.sort_values(by='Similarity', ascending=False).head(top_n)
    #print(recommended_plants)
    return recommended_plants[['Similarity']]


def recommend_plants_priority(input_values,data_jaccard, cluster_means_df, data, top_n=20):
    # Recommendations using jaccard similarity method
    recommendations_jaccard = recommend_plants_jaccard(input_values, data_jaccard, top_n)
    
    # Check if enough recommendations are obtained from jaccard similarity method
    if recommendations_jaccard is not None and len(recommendations_jaccard) >= top_n:
        recommendations_jaccard['Source'] = 'Jaccard Similarity'
        return recommendations_jaccard
    else:
        remaining_recommendations = top_n - len(recommendations_jaccard) if recommendations_jaccard is not None else top_n
        # Recommendations using k-means method
        recommendations_kmeans = recommend_plants_kmeans(input_values, cluster_means_df, data, remaining_recommendations)
        
        # Filter out any duplicate plant IDs from k-means recommendations
        if recommendations_jaccard is not None:
            duplicate_plant_ids = recommendations_jaccard.index.intersection(recommendations_kmeans.index)
            recommendations_kmeans = recommendations_kmeans[~recommendations_kmeans.index.isin(duplicate_plant_ids)]
        
        recommendations_kmeans['Source'] = 'K-Means'
        # Concatenate both sets of recommendations
        if recommendations_jaccard is not None:
            all_recommendations = pd.concat([recommendations_jaccard, recommendations_kmeans])
        else:
            all_recommendations = recommendations_kmeans
        
        # If the total number of recommendations is less than 10, obtain additional recommendations from either method
        if len(all_recommendations) < top_n:
            remaining_recommendations = top_n - len(all_recommendations)
            additional_recommendations = recommend_plants_kmeans(input_values, cluster_means_df, data, remaining_recommendations)
            
            # Filter out any duplicate plant IDs from additional recommendations
            duplicate_plant_ids = all_recommendations.index.intersection(additional_recommendations.index)
            additional_recommendations = additional_recommendations[~additional_recommendations.index.isin(duplicate_plant_ids)]
            additional_recommendations['Source'] = 'K-Means additional'
            
            # Concatenate additional recommendations with existing recommendations
            all_recommendations = pd.concat([all_recommendations, additional_recommendations])
        
        return all_recommendations.head(top_n)



    # Recommendations using cosine similarity method
    recommendations_cosine = recommend_plants_cosine(input_values, data_cosine, top_n)
    
    # Check if enough recommendations are obtained from cosine similarity method
    if recommendations_cosine is not None and len(recommendations_cosine) >= top_n:
        recommendations_cosine['Source'] = 'Cosine Similarity'
        return recommendations_cosine
    else:
        remaining_recommendations = top_n - len(recommendations_cosine) if recommendations_cosine is not None else top_n
        # Recommendations using k-means method
        recommendations_kmeans = recommend_plants_kmeans(input_values, cluster_means_df, data, remaining_recommendations)
        
        # Filter out any duplicate plant IDs from k-means recommendations
        if recommendations_cosine is not None:
            duplicate_plant_ids = recommendations_cosine.index.intersection(recommendations_kmeans.index)
            recommendations_kmeans = recommendations_kmeans[~recommendations_kmeans.index.isin(duplicate_plant_ids)]
        
        recommendations_kmeans['Source'] = 'K-Means'
        # Concatenate both sets of recommendations
        if recommendations_cosine is not None:
            all_recommendations = pd.concat([recommendations_cosine, recommendations_kmeans])
        else:
            all_recommendations = recommendations_kmeans
        
        # If the total number of recommendations is less than 10, obtain additional recommendations from either method
        if len(all_recommendations) < top_n:
            remaining_recommendations = top_n - len(all_recommendations)
            additional_recommendations = recommend_plants_kmeans(input_values, cluster_means_df, data, remaining_recommendations)
            
            # Filter out any duplicate plant IDs from additional recommendations
            duplicate_plant_ids = all_recommendations.index.intersection(additional_recommendations.index)
            additional_recommendations = additional_recommendations[~additional_recommendations.index.isin(duplicate_plant_ids)]
            additional_recommendations['Source'] = 'K-Means additional'
            
            # Concatenate additional recommendations with existing recommendations
            all_recommendations = pd.concat([all_recommendations, additional_recommendations])
        
        return all_recommendations.head(top_n)
    
    
def recommend_medical_plants_cosine(input_values, data, top_n=20):
    # Filter data based on input values
    filtered_data = data.copy()
    for column, value in input_values.items():
        if column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[column] == value]

    if len(filtered_data) == 0:
        print("No matching plants found for the input criteria.")
        return None
    
    # Prepare input vector for cosine similarity
    input_vector = pd.DataFrame(index=[0], columns=data.columns)
    input_vector = input_vector.fillna(0)  # Fill with zeros initially
    for column, value in input_values.items():
        if column in input_vector.columns:
            input_vector.at[0, column] = 1  # Set the input value to 1 in the input vector

    # Calculate cosine similarity between input vector and filtered data
    similarity_scores = cosine_similarity(input_vector.values, filtered_data.values)
    # Add similarity scores to the filtered data
    filtered_data['Similarity'] = similarity_scores.flatten()

    # Sort by similarity scores in descending order
    recommended_plants = filtered_data.sort_values(by='Similarity', ascending=False).head(top_n)
    #print(recommended_plants)
    return recommended_plants[['Similarity']]


    
custom_stopwords = stopwords.words('english')
custom_stopwords.remove('more')
custom_stopwords.remove('not')

df_temp = pd.read_csv('plant_doc.csv')
df_temp = df_temp.drop(['Unnamed: 0'], axis=1)

tfidf = TfidfVectorizer(stop_words=custom_stopwords)
matrix = tfidf.fit_transform(df_temp['clean_info'])
similarity = cosine_similarity(matrix)
df_temp = df_temp.drop(['clean_info', 'Plant_Id'], axis=1)

def recommend_plant_cn(plant_common_name):
        plant_name = df_temp.loc[df_temp['Common_Name'] == plant_common_name]['Scientific_Name'].values[0]
        indx = df_temp[df_temp['Scientific_Name'] == plant_name].index[0]
        indx = df_temp.index.get_loc(indx)
        distances = sorted(list(enumerate(similarity[indx])), key=lambda x: x[1], reverse=True)[:20]

        list_plant = []
        for i in distances:
            if(i[1]>0.3):
                list_plant.append(i[0])
        return list_plant[:5]

def extract_image_list_from_string(scientific_name):
    image_path = image_file[image_file['Scientific_Name'] == scientific_name]['Image_Paths']
    if(len(image_path)>0):
        x = image_path.values[0]
        x = ast.literal_eval(x)
        x = [n.strip() for n in x]
        return x
    else:
        return []

    
def extract_image_list(scientific_name):
    image_path = image_file[image_file['Scientific_Name'] == scientific_name]['Image_Paths']
    if(len(image_path)>0):
        return image_path.values[0]
    else:
        return []
    
def extract_medical_image_list(scientific_name):
    image_path = medical_image_file[medical_image_file['Scientific_Name'] == scientific_name]['Image_Paths']
    if(len(image_path)>0):
        return image_path.values[0]
    else:
        return []

#%% 
#helper constants

feature_value_dict = {
    'Light_Low light tolerant': 'Low light tolerant',
    'Light_Semi shade': 'Semi shade',
    'Light_Shade growing': 'Shade growing',
    'Light_Sun growing': 'Sun growing',
    'Water_Requires_less': 'Requires less',
    'Water_Normal': 'Normal',
    'Water_Requires more': 'Requires more',
    'Category_Bamboos': 'Bamboos',
    'Category_Bromeliads': 'Bromeliads',
    'Category_Cacti & Succulents': 'Cacti & Succulents',
    'Category_Climbers': 'Climbers',
    'Category_Creepers & Vines': 'Creepers & Vines',
    'Category_Ferns': 'Ferns',
    'Category_Flowering Pot Plants': 'Flowering Pot Plants',
    'Category_Fruit Plants': 'Fruit Plants',
    'Category_Grasses & Grass like plants': 'Grasses & Grass like plants',
    'Category_Groundcovers_lawns': 'Groundcovers lawn',
    'Category_Indoor Plants': 'Indoor Plants',
    'Category_Lilies & Bulbous plants': 'Lilies & Bulbous plants',
    'Category_Medicinal Plants': 'Medicinal Plants',
    'Category_Orchids': 'Orchids',
    'Category_Palms and Cycads': 'Palms and Cycads',
    'Category_Rose_Hybrid_Climbers': 'Rose Hybrid Climbers',
    'Category_Rose_Miniature_Floribundas': 'Rose Miniature Floribundas',
    'Category_Shrubs': 'Shrubs',
    'Category_Spice plants & edible Herbs': 'Spice plants & edible Herbs',
    'Category_Terrific Tropicals  The Ideal Gifts': 'Terrific Tropicals',
    'Category_Trees': 'Trees',
    'Category_Vegetable': 'Vegetable',
    'Category_Water & Aquatic Plants': 'Water & Aquatic Plants',
    "Plant_Form_Columnar": "Columnar",
"Plant_Form_Low spreading": "Low spreading",
"Plant_Form_Oval":"Oval",
"Plant_Form_Pyramidal": "Pyramidal",
"Plant_Form_Spherical or rounded": "Spherical or rounded",
"Plant_Form_Spreading": "Spreading",
"Plant_Form_Upright or Erect": "Upright or Erect",
"Plant_Form_Weeping": "Weeping",
    "Primary_Grown_for_Flowers": "Flowers",
"Primary_Grown_for_Foliage": "Foliage",
"Primary_Grown_for_Fruit or Seed": "Fruit or Seed",
"Primary_Grown_for_Roots or tubers": "Roots or tubers",
"Primary_Grown_for_Stems or Timber": "Stems or Timber",
"Height_or_Length_Less than 50 cms": "Less than 50 cms",
"Height_or_Length_50 cms to 100 cms": "50 cms to 100 cms",
"Height_or_Length_1 to 2 meters": "1 to 2 meters",
"Height_or_Length_2 to 4 meters": "2 to 4 meters",
"Height_or_Length_4 to 6 meters": "4 to 6 meters",
"Height_or_Length_6 to 8 meters": "6 to 8 meters",
"Height_or_Length_8 to 12 meters": "8 to 12 meters",
"Height_or_Length_More than 12 meters": "More than 12 meters",
"Spread_or_Width_less_than_1_meter": "Less than 1 meter",
"Spread_or_Width_1 to 2 meters": "1 to 2 meters",
"Spread_or_Width_2 to 4 meters": "2 to 4 meters",
"Spread_or_Width_4 to 6 meters": "4 to 6 meters",
"Spread_or_Width_6 to 8 meters": "6 to 8 meters",
"Spread_or_Width_8 to 12 meters": "8 to 12 meters",
"Spread_or_Width_More than 12 meters": "More than 12 meters",
"Flowering_Season_Feb_Mar_Spring": "Feb to Mar Spring",
"Flowering_Season_Apr_May_Summer": "Apr May Summer",
"Flowering_Season_Jun_Sep_Monsoon": "Jun Sep Monsoon",
"Flowering_Season_Oct_Nov_Autumn": "Oct Nov Autumn",
"Flowering_Season_Dec_Jan_Winter": "Dec Jan Winter",
"Flowering_Season_Year_round_flowering": "Year round flowering",
"Flowering_Season_Flowers once in many years": "Flowers once in many years",
"Flowering_Season_Non Flowering": "Non Flowering",
"Flowering_Season_Flowers are inconspicuous": "Flowers are inconspicuous",
"Plant_Form_Climbing or growing on support": "Climbing or growing on support",
'Foliage_Color_Blue': 'Blue',
'Foliage_Color_Blue Grey or Silver': 'Grey',
'Foliage_Color_Bronze or coppery': 'Bronze',
'Foliage_Color_Brown': 'Brown',
'Foliage_Color_Cream or off white': 'Cream',
'Foliage_Color_Green': 'Green',
'Foliage_Color_Orange': 'Orange',
'Foliage_Color_Pink': 'Pink',
'Foliage_Color_Purple': 'Purple',
'Foliage_Color_Red': 'Red',
'Foliage_Color_Variegated': 'Variegated',
'Foliage_Color_Very dark green almost black': 'Dark green',
'Foliage_Color_White': 'White',
'Foliage_Color_Yellow': 'Yellow'
    
}

#%% 

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = []
    if "light" in request.form:
        light = request.form.getlist('light')
        #print('Hello world!', file=sys.stderr)
        #print(light, file=sys.stderr)
        features = features + light
        
    if "water" in request.form:
        water = request.form.getlist('water')
        features = features + water
        
    if "category" in request.form:
        category = request.form.getlist('category')
        features = features + category
        
    if "color" in request.form:
        color = request.form.getlist('color')
        features = features + color
        
    if "plantform" in request.form:
        plantform = request.form.getlist('plantform')
        features = features + plantform
        
    if "grownfor" in request.form:
        grownfor = request.form.getlist('grownfor')
        features = features + grownfor
        
    if "height" in request.form:
        height = request.form.getlist('height')
        features = features + height
        
    if "spread" in request.form:
        spread = request.form.getlist('spread')
        features = features + spread
        
    if "flowring" in request.form:
        flowring = request.form.getlist('flowring')
        features = features + flowring
        
    
    
        
    
    #result = ' '.join(features)
    #return render_template('index.html', result=result)
    '''
    
    plantResult = [{"heading":'card 1', "body": 'this is card body 1'},
                   {"heading":'card 2', "body": 'this is card body 2'}]
    
    
    
    headings = ("Name", "Category", "Light")
    data = (
        ("Rose", "Flower", "More light"),
        ("Money plant", "Creeper", "Less light"),
        ("Bamboo plant", "Bamboo", "Less light")
        )
    '''
    input_values = {}
    for feature in features:
        input_values[feature] = 1
        
    input_feature = []
    
    for feature in features:
        if feature in feature_value_dict.keys():
            input_feature.append(feature_value_dict[feature])
        else:
            input_feature.append(feature)
            
    input_feature_str = ", ".join(input_feature)
    new_feature= []
    new_feature.append("<b> Plant Features Selected</b>")
    new_feature.append(input_feature_str)
        
    
    
    #input_values = {'Category_Bamboos': 1,'Light_Sun growing': 1,'Light_Semi shade': 1, 'Water_Normal': 1,'Plant_Form_Upright or Erect' : 1,'Primarily_Grown_for_Flowers': 1,'Flowering_Season_Year-around flowering': 1,'Spread_or_Width_1 to 2 meters': 1,'Height_or_Length_1 to 2 meters': 1}

    recommendations = recommend_plants_priority(input_values,data_cosine, cluster_means_df, data)
    if recommendations is not None:
        list_plant = recommendations.index.tolist()
    extracted_rows = plant_data.loc[list_plant]
    extracted_rows.rename(columns = {'Scientific_Name':'Scientific Name', 'Common_Name':'Common Name', 'Regional_Name': 'Regional Name', 'Primary_Grown_for': 'Primary Grown for', 'Flowering_Season': 'Flowering Season', 'Foliage_Color': 'Foliage Color','Height_or_Length': 'Height or Length', 'Spread_or_Width': 'Spread or Width', 'Plant_Form':'Plant Form','Special_Feature': 'Special Feature', 'Plant_Description': 'Plant Description', 'Growing_Tips' : 'Growing Tips'}, inplace = True) 
    
    '''
    plants = []
    plants.append(input_feature)
    for index, row in extracted_rows.iterrows():
        plant_item = []
        for col in extracted_rows.columns:
            if(str(row[col]).lower() !=  "nan".lower()):
                plant_item.append(str(col)+ " : "+ str(row[col]))
        plants.append(plant_item)
    '''
    
    plants = []
    for index, row in extracted_rows.iterrows():
        plant_dict_item = []
        for col in extracted_rows.columns:
            if(str(row[col]).lower() !=  "nan".lower()):
                plant_dict_item.append("<b>"+str(col)+ "</b> : "+ "<thin>"+str(row[col])+"</thin>")
        plants.append(plant_dict_item)
        
    images = []
    images_flag = []
    for index, row in extracted_rows.iterrows():
        image_list = extract_image_list(row['Scientific Name'])
        
        if len(image_list)>0:
            images_flag.append(True)
        else:
            images_flag.append(False)
            
        alt_len = 5-len(image_list)
        if(alt_len>0):
            for j in range(alt_len):
                image_list.append('alternate.png')
        images.append(image_list)
    total = len(plants)
    
    
    #return render_template('index.html', headings=headings, data=data)
    return render_template('index.html', plantsimage=plants, input_feature = new_feature, images = images, images_flag = images_flag, total=total)

@app.route('/medical')
def medical():
    return render_template("medical.html")

@app.route("/predict-medical", methods=['POST'])
def predictMedical():
    features = []
    if "diseases" in request.form:
        disease = request.form.getlist('diseases')
        #print('Hello world!', file=sys.stderr)
        #print(light, file=sys.stderr)
        features = features + disease

        
    if "category" in request.form:
        category = request.form.getlist('category')
        features = features + category
        
        
    input_values = {}
    for feature in features:
        input_values[feature] = 1
        
    input_feature = []
    input_feature.append("<b> Plant Features </b>")
    for feature in features:
        input_feature.append(feature)
    
    #input_values = {'Category_Bamboos': 1,'Light_Sun growing': 1,'Light_Semi shade': 1, 'Water_Normal': 1,'Plant_Form_Upright or Erect' : 1,'Primarily_Grown_for_Flowers': 1,'Flowering_Season_Year-around flowering': 1,'Spread_or_Width_1 to 2 meters': 1,'Height_or_Length_1 to 2 meters': 1}

    recommendations = recommend_medical_plants_cosine(input_values,data_cosine_med)
    list_plant=[]
    if recommendations is not None:
        list_plant = recommendations.index.tolist()
    extracted_rows = plant_medical_data.loc[list_plant]
    #extracted_rows.rename(columns = {'Scientific_Name':'Scientific Name', 'Common_Name':'Common Name', 'Regional_Name': 'Regional Name', 'Primary_Grown_for': 'Primary Grown for', 'Flowering_Season': 'Flowering Season', 'Foliage_Color': 'Foliage Color','Height_or_Length': 'Height or Length', 'Spread_or_Width': 'Spread or Width', 'Plant_Form':'Plant Form','Special_Feature': 'Special Feature', 'Plant_Description': 'Plant Description', 'Growing_Tips' : 'Growing Tips'}, inplace = True) 
    
    df=extracted_rows
    curated_zero = df[df['Curated'] == 0]
    curated_non_zero = df[df['Curated'] != 0]
    df_reordered = pd.concat([curated_zero, curated_non_zero])
    df_reordered = df_reordered.loc[:,['Med_Plant_Use','Scientific_Name','Med_Kingdom','Med_Family', 'Med_Common_Name', 'Med_Synonymous_names', 'Med_System_of_medicine', 'Category', 
       'Light', 'Water', 'Primary_Grown_for', 'Flowering_Season',
       'Foliage_Color', 'Height_or_Length', 'Spread_or_Width', 'Plant_Form',
       'Lifespan', 'Special_Feature', 'Plant_Description', 'Growing_Tips']]
    df_reordered.rename(columns = {'Med_Plant_Use':'Plant Use', 'Scientific_Name':'Scientific Name', 'Med_Kingdom': 'Kingdom', 'Med_Family': 'Family', 'Med_Common_Name': 'Common Name', 'Med_Synonymous_names': 'Synonymous names','Med_System_of_medicine': 'System of medicine', 'Common_Name':'Common Name', 'Regional_Name': 'Regional Name', 'Primary_Grown_for': 'Primary Grown for', 'Flowering_Season': 'Flowering Season', 'Foliage_Color': 'Foliage Color','Height_or_Length': 'Height or Length', 'Spread_or_Width': 'Spread or Width', 'Plant_Form':'Plant Form','Special_Feature': 'Special Feature', 'Plant_Description': 'Plant Description', 'Growing_Tips' : 'Growing Tips'}, inplace = True) 

    plants_feature = []
    plants_feature.append(input_feature)
    plants = []
    
    for index, row in df_reordered.iterrows():
        plant_item = []
        for col in df_reordered.columns:
            if(str(row[col]).lower() !=  "nan".lower()):
                plant_item.append("<b>"+str(col)+ "</b> : "+ "<thin>"+str(row[col])+"</thin>")
        plants.append(plant_item)
        
    images = []
    images_flag = []
    for index, row in df_reordered.iterrows():
        image_list = extract_medical_image_list(row['Scientific Name'])
        
        if len(image_list)>0:
            images_flag.append(True)
        else:
            images_flag.append(False)
            
        alt_len = 5-len(image_list)
        if(alt_len>0):
            for j in range(alt_len):
                image_list.append('alternate.png')
        images.append(image_list)
    total = len(plants)
    
    #return render_template('index.html', headings=headings, data=data)
    return render_template('medical.html', planting=plants, images = images, images_flag = images_flag, total=total)
    
@app.route('/personalise')
def personalise():
    return render_template("personalised.html")

@app.route("/personalised-plant", methods=['POST'])
def predictPersonalise():
    plants = []
    if "personalise" in request.form:
        plant_names = request.form.getlist('personalise')
        for plant_name in plant_names:
        #print('Hello world!', file=sys.stderr)
        #print(light, file=sys.stderr)
            list_plant = recommend_plant_cn(plant_name)
            extracted_rows = df_temp.loc[list_plant]
            extracted_rows.rename(columns = {'Scientific_Name':'Scientific Name', 'Common_Name':'Common Name', 'Regional_Name': 'Regional Name', 'Primary_Grown_for': 'Primary Grown for', 'Flowering_Season': 'Flowering Season', 'Foliage_Color': 'Foliage Color','Height_or_Length': 'Height or Length', 'Spread_or_Width': 'Spread or Width', 'Plant_Form':'Plant Form','Special_Feature': 'Special Feature', 'Plant_Description': 'Plant Description', 'Growing_Tips' : 'Growing Tips'}, inplace = True) 
            
        
            for index, row in extracted_rows.iterrows():
                plant_item = []
                for col in extracted_rows.columns:
                    if(str(row[col]).lower() !=  "nan".lower()):
                        plant_item.append("<b>"+str(col)+ "</b> : "+ "<thin>"+str(row[col])+"</thin>")
                plants.append(plant_item)
                
    images = []
    images_flag = []
    for index, row in extracted_rows.iterrows():
        image_list = extract_image_list(row['Scientific Name'])
        
        if len(image_list)>0:
            images_flag.append(True)
        else:
            images_flag.append(False)
            
        alt_len = 5-len(image_list)
        if(alt_len>0):
            for j in range(alt_len):
                image_list.append('alternate.png')
        images.append(image_list)
    total = len(plants)
    
    
        
        
    return render_template('personalised.html', plants=plants, images = images, images_flag = images_flag, total=total)



#%% 
# main
if __name__ == "__main__":
    app.run(debug=True)
