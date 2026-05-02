import sys
import csv
csv.field_size_limit(1000000000)

import pandas as pd
import string
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# LOAD DATA
df = pd.read_csv("../Dataset/zomato.csv", engine='python', on_bad_lines='skip')

# FIX COST COLUMN
if 'approx_cost(for two people)' in df.columns:
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str)
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '')
    df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

    df.rename(columns={
        'approx_cost(for two people)': 'cost'
    }, inplace=True)

# CLEAN
df.dropna(subset=['name', 'rate', 'cuisines', 'reviews_list', 'location'], inplace=True)
df.drop_duplicates(inplace=True)

df = df[df['rate'] != 'NEW']
df = df[df['rate'] != '-']

df['rate'] = df['rate'].str.replace('/5', '', regex=False)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
df.dropna(subset=['rate'], inplace=True)

df = df.sample(5000, random_state=42).reset_index(drop=True)

# TEXT CLEAN
df['reviews_list'] = df['reviews_list'].str.lower()

def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    return ""

df['reviews_list'] = df['reviews_list'].apply(remove_punctuation)

# MODEL
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['reviews_list'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# RECOMMEND FUNCTION
def recommend_model(name, location):
    if not name or not location:
        return [{"name": "No restaurant found", "rate": "", "cuisines": ""}]

    # try filtering by location
    df_loc = df[df['location'].str.contains(location, case=False, na=False)]

# 🔥 ADD THIS (IMPORTANT)
    if df_loc.empty:
        df_loc = df

    if df_loc.empty:
        return [{"name": "No restaurant found in this location", "rate": "", "cuisines": ""}]

    matches = df_loc[df_loc['name'].str.contains(name, case=False, na=False)]

    if matches.empty:
        return [{"name": "No restaurant found", "rate": "", "cuisines": ""}]

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    indices = [i[0] for i in sim_scores]

    result = df.iloc[indices][['name', 'rate', 'cuisines', 'cost']].to_dict('records')

    # remove duplicates
    seen = set()
    final = []
    for r in result:
        if r['name'] not in seen:
            seen.add(r['name'])
            final.append(r)

    return final

# AUTO LOCATION API
@app.route('/get_location')
def get_location():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"

        headers = {
            "User-Agent": "restaurant-app"
        }

        res = requests.get(url, headers=headers, timeout=5)

        if res.status_code != 200:
            return jsonify({"location": ""})

        data = res.json()

        address = data.get('address', {})

        location = (
            address.get('suburb')
            or address.get('city')
            or address.get('town')
            or address.get('village')
            or ""
        )

        return jsonify({"location": location})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"location": ""})

# ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        name = request.form.get('restaurant')
        location = request.form.get('location')

        result = recommend_model(name, location)
        print("RESULT:", result) 

        return render_template('result.html', restaurants=result)

    return render_template('recommend.html')

if __name__ == '__main__':
    app.run(debug=True)