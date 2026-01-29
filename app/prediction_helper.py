import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'app/model_data.pkl'
COLUMNS_PATH = 'app/model_columns.pkl'


def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        model_cols = joblib.load(COLUMNS_PATH)
        return model, model_cols
    except FileNotFoundError:
        return None, None


def calculate_features(input_dict):
    # (Keep your existing calculate_features logic exactly the same as before)
    zone_map = {'Rural': 1, 'Semi-Urban': 2, 'Urban': 3, 'Metro': 4}
    income_map_score = {'<10L': 1, '10L - 15L': 2, '16L - 25L': 3, '26L - 35L': 4, '> 35L': 5, 'Not Reported': 0}
    z_score = zone_map.get(input_dict['zone'], 1)
    i_score = income_map_score.get(input_dict['income_levels'], 0)
    zas_score = z_score * i_score

    is_not_established = input_dict['current_brand'] != 'Established'
    is_price_quality = input_dict['reasons_for_choosing_brands'] in ['Price', 'Quality']
    bsi = 1 if (is_not_established and is_price_quality) else 0

    freq_map_score = {'0-2 times': 1, '3-4 times': 2, '5-7 times': 3}
    aware_map_score = {'0 to 1': 1, '2 to 4': 2, 'more than 4': 3}
    f_score = freq_map_score.get(input_dict['consume_frequency(weekly)'], 1)
    a_score = aware_map_score.get(input_dict['awareness_of_other_brands'], 1)
    cf_ab_score = f_score / (a_score + f_score)

    return zas_score, bsi, cf_ab_score


def prepare_input(input_dict, model_columns):
    # (Keep your existing prepare_input logic exactly the same as before)
    gender_map = {'Male': 'M', 'Female': 'F'}
    age_map = {'18-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56-70': 4, '70+': 5}
    income_map = {'<10L': 0, '10L - 15L': 1, '16L - 25L': 2, '26L - 35L': 3, '> 35L': 4, 'Not Reported': 5}
    health_map = {'Low (Not very concerned)': 0, 'Medium (Moderately Health Conscious)': 1,
                  'High (Very Health Conscious)': 2}
    freq_map = {'0-2 times': 0, '3-4 times': 1, '5-7 times': 2}
    size_map = {'Small (250ml)': 0, 'Medium (500ml)': 1, 'Large (1L)': 2}

    zas, bsi, cf_ab = calculate_features(input_dict)

    input_df = pd.DataFrame([{
        'age_group': age_map.get(input_dict['age_group'], 0),
        'income_levels': income_map.get(input_dict['income_levels'], 0),
        'health_concerns': health_map.get(input_dict['health_concerns'], 0),
        'consume_frequency(weekly)': freq_map.get(input_dict['consume_frequency(weekly)'], 0),
        'preferable_consumption_size': size_map.get(input_dict['preferable_consumption_size'], 0),
        'gender': gender_map.get(input_dict['gender'], 'M'),
        'zone': input_dict['zone'],
        'occupation': input_dict['occupation'],
        'current_brand': input_dict['current_brand'],
        'awareness_of_other_brands': input_dict['awareness_of_other_brands'],
        'reasons_for_choosing_brands': input_dict['reasons_for_choosing_brands'],
        'flavor_preference': input_dict['flavor_preference'],
        'purchase_channel': input_dict['purchase_channel'],
        'packaging_preference': input_dict['packaging_preference'],
        'typical_consumption_situations': input_dict['typical_consumption_situations'],
        'zas_score': zas,
        'bsi': bsi,
        'cf_ab_score': cf_ab
    }])

    input_df = pd.get_dummies(input_df)
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    return final_df


def predict_price(input_dict):
    model, cols = load_model_assets()
    if model is None:
        return "Error", {}

    # Handle Age Group Logic
    age_val = input_dict['age']
    if age_val <= 25:
        group = '18-25'
    elif age_val <= 35:
        group = '26-35'
    elif age_val <= 45:
        group = '36-45'
    elif age_val <= 55:
        group = '46-55'
    elif age_val <= 70:
        group = '56-70'
    else:
        group = '70+'
    input_dict['age_group'] = group

    df = prepare_input(input_dict, cols)

    # Get Prediction
    prediction = model.predict(df)[0]

    # Get Probabilities (Confidence)
    try:
        probs = model.predict_proba(df)[0]
    except:
        probs = [0, 0, 0, 0, 0]  # Fallback if model doesn't support probability

    price_labels = {0: '50-100', 1: '100-150', 2: '150-200', 3: '200-250', 4: '>250'}

    # Map probabilities to labels for the chart
    prob_dict = {price_labels[i]: round(prob, 4) for i, prob in enumerate(probs)}

    return price_labels.get(prediction, "Unknown"), prob_dict