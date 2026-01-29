# import prediction_helper
#
# # 1. Define a sample input dictionary (simulating a user input from the app)
# # This represents a "High Income, Metro, Brand Conscious" customer
# sample_input = {
#     "age": 30,
#     "gender": "Female",
#     "zone": "Metro",
#     "occupation": "Working Professional",
#     "income_levels": "> 35L",
#     "consume_frequency(weekly)": "5-7 times",
#     "current_brand": "Established",
#     "preferable_consumption_size": "Large (1L)",
#     "awareness_of_other_brands": "more than 4",
#     "reasons_for_choosing_brands": "Brand Reputation",
#     "flavor_preference": "Exotic",
#     "purchase_channel": "Online",
#     "packaging_preference": "Premium",
#     "health_concerns": "High (Very Health Conscious)",
#     "typical_consumption_situations": "Social (Parties)"
# }
#
# print("--- Starting Prediction Test ---")
#
# # 2. Call the function
# try:
#     price_range, probabilities = prediction_helper.predict_price(sample_input)
#
#     # 3. Print Results
#     print("\n✅ Prediction Successful!")
#     print(f"🎯 Recommended Price Range: {price_range}")
#     print("\n📊 Probabilities:")
#     for price, prob in probabilities.items():
#         print(f"   - {price}: {prob:.4f}")
#
# except Exception as e:
#     print(f"\n❌ Error during prediction: {e}")
#     # Optional: Print full traceback if needed
#     import traceback
#
#     traceback.print_exc()
#
# print("\n--- End of Test ---")

import prediction_helper

# TEST CASE 2: The "Budget-Conscious Student"
# - Low Income (<10L)
# - Rural Area
# - Priortizes "Price"
# - Low consumption frequency

student_input = {
    "age": 22,
    "gender": "Male",
    "zone": "Rural",
    "occupation": "Student",
    "income_levels": "<10L",
    "consume_frequency(weekly)": "0-2 times",
    "current_brand": "Newcomer",
    "preferable_consumption_size": "Small (250ml)",
    "awareness_of_other_brands": "0 to 1",
    "reasons_for_choosing_brands": "Price",
    "flavor_preference": "Traditional",
    "purchase_channel": "Retail Store",
    "packaging_preference": "Simple",
    "health_concerns": "Low (Not very concerned)",
    "typical_consumption_situations": "Casual (At home)"
}

print("--- Starting Student Prediction Test ---")

try:
    price_range, probabilities = prediction_helper.predict_price(student_input)

    print("\n✅ Prediction Successful!")
    print(f"🎯 Recommended Price Range: {price_range}")
    print("\n📊 Probabilities:")
    for price, prob in probabilities.items():
        print(f"   - {price}: {prob:.4f}")

except Exception as e:
    print(f"\n❌ Error: {e}")