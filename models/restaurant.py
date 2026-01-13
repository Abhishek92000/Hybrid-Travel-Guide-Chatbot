import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv(r"D:\travel\dataset\Restaurant_Dataset.csv")
# print(df)

df["Area_Code"] = df["Area Part"].apply(lambda x:1 if 'West' in x else 0)
# print(df)
def restaurant_label (row):
    price = row["Price for Two (INR)"]
    rating = row['Rating']

    if price > 1500:
        return  1 if rating >= 4.0 else 0
    elif price > 800:
        return  1 if rating >= 3.8 else 0
    else:
        return 1 if rating <= 3.7 else 0

df["Is_Recommended"] = df.apply(restaurant_label, axis=1)

X = df[["Cuisine Type", "Rating", "Price for Two (INR)", "Area_Code"]]
y = df["Is_Recommended"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Rating", "Price for Two (INR)", "Area_Code"]),
        ("cat", OneHotEncoder(handle_unknown='ignore'), ["Cuisine Type"])
    ]
)

res_pipeline = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", LogisticRegression())
])

res_pipeline.fit(X, y)

def find_best_restaurant(total_budget, group_size, preferred_area, preferred_cuisine):
    area_bit = 1 if 'West' in preferred_area else 0

    temp_df = df.copy()
    temp_df['Price_Per_Person'] = temp_df['Price for Two (INR)'] / 2
    temp_df['Total_Group_Cost'] = temp_df['Price_Per_Person'] * group_size

    candidates = temp_df[
        (temp_df["Total_Group_Cost"] <= total_budget) &
        (temp_df["Area_Code"] == area_bit) &
        (temp_df["Cuisine Type"] == preferred_cuisine)
    ].copy()

    if candidates.empty:
        return f"Budget too low! For {group_size} people, you need at least ₹{temp_df[temp_df['Cuisine Type']==preferred_cuisine]['Total_Group_Cost'].min()} for {preferred_cuisine}."

    features = candidates[["Cuisine Type", "Rating", "Price for Two (INR)", "Area_Code"]]
    candidates['Match_Score'] = res_pipeline.predict_proba(features)[:, 1]

    return candidates.sort_values(by="Match_Score", ascending=False)[
        ['Restaurant Name', 'Rating', 'Price for Two (INR)', 'Total_Group_Cost', 'Match_Score']
    ].head(3)

# print(find_best_restaurant(total_budget=1500, group_size=1, preferred_area="Jogeshwari West", preferred_cuisine="Budget"))