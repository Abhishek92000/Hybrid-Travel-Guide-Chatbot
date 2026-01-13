import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv(r"D:\travel\dataset\Hotel_Dataset csv")
# print(df)

df["Area_Code"]= df["Area Part"].apply(lambda x: 1 if 'West' in x else 0)
# print(df["Area_Code"])

def smart_label(row):
    # Luxury rooms are allowed to be expensive but must have high ratings
    if row['Room Type'] in ['Executive Suite', 'Luxury Suite', '3 BHK Apartment', 'Executive Room']:
        return 1 if (row['Rating'] >= 4.0 and row['Price (INR)'] <= 10000) else 0
    # Normal/Standard rooms
    elif row['Room Type'] in ['Deluxe Room', 'Standard Room', 'Classic Room']:
        return 1 if (row['Rating'] >= 3.8 and row['Price (INR)'] <= 5000) else 0
    # Budget/Dorm rooms
    else:
        return 1 if (row['Rating'] <= 3.5 and row['Price (INR)'] <= 2500) else 0

df["Is_Recommended"] = df.apply(smart_label, axis=1)

# print(df)
X = df[["Room Type",'Rating', "Price (INR)", "Area_Code"]]
y = df["Is_Recommended"]

num_cols = ['Rating', "Price (INR)", "Area_Code"]
cat_cols = ["Room Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression())
    ]
)

pipeline.fit(X,y)


def find_best_hotel(user_budget, nights, preferred_area, room_type):
    budget_per_night = user_budget / nights
    area_input = preferred_area.lower()

    # 1. Determine the Area Filter
    if 'west' in area_input:
        # User specified West
        area_filter = [1]
        area_label = "Jogeshwari West"
    elif 'east' in area_input:
        # User specified East
        area_filter = [0]
        area_label = "Jogeshwari East"
    else:
        # User just said "Jogeshwari" -> Show both (0 and 1)
        area_filter = [0, 1]
        area_label = "all of Jogeshwari"

    # 2. Apply the Filter using .isin() for the Area_Code
    candidates = df[
        (df["Price (INR)"] <= budget_per_night) &
        (df["Area_Code"].isin(area_filter)) &
        (df["Room Type"] == room_type)
        ].copy()

    # 3. Handle Empty Results
    if candidates.empty:
        return f"I'm sorry, I couldn't find any {room_type}s in {area_label} under ₹{budget_per_night} per night."

    # 4. Rank and Return
    cand_features = candidates[["Room Type", 'Rating', "Price (INR)", "Area_Code"]]
    candidates['Match_Score'] = pipeline.predict_proba(cand_features)[:, 1]

    return candidates.sort_values(by='Match_Score', ascending=False).head(3)

# print(find_best_hotel(3000, 1, "Jogeshwari west","Standard Room"))



