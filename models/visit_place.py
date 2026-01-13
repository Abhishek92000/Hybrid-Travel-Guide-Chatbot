import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df_p = pd.read_csv(r"D:\travel\dataset\spot.csv")

df_p['Is_Must_Visit'] = ((df_p['Rating'] >= 4.2) | (df_p['Entry_Fee'] <= 50)).astype(int)
# print(df_p)

X = df_p[["Type", "Rating", "Entry_Fee", "Dist_from_Jogeshwari_KM"]]
y = df_p['Is_Must_Visit']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["Rating", "Entry_Fee", "Dist_from_Jogeshwari_KM"]),
    ("cat", OneHotEncoder(handle_unknown='ignore'), ["Type"])
])

place_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression())
])
place_pipeline.fit(X, y)


def calculate_transport_cost(dist, mode, group_size):
    if mode == "Auto":
        # 1. Determine how many autos are needed (Max 3 people per auto)
        num_autos = -(-group_size // 3)  # Ceiling division

        # 2. Define Fare Components
        base_fare = 23.0  # Minimum fare for first 1.5 km
        rate_per_km = 18.0  # Rate for every km after 1.5 km

        if dist <= 1.5:
            total_fare_per_auto = base_fare
        else:
            # Calculate extra distance and multiply by rate
            extra_dist = dist - 1.5
            total_fare_per_auto = base_fare + (extra_dist * rate_per_km)

        return num_autos * total_fare_per_auto

    elif mode == "Private_Car":
        # Private cars (Cabs) usually carry 4 people
        num_cars = -(-group_size // 4)
        return num_cars * (800 + (dist * 35))

    else:  # Public Bus
        return group_size * 20


# 4. Final Recommender
def suggest_trip(total_budget, group_size, transport_mode, preferred_type=""):
    # 1. Predict match scores using your Logistic Regression model
    df_p['Match_Score'] = place_pipeline.predict_proba(X)[:, 1]

    # 2. Calculate transport and total costs
    df_p['Predicted_Transport_Cost'] = df_p['Dist_from_Jogeshwari_KM'].apply(
        lambda x: calculate_transport_cost(x, transport_mode, group_size)
    )
    df_p['Total_Trip_Cost'] = df_p['Predicted_Transport_Cost'] + (df_p['Entry_Fee'] * group_size)

    # 3. Filter by budget first
    options = df_p[df_p['Total_Trip_Cost'] <= total_budget].copy()

    # 4. Search by Type (Case-insensitive)
    if preferred_type.strip():
        # Filter strictly by the type provided (e.g., 'Beach')
        options = options[options['Type'].str.contains(preferred_type, case=False)]

    # 5. Handle empty results or general search
    if options.empty:
        return pd.DataFrame()

        # 6. Always return top 3 based on ML Match_Score
    return options.sort_values(by="Match_Score", ascending=False).head(3)


# TEST: Group of 4 people, Total Budget ₹1000, using Auto
# print(suggest_trip(4000, 8, "", ""))

# print(df_p["Place_Name"])