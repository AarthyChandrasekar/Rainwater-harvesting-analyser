from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from database import initialize_database, create_user, verify_user

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# =====================================================
# FLASK APPLICATION
# =====================================================

app = Flask(__name__)

app.secret_key = "rainwater-harvesting-project-secret-key"

initialize_database()


# =====================================================
# DATASET PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "daily_rainfall_2010_2024_320_locations.csv"
)


# Dataset will be loaded only once
rainfall_data = None


# =====================================================
# LOAD RAINFALL DATA
# =====================================================

def load_rainfall_data():

    global rainfall_data

    if rainfall_data is None:

        if not os.path.exists(DATASET_PATH):

            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}"
            )

        rainfall_data = pd.read_csv(
            DATASET_PATH
        )

        # Convert TIME column
        rainfall_data["TIME"] = pd.to_datetime(
            rainfall_data["TIME"],
            errors="coerce"
        )

        # Remove invalid dates
        rainfall_data = rainfall_data.dropna(
            subset=["TIME"]
        )

        # Convert rainfall to numeric
        rainfall_data["RAINFALL"] = pd.to_numeric(
            rainfall_data["RAINFALL"],
            errors="coerce"
        )

        # Remove invalid rainfall values
        rainfall_data = rainfall_data.dropna(
            subset=["RAINFALL"]
        )

        # Create month and year columns
        rainfall_data["MONTH"] = (
            rainfall_data["TIME"].dt.month
        )

        rainfall_data["YEAR"] = (
            rainfall_data["TIME"].dt.year
        )

    return rainfall_data


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return redirect(
        url_for("login")
    )


# =====================================================
# REGISTER
# =====================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form["name"].strip()

        email = request.form["email"].strip()

        phone = request.form["phone"].strip()

        password = request.form["password"]

        confirm_password = request.form[
            "confirm_password"
        ]


        if not name or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all required fields."
            )


        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )


        success = create_user(
            name,
            email,
            phone,
            password
        )


        if not success:

            return render_template(
                "register.html",
                error="Email already registered."
            )


        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# =====================================================
# LOGIN
# =====================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"].strip()

        password = request.form["password"]


        user = verify_user(
            email,
            password
        )


        if user:

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            return redirect(
                url_for("dashboard")
            )


        return render_template(
            "login.html",
            error="Invalid email or password."
        )


    return render_template(
        "login.html"
    )


# =====================================================
# DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "dashboard.html",
        user_name=session["user_name"]
    )


# =====================================================
# PROPERTY DETAILS
# =====================================================

@app.route(
    "/property",
    methods=["GET", "POST"]
)
def property_details():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        property_data = {

            "location":
                request.form.get("location"),

            "roof_area":
                request.form.get("roof_area"),

            "roof_type":
                request.form.get("roof_type"),

            "people":
                request.form.get("people"),

            "daily_water":
                request.form.get("daily_water"),

            "purpose":
                request.form.get("purpose")
        }


        # Save latest property details
        session["property"] = property_data

        session.modified = True


        return redirect(
            url_for("analysis")
        )


    return render_template(
        "property.html"
    )


# =====================================================
# ANALYSIS PAGE
# =====================================================

@app.route("/analysis")
def analysis():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    property_data = session.get(
        "property"
    )


    if not property_data:

        return redirect(
            url_for("property_details")
        )


    return render_template(
        "analysis.html",
        property_data=property_data
    )


# =====================================================
# RAINFALL ANALYSIS API
# =====================================================

@app.route("/api/analyze")
def analyze():

    # -------------------------------------------------
    # LOGIN CHECK
    # -------------------------------------------------

    if "user_id" not in session:

        return jsonify({
            "error": "User not logged in"
        }), 401


    # -------------------------------------------------
    # GET PROPERTY DATA
    # -------------------------------------------------

    property_data = session.get(
        "property"
    )


    if not property_data:

        return jsonify({
            "error": "Property information not found."
        }), 400


    try:

        # =================================================
        # LOAD DATASET
        # =================================================

        df = load_rainfall_data().copy()


        df = df.dropna(
            subset=["TIME"]
        )


        # =================================================
        # USER LOCATION
        # =================================================

        user_location = (
            property_data["location"]
            .strip()
            .lower()
        )


        # =================================================
        # TAMIL NADU CITY COORDINATES
        # =================================================

        location_coordinates = {

            "chennai":
                (13.0827, 80.2707),

            "coimbatore":
                (11.0168, 76.9558),

            "madurai":
                (9.9252, 78.1198),

            "tiruchirappalli":
                (10.7905, 78.7047),

            "trichy":
                (10.7905, 78.7047),

            "salem":
                (11.6643, 78.1460),

            "tirunelveli":
                (8.7139, 77.7567),

            "erode":
                (11.3410, 77.7172),

            "vellore":
                (12.9165, 79.1325),

            "thanjavur":
                (10.7870, 79.1378),

            "thoothukudi":
                (8.7642, 78.1348),

            "nagercoil":
                (8.1833, 77.4119)
        }


        # =================================================
        # CHECK LOCATION
        # =================================================

        if user_location not in location_coordinates:

            return jsonify({

                "error":
                    f"Location '{property_data['location']}' "
                    "is not currently supported. "
                    "Please enter a supported Tamil Nadu city."

            }), 400


        # =================================================
        # USER COORDINATES
        # =================================================

        user_lat, user_lon = (
            location_coordinates[user_location]
        )


        # =================================================
        # FIND NEAREST DATASET LOCATION
        # =================================================

        locations = (
            df[
                [
                    "LATITUDE",
                    "LONGITUDE"
                ]
            ]
            .drop_duplicates()
            .copy()
        )


        locations["distance"] = np.sqrt(

            (
                locations["LATITUDE"]
                - user_lat
            ) ** 2

            +

            (
                locations["LONGITUDE"]
                - user_lon
            ) ** 2

        )


        nearest_location = (
            locations
            .sort_values("distance")
            .iloc[0]
        )


        selected_lat = float(
            nearest_location["LATITUDE"]
        )


        selected_lon = float(
            nearest_location["LONGITUDE"]
        )


        # =================================================
        # FILTER DATA FOR SELECTED LOCATION
        # =================================================

        location_df = df[

            (df["LATITUDE"] == selected_lat)

            &

            (df["LONGITUDE"] == selected_lon)

        ].copy()


        if location_df.empty:

            return jsonify({

                "error":
                    "No rainfall data found for "
                    "the selected location."

            }), 404


        # =================================================
        # PROPERTY VALUES
        # =================================================

        roof_area = float(
            property_data["roof_area"]
        )


        roof_type = (
            property_data["roof_type"]
        )


        daily_requirement = float(
            property_data["daily_water"]
        )


        # =================================================
        # ROOF RUNOFF COEFFICIENTS
        # =================================================

        roof_coefficients = {

            "Metal": 0.90,

            "Concrete": 0.80,

            "Tile": 0.75,

            "Asbestos": 0.70,

            "Other": 0.65
        }


        runoff_coefficient = (
            roof_coefficients.get(
                roof_type,
                0.75
            )
        )


        # =================================================
        # HISTORICAL AVERAGE RAINFALL
        # =================================================

        historical_average = float(

            location_df[
                "RAINFALL"
            ].mean()

        )


        # =================================================
        # YEARLY RAINFALL
        # =================================================

        yearly = (

            location_df
            .groupby("YEAR")[
                "RAINFALL"
            ]
            .sum()
            .reset_index()

        )
        total_observations = len(location_df)

        total_locations = df[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]

        highest_rainfall_year = int(
    yearly.loc[yearly["RAINFALL"].idxmax(), "YEAR"]
)

        highest_rainfall_value = float(
    yearly["RAINFALL"].max()
)

        lowest_rainfall_year = int(
    yearly.loc[yearly["RAINFALL"].idxmin(), "YEAR"]
)

        lowest_rainfall_value = float(
    yearly["RAINFALL"].min()
)

        # =================================================
        # MONTHLY RAINFALL ANALYSIS
        # =================================================

        monthly_totals = (

            location_df

            .groupby(
                [
                    "YEAR",
                    "MONTH"
                ]
            )["RAINFALL"]

            .sum()

            .reset_index()

        )


        monthly_average = (

            monthly_totals

            .groupby("MONTH")[
                "RAINFALL"
            ]

            .mean()

            .reindex(
                range(1, 13),
                fill_value=0
            )

        )


        # =================================================
        # MONTHLY HARVESTING FROM HISTORICAL RAINFALL
        # =================================================

        monthly_harvesting_series = (

            monthly_average

            * roof_area

            * runoff_coefficient

        )


        # =================================================
        # MONTHLY RAINFALL JSON DATA
        # =================================================

        monthly_rainfall = [

            {

                "month":
                    int(month),

                "rainfall":
                    round(
                        float(rainfall),
                        2
                    )

            }

            for month, rainfall
            in monthly_average.items()

        ]


        # =================================================
        # MONTHLY HARVESTING JSON DATA
        # =================================================

        monthly_harvesting_data = [

            {

                "month":
                    int(month),

                "harvesting":
                    round(
                        float(harvesting),
                        2
                    )

            }

            for month, harvesting
            in monthly_harvesting_series.items()

        ]


        # =================================================
        # FUTURE RAINFALL PREDICTION
        # =================================================

        X = yearly[
            ["YEAR"]
        ]


        y = yearly[
            "RAINFALL"
        ]


        model = RandomForestRegressor(

            n_estimators=100,

            random_state=42

        )


        model.fit(
            X,
            y
        )


        # =================================================
        # PREDICT 2026
        # =================================================

        prediction_year = 2026


        predicted_rainfall = float(

            model.predict(

                pd.DataFrame({

                    "YEAR":
                        [prediction_year]

                })

            )[0]

        )


        predicted_rainfall = max(
            0,
            predicted_rainfall
        )
        yearly_rainfall = [
    {
        "year": int(row["YEAR"]),
        "rainfall": round(float(row["RAINFALL"]), 2)
    }
    for _, row in yearly.iterrows()
]

        yearly_rainfall.append({
    "year": prediction_year,
    "rainfall": round(float(predicted_rainfall), 2)
})

        # =================================================
        # ANNUAL HARVESTING
        # =================================================

        annual_harvesting = (

            predicted_rainfall

            * roof_area

            * runoff_coefficient

        )


        # =================================================
        # MONTHLY HARVESTING BASED ON PREDICTION
        # =================================================

        predicted_monthly_harvesting = (

            annual_harvesting
            / 12

        )


        # =================================================
        # STORAGE REQUIREMENT
        # =================================================

        monthly_requirement = (

            daily_requirement
            * 30

        )


        recommended_tank = min(

            predicted_monthly_harvesting,

            monthly_requirement * 2

        )


        recommended_tank = max(

            1000,

            recommended_tank

        )


        # =================================================
        # CONVERT TO PRACTICAL TANK SIZE
        # =================================================

        if recommended_tank <= 2000:

            recommended_tank = 2000

        elif recommended_tank <= 5000:

            recommended_tank = 5000

        elif recommended_tank <= 10000:

            recommended_tank = 10000

        else:

            recommended_tank = (

                round(
                    recommended_tank / 1000
                )
                * 1000

            )


        # =================================================
        # FEASIBILITY
        # =================================================

        monthly_demand = (

            daily_requirement
            * 30

        )


        if monthly_demand > 0:

            coverage_ratio = (

                predicted_monthly_harvesting
                / monthly_demand

            )

        else:

            coverage_ratio = 0


        score = min(

            100,

            max(

                0,

                round(
                    coverage_ratio * 100
                )

            )

        )


        # =================================================
        # FEASIBILITY LEVEL
        # =================================================

        if score >= 80:

            feasibility = "Excellent"

        elif score >= 60:

            feasibility = "Good"

        elif score >= 40:

            feasibility = "Moderate"

        elif score >= 20:

            feasibility = "Low"

        else:

            feasibility = "Very Low"


        # =================================================
        # PERSONALIZED RECOMMENDATION
        # =================================================

        if score >= 80:

            recommendation = (

                "Your property has excellent "
                "rainwater harvesting potential. "
                "A dedicated storage tank and "
                "proper filtration system are recommended."

            )

        elif score >= 60:

            recommendation = (

                "Your property has good rainwater "
                "harvesting potential. A moderate "
                "storage system can provide useful "
                "water during rainy periods."

            )

        elif score >= 40:

            recommendation = (

                "Your property has moderate harvesting "
                "potential. Rainwater harvesting can "
                "provide useful supplementary water."

            )

        elif score >= 20:

            recommendation = (

                "Your harvesting potential is relatively "
                "low. A smaller storage system may be "
                "more suitable."

            )

        else:

            recommendation = (

                "The predicted rainfall may not provide "
                "enough water to meet your requirement. "
                "Consider rainwater harvesting as a "
                "supplementary water source."

            )


        # =================================================
        # FINAL JSON RESPONSE
        # =================================================

        return jsonify({
            "dataset_summary": {
    "total_observations": total_observations,
    "total_locations": total_locations,
    "data_start_year": 2010,
    "data_end_year": 2024,
    "highest_rainfall_year": highest_rainfall_year,
    "highest_rainfall": round(highest_rainfall_value, 2),
    "lowest_rainfall_year": lowest_rainfall_year,
    "lowest_rainfall": round(lowest_rainfall_value, 2)
},
            # ---------------------------------------------
            # PROPERTY INFORMATION
            # ---------------------------------------------

            "location":
                property_data["location"],

            "roof_area":
                roof_area,

            "roof_type":
                roof_type,

            "people":
                property_data["people"],

            "daily_requirement":
                daily_requirement,

            "purpose":
                property_data["purpose"],


            # ---------------------------------------------
            # DATASET LOCATION
            # ---------------------------------------------

            "dataset_latitude":
                selected_lat,

            "dataset_longitude":
                selected_lon,


            # ---------------------------------------------
            # RAINFALL
            # ---------------------------------------------

            "average_rainfall":
                historical_average,

            "predicted_rainfall":
                predicted_rainfall,

            "prediction_year":
                prediction_year,
            "yearly_rainfall": yearly_rainfall,


            # ---------------------------------------------
            # HARVESTING
            # ---------------------------------------------

            "annual_harvesting":
                annual_harvesting,

            "monthly_harvesting":
                predicted_monthly_harvesting,


            # ---------------------------------------------
            # MONTHLY CHART DATA
            # ---------------------------------------------

            "monthly_rainfall":
                monthly_rainfall,

            "monthly_harvesting_data":
                monthly_harvesting_data,


            # ---------------------------------------------
            # STORAGE
            # ---------------------------------------------

            "recommended_tank":
                recommended_tank,


            # ---------------------------------------------
            # RUNOFF
            # ---------------------------------------------

            "runoff_coefficient":
                runoff_coefficient,


            # ---------------------------------------------
            # FEASIBILITY
            # ---------------------------------------------

            "score":
                score,

            "feasibility":
                feasibility,


            # ---------------------------------------------
            # RECOMMENDATION
            # ---------------------------------------------

            "recommendation":
                recommendation

        })


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        print(
            "Analysis error:",
            str(e)
        )


        return jsonify({

            "error":
                "Unable to perform rainfall analysis.",

            "details":
                str(e)

        }), 500


# =====================================================
# LOGOUT
# =====================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

        
              