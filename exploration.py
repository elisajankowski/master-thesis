# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from pandas.api.types import is_numeric_dtype
from pyprocessmacro import Process
import pingouin

# %%
control = pd.read_excel(
    "Master Thesis Questionnaire analysis.xlsx",
    sheet_name="Control Group",
)
AGE = "Age"
EDUCATION = "Education"
INCOME = "Income"
CLEANED_INCOME = "CLEANED INCOME"
GENDER = "Gender"
LIKELY_LIMITED_SHOE_NEXT_MONTH = (
    "How likely are you to purchase a recycled limited edition shoe in the next month?"
)
PURCHASE_INTENTION_ITEM_1 = "Purchase Intention Item 1"
INTEND_LIMITED_SHOE_NEAR_FUTURE = "To what extent do you intend to buy a recycled limited edition shoe in the near future?"
PURCHASE_INTENTION_ITEM_2 = "Purchase Intention Item 2"
PROBABLE_BUY_LIMITED_SHOE = (
    "How probable is it that you will buy a recycled limited edition shoe?"
)
PURCHASE_INTENTION_ITEM_3 = "Purchase Intention Item 3"

ENVIRONMENTALLY_PERCEIVE_LIMITED_SHOE = "How environmentally friendly do you perceive a recycled limited edition shoe to be?"
CONSUMER_PERCEPTION_ITEM_1 = "Consumer Perception Item 1"
INFLUENCE_PERCEPTION_LIMITED_SHOE = "To what extent does a recycled limited edition shoe influence your perception of the brand's commitment to sustainability?"
CONSUMER_PERCEPTION_ITEM_2 = "Consumer Perception Item 2"
LIKELY_RECOMMEND_LIMITED_SHOE = "How likely are you to recommend a recycled limited edition shoe to others based on its environmental impact?"
CONSUMER_PERCEPTION_ITEM_3 = "Consumer Perception Item 3"

INTEREST_LIMITED_SHOE = (
    "How interested are you in a limited edition shoe made from recycled materials?"
)
LIMITED_EDITION_ITEM_1 = "Limited Edition Item 1"
IMPORTANT_LIMITED_SHOE = "To what extent do you believe that using recycled materials in limited edition shoes is important for environmental sustainability?"
LIMITED_EDITION_ITEM_2 = "Limited Edition Item 2"
INFLUENCE_LIMITED_SHOE = "How much influence does the use of recycled materials have on your decision to buy limited edition shoes?"
LIMITED_EDITION_ITEM_3 = "Limited Edition Item 3"

LIKELY_PURCHASE_USAIN = "How likely are you to purchase a recycled limited edition shoe endorsed by Usain Bolt who holds a world record?"
ATHLETE_ACHIEVEMENT_ITEM_1_USAIN = "Athlete Achievement Item 1 Usain"
INFLUENCE_PERCEPTION_USAIN = "To what extent does Usain Bolt's world record achievement influence your perception on the quality and value of a recycled limited edition shoe?"
ATHLETE_ACHIEVEMENT_ITEM_2_USAIN = "Athlete Achievement Item 2 Usain"
ENHANCE_APPEAL_USAIN = "Do you believe that Usain Bolt's world record achievement enhances the appeal of a recycled limited edition shoe, making you more likely to consider purchasing it?"
ATHLETE_ACHIEVEMENT_ITEM_3_USAIN = "Athlete Achievement Item 3 Usain"

LIKELY_PURCHASE_DANIELA = "How likely are you to purchase a recycled limited edition shoe endorsed by Daniela Ryf who holds a world record?"
ATHLETE_ACHIEVEMENT_ITEM_1_DANIELA = "Athlete Achievement Item 1 Daniela"
INFLUENCE_PERCEPTION_DANIELA = "To what extent does Daniela Ryf's world record achievement influence your perception on the quality and value of a recycled limited edition shoe?"
ATHLETE_ACHIEVEMENT_ITEM_2_DANIELA = "Athlete Achievement Item 2 Daniela"
ENHANCE_APPEAL_DANIELA = "Do you believe that Daniela Ryf's world record achievement enhances the appeal of a recycled limited edition shoe, making you more likely to consider purchasing it?"
ATHLETE_ACHIEVEMENT_ITEM_3_DANIELA = "Athlete Achievement Item 3 Daniela"

LIKELY_PURCHASE_ELIUD = "How likely are you to purchase a recycled limited edition shoe endorsed by Eliud Kipchoge who had a world record?"
ATHLETE_ACHIEVEMENT_ITEM_1_ELIUD = "Athlete Achievement Item 1 Eliud"
INFLUENCE_PERCEPTION_ELIUD = "To what extent does Eliud Kipchoge's world record achievement influence your perception on the quality and value of a recycled limited edition shoe?"
ATHLETE_ACHIEVEMENT_ITEM_2_ELIUD = "Athlete Achievement Item 2 Eliud"
ENHANCE_APPEAL_ELIUD = "Do you believe that Eliud Kipchoge's world record achievement enhances the appeal of a recycled limited edition shoe, making you more likely to consider purchasing it?"
ATHLETE_ACHIEVEMENT_ITEM_3_ELIUD = "Athlete Achievement Item 3 Eliud"

rename_column_control = {
    "How often do you buy sportswear (sport shoes and sport apparel)?": "Purchase Frequency",
    "How important is sustainability in your fashion choices?": "Importance Sustainability In Fashion",
    LIKELY_LIMITED_SHOE_NEXT_MONTH: PURCHASE_INTENTION_ITEM_1,
    INTEND_LIMITED_SHOE_NEAR_FUTURE: PURCHASE_INTENTION_ITEM_2,
    PROBABLE_BUY_LIMITED_SHOE: PURCHASE_INTENTION_ITEM_3,
    "How much would you pay for a recycled limited edition shoe?": "Willigness To Pay",
    "Please think back to the introduction you’ve read. What is unique about this limited edition shoe?": "Control Question",
    ENVIRONMENTALLY_PERCEIVE_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_1,
    INFLUENCE_PERCEPTION_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_2,
    LIKELY_RECOMMEND_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_3,
    INTEREST_LIMITED_SHOE: LIMITED_EDITION_ITEM_1,
    IMPORTANT_LIMITED_SHOE: LIMITED_EDITION_ITEM_2,
    INFLUENCE_LIMITED_SHOE: LIMITED_EDITION_ITEM_3,
    "To what extent does an athlete's world record achievement influence your perception on the quality and value of a recycled limited edition shoe?": "Control Influence Athlete",
    "What is your age?": AGE,
    "Which gender do you identify most with?": GENDER,
    "What is your highest level of education completed?": EDUCATION,
    "What is your total annual net household income?": INCOME,
    "What is your current location? (country/region)": "Location",
    "Which statement do you most agree with?": "Interest Recycled Sportswear/Shoes ",
}

for k in rename_column_control:
    if k not in control.columns:
        print(f"{k} not in Control Dataframe")

control = control.rename(rename_column_control, axis=1)
control["athlete"] = ""
# %%

ryf = pd.read_excel(
    "Master Thesis Questionnaire analysis.xlsx",
    sheet_name="Daniela Ryf",
)

rename_column_ryf = {
    "How often do you buy sportswear (sport shoes and sport apparel)?": "Purchase Frequency",
    "How important is sustainability in your fashion choices?": "Importance Sustainability In Fashion",
    LIKELY_LIMITED_SHOE_NEXT_MONTH: PURCHASE_INTENTION_ITEM_1,
    INTEND_LIMITED_SHOE_NEAR_FUTURE: PURCHASE_INTENTION_ITEM_2,
    PROBABLE_BUY_LIMITED_SHOE: PURCHASE_INTENTION_ITEM_3,
    "How much would you pay for a recycled limited edition shoe?": "Willigness To Pay",
    "Please think back to the introduction you’ve read. What is unique about this limited edition shoe?": "Control Question",
    "Do you know this athlete?": "Athlete Knowing",
    LIKELY_PURCHASE_DANIELA: ATHLETE_ACHIEVEMENT_ITEM_1_DANIELA,
    INFLUENCE_PERCEPTION_DANIELA: ATHLETE_ACHIEVEMENT_ITEM_2_DANIELA,
    ENHANCE_APPEAL_DANIELA: ATHLETE_ACHIEVEMENT_ITEM_3_DANIELA,
    ENVIRONMENTALLY_PERCEIVE_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_1,
    INFLUENCE_PERCEPTION_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_2,
    LIKELY_RECOMMEND_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_3,
    INTEREST_LIMITED_SHOE: LIMITED_EDITION_ITEM_1,
    IMPORTANT_LIMITED_SHOE: LIMITED_EDITION_ITEM_2,
    INFLUENCE_LIMITED_SHOE: LIMITED_EDITION_ITEM_3,
    "What is your age?": AGE,
    "How do you identify your gender?": GENDER,
    "What is your highest level of education completed?": EDUCATION,
    "What is your total annual net household income?": INCOME,
    "What is your current location? (country/region)": "Location",
    "Which statement do you most agree with?": "Interest Recycled Sportswear/Shoes ",
    "Which statement do you most agree with?2": "World/Olympic Record",
}

for k in rename_column_ryf:
    if k not in ryf.columns:
        print(f"{k} not in Ryf Dataframe")

ryf = ryf.rename(rename_column_ryf, axis=1)
ryf["athlete"] = "Ryf"
# %%

kipchoge = pd.read_excel(
    "Master Thesis Questionnaire analysis.xlsx",
    sheet_name="Eliud Kipchoge",
)
rename_column_kipchoge = {
    "How often do you buy sportswear (sport shoes and sport apparel)?": "Purchase Frequency",
    "How important is sustainability in your fashion choices?": "Importance Sustainability In Fashion",
    LIKELY_LIMITED_SHOE_NEXT_MONTH: PURCHASE_INTENTION_ITEM_1,
    INTEND_LIMITED_SHOE_NEAR_FUTURE: PURCHASE_INTENTION_ITEM_2,
    PROBABLE_BUY_LIMITED_SHOE: PURCHASE_INTENTION_ITEM_3,
    "How much would you pay for a recycled limited edition shoe?": "Willigness To Pay",
    "Please think back to the introduction you’ve read. What is unique about this limited edition shoe?": "Control Question",
    "Do you know this athlete?": "Athlete Knowing",
    LIKELY_PURCHASE_ELIUD: ATHLETE_ACHIEVEMENT_ITEM_1_ELIUD,
    INFLUENCE_PERCEPTION_ELIUD: ATHLETE_ACHIEVEMENT_ITEM_2_ELIUD,
    ENHANCE_APPEAL_ELIUD: ATHLETE_ACHIEVEMENT_ITEM_3_ELIUD,
    ENVIRONMENTALLY_PERCEIVE_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_1,
    INFLUENCE_PERCEPTION_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_2,
    LIKELY_RECOMMEND_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_3,
    INTEREST_LIMITED_SHOE: LIMITED_EDITION_ITEM_1,
    IMPORTANT_LIMITED_SHOE: LIMITED_EDITION_ITEM_2,
    INFLUENCE_LIMITED_SHOE: LIMITED_EDITION_ITEM_3,
    "What is your age?": AGE,
    "How do you identify your gender?": GENDER,
    "What is your highest level of education completed?": EDUCATION,
    "What is your total annual net household income?": INCOME,
    "What is your current location? (country/region)": "Location",
    "Which statement do you most agree with?": "Interest Recycled Sportswear/Shoes ",
    "Which statement do you most agree with?2": "World/Olympic Record",
}

for k in rename_column_kipchoge:
    if k not in kipchoge.columns:
        print(f"{k} not in Kipchoge Dataframe")

kipchoge = kipchoge.rename(rename_column_kipchoge, axis=1)

kipchoge["athlete"] = "Kipchoge"
# %%

bolt = pd.read_excel(
    "Master Thesis Questionnaire analysis.xlsx",
    sheet_name="Usain Bolt",
)
bolt["athlete"] = "Bolt"

rename_column_bolt = {
    "How often do you buy sportswear (sport shoes and sport apparel)?": "Purchase Frequency",
    "How important is sustainability in your fashion choices?": "Importance Sustainability In Fashion",
    LIKELY_LIMITED_SHOE_NEXT_MONTH: PURCHASE_INTENTION_ITEM_1,
    INTEND_LIMITED_SHOE_NEAR_FUTURE: PURCHASE_INTENTION_ITEM_2,
    PROBABLE_BUY_LIMITED_SHOE: PURCHASE_INTENTION_ITEM_3,
    "How much would you pay for a recycled limited edition shoe?": "Willigness To Pay",
    "Please think back to the introduction you’ve read. What is unique about this limited edition shoe?": "Control Question",
    "Do you know this athlete?": "Athlete Knowing",
    LIKELY_PURCHASE_USAIN: ATHLETE_ACHIEVEMENT_ITEM_1_USAIN,
    INFLUENCE_PERCEPTION_USAIN: ATHLETE_ACHIEVEMENT_ITEM_2_USAIN,
    ENHANCE_APPEAL_USAIN: ATHLETE_ACHIEVEMENT_ITEM_3_USAIN,
    ENVIRONMENTALLY_PERCEIVE_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_1,
    INFLUENCE_PERCEPTION_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_2,
    LIKELY_RECOMMEND_LIMITED_SHOE: CONSUMER_PERCEPTION_ITEM_3,
    INTEREST_LIMITED_SHOE: LIMITED_EDITION_ITEM_1,
    IMPORTANT_LIMITED_SHOE: LIMITED_EDITION_ITEM_2,
    INFLUENCE_LIMITED_SHOE: LIMITED_EDITION_ITEM_3,
    "What is your age?": AGE,
    "How do you identify your gender?": GENDER,
    "What is your highest level of education completed?": EDUCATION,
    "What is your total annual net household income?": INCOME,
    "What is your current location? (country/region)": "Location",
    "Which statement do you most agree with?": "Interest Recycled Sportswear/Shoes ",
    "Which statement do you most agree with?2": "World/Olympic Record",
}

for k in rename_column_bolt:
    if k not in bolt.columns:
        print(f"{k} not in Bolt Dataframe")

bolt = bolt.rename(rename_column_bolt, axis=1)

# %%
merged = pd.concat([control, ryf, kipchoge, bolt])

# %%

merged[GENDER].value_counts(dropna=False)
# %%
merged.columns
# %%
merged = merged.reset_index(drop=True)
merged.to_excel("merged_dataset.xlsx")


# %%
print(merged[GENDER].value_counts())
print(merged[GENDER].value_counts() / len(merged))
# %%
AGE_GROUP = "Age Group"
age_group_map = {25: "<25", 35: "26 - 35", 45: "36 - 45", 55: "46 - 55", 65: "56-65"}


def determine_age_group(age: int) -> str:
    for upper_age, group_name in age_group_map.items():
        if upper_age > age:
            return group_name
    return ">65"


merged[AGE_GROUP] = merged[AGE].apply(determine_age_group)
print(merged[AGE_GROUP].value_counts())
print(merged[AGE_GROUP].value_counts() / len(merged) * 100)
# %%
merged[EDUCATION] = merged[EDUCATION].str.replace("’", "'")
print(merged[EDUCATION].value_counts())
print(merged[EDUCATION].value_counts() / len(merged) * 100)

# %%
INCOME_GROUP = "Income Group"
income_group_map = {
    25_000: "<25.000",
    50_000: "25.001 - 50.000",
    75_000: "50.001 - 75.000",
    100_000: "75.001 - 100.000",
    250_000: "100.001 - 250.000",
}


def determine_income_group(income: int) -> str:
    for upper_income, group_name in income_group_map.items():
        if upper_income > income:
            return group_name
    return ">250.000"


merged[INCOME_GROUP] = merged[CLEANED_INCOME].apply(determine_income_group)
print(merged[INCOME_GROUP].value_counts())
print(merged[INCOME_GROUP].value_counts() / len(merged) * 100)

# %%
PURCHASE_KEY = "purchase"
CONSUMER_KEY = "consumer"
LIMITED_KEY = "limited"
USAIN_KEY = "loaded_usain"
DANIELA_KEY = "loaded_daniela"
ELIUD_KEY = "loaded_eliud"


grouping = {
    PURCHASE_KEY: [
        PURCHASE_INTENTION_ITEM_1,
        PURCHASE_INTENTION_ITEM_2,
        PURCHASE_INTENTION_ITEM_3,
    ],
    CONSUMER_KEY: [
        CONSUMER_PERCEPTION_ITEM_1,
        CONSUMER_PERCEPTION_ITEM_2,
        CONSUMER_PERCEPTION_ITEM_3,
    ],
    LIMITED_KEY: [
        LIMITED_EDITION_ITEM_1,
        LIMITED_EDITION_ITEM_2,
        LIMITED_EDITION_ITEM_3,
    ],
    USAIN_KEY: [
        ATHLETE_ACHIEVEMENT_ITEM_1_USAIN,
        ATHLETE_ACHIEVEMENT_ITEM_2_USAIN,
        ATHLETE_ACHIEVEMENT_ITEM_3_USAIN,
    ],
    DANIELA_KEY: [
        ATHLETE_ACHIEVEMENT_ITEM_1_DANIELA,
        ATHLETE_ACHIEVEMENT_ITEM_2_DANIELA,
        ATHLETE_ACHIEVEMENT_ITEM_3_DANIELA,
    ],
    ELIUD_KEY: [
        ATHLETE_ACHIEVEMENT_ITEM_1_ELIUD,
        ATHLETE_ACHIEVEMENT_ITEM_2_ELIUD,
        ATHLETE_ACHIEVEMENT_ITEM_3_ELIUD,
    ],
}
likely_map = {
    "Very unlikely": 0,
    "Unlikely": 1,
    "Neutral": 2,
    "Likely": 3,
    "Very likely": 4,
}
# %%
influence_map = {
    "No influence": 0,
    "Little influence": 1,
    "Moderate influence": 2,
    "Significant influence": 3,
    "Very significant influence": 4,
}

influential_map = {
    "Not at all influential": 0,
    "Slightly influential": 1,
    "Moderately influential": 2,
    "Very influential": 3,
    "Highly influential": 4,
}

influential_map_2 = {
    "Not influential at all": 0,
    "Slightly influential": 1,
    "Moderately influential": 2,
    "Very influential": 3,
    "Extremely influential": 4,
}

agree_map = {
    "Strongly disagree": 0,
    "Disagree": 1,
    "Neutral": 2,
    "Agree": 3,
    "Strongly agree": 4,
}

environmental_map = {
    "Not at all environmentally friendly": 0,
    "Slightly environmentally friendly": 1,
    "Moderately environmentally friendly": 2,
    "Very environmentally friendly": 3,
    "Extremely environmentally friendly": 4,
}

recommendation_map = {
    "Very unlikely to recommend": 0,
    "Unlikely to recommend": 1,
    "Neutral": 2,
    "Likely to recommend": 3,
    "Highly likely to recommend": 4,
}

interested_map = {
    "Not interested at all": 0,
    "Slightly interested": 1,
    "Moderately interested": 2,
    "Very interested": 3,
    "Extremely interested": 4,
}

importance_map = {
    "Not important at all": 0,
    "Slightly important": 1,
    "Moderately important": 2,
    "Very important": 3,
    "Extremely important": 4,
}

education_map = {
    "High school or below": 0,
    "Bachelor's degree": 1,
    "Master's degree or higher": 2,
}
# %% FACTOR LOADING (MEAN/STD) MERGED DATA PURCHASE INTENTION

for col in grouping[PURCHASE_KEY]:
    merged[col] = merged[col].apply(lambda x: likely_map[x])

# %% FACTOR LOADING (MEAN/STD) MERGED DATA CONSUMER PERCEPTION
merged[CONSUMER_PERCEPTION_ITEM_1] = merged[CONSUMER_PERCEPTION_ITEM_1].apply(
    lambda x: environmental_map[x]
)
merged[CONSUMER_PERCEPTION_ITEM_2] = merged[CONSUMER_PERCEPTION_ITEM_2].apply(
    lambda x: influential_map[x]
)
merged[CONSUMER_PERCEPTION_ITEM_3] = merged[CONSUMER_PERCEPTION_ITEM_3].apply(
    lambda x: recommendation_map[x]
)
merged = merged.join(pd.get_dummies(merged[GENDER]))
merged[LIMITED_EDITION_ITEM_1] = merged[LIMITED_EDITION_ITEM_1].apply(
    lambda x: interested_map[x]
)
merged[LIMITED_EDITION_ITEM_2] = merged[LIMITED_EDITION_ITEM_2].apply(
    lambda x: importance_map[x]
)
merged[LIMITED_EDITION_ITEM_3] = merged[LIMITED_EDITION_ITEM_3].apply(
    lambda x: influence_map[x]
)
merged[EDUCATION] = merged[EDUCATION].apply(lambda x: education_map[x])
merged[ATHLETE_ACHIEVEMENT_ITEM_1_USAIN] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_1_USAIN
].apply(lambda x: likely_map.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_2_USAIN] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_2_USAIN
].apply(lambda x: influential_map_2.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_3_USAIN] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_3_USAIN
].apply(lambda x: agree_map.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_1_DANIELA] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_1_DANIELA
].apply(lambda x: likely_map.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_2_DANIELA] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_2_DANIELA
].apply(lambda x: influential_map_2.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_3_DANIELA] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_3_DANIELA
].apply(lambda x: agree_map.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_1_ELIUD] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_1_ELIUD
].apply(lambda x: likely_map.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_2_ELIUD] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_2_ELIUD
].apply(lambda x: influential_map_2.get(x, None))
merged[ATHLETE_ACHIEVEMENT_ITEM_3_ELIUD] = merged[
    ATHLETE_ACHIEVEMENT_ITEM_3_ELIUD
].apply(lambda x: agree_map.get(x, None))
# %%
for col in merged.columns:
    print(col)
    print(merged[col].value_counts() / pd.notnull(merged[col]).sum())
    # for k, v in merged[col].value_counts().items():
    #     print(f"{k} & {v:.2f} \\\\")

# %%
numeric_columns = [col for col in merged.columns if is_numeric_dtype(merged[col])]
control = merged.loc[merged["athlete"] == ""]
print(control[numeric_columns].mean())
print(control[numeric_columns].std())
# %%
numeric_columns = [col for col in merged.columns if is_numeric_dtype(merged[col])]
ryf = merged.loc[merged["athlete"] == "Ryf"]
print(control[numeric_columns].mean())
print(control[numeric_columns].std())
# %%
numeric_columns = [col for col in merged.columns if is_numeric_dtype(merged[col])]
kipchoge = merged.loc[merged["athlete"] == "Kipchoge"]
print(control[numeric_columns].mean())
print(control[numeric_columns].std())
# %%
numeric_columns = [col for col in merged.columns if is_numeric_dtype(merged[col])]
bolt = merged.loc[merged["athlete"] == "Bolt"]
print(control[numeric_columns].mean())
print(control[numeric_columns].std())
# %%

# %%
from scipy import stats


def get_factors_for_grouping(
    df: pd.DataFrame, questions: list[str]
) -> dict[str, float]:
    for col in questions:
        df = df.loc[pd.notnull(df[col])]
    if len(df) == 0:
        return {}
    for col in questions:
        print(f"{col=} {df[col].mean()=} {df[col].std()=}")

    fa = FactorAnalysis(n_components=1, rotation="varimax")
    fa.fit(df[questions])
    print(f"{stats.chisquare(fa.transform(df[questions]))=}")
    print(f"{fa.loglike_[-1]=}")
    loading_factors = list(fa.components_[0])
    if all(f < 0 for f in loading_factors):
        loading_factors = [-f for f in loading_factors]
    return {q: factor for q, factor in zip(questions, loading_factors)}


# %%

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

loading_factors_merged = {
    key: get_factors_for_grouping(merged, group) for key, group in grouping.items()
}
for key, group in grouping.items():
    print(f"{key=} {calculate_bartlett_sphericity(merged[group])=}")
# %%
loading_factors_control = {
    key: get_factors_for_grouping(control, group) for key, group in grouping.items()
}
for key, group in grouping.items():
    print(f"{key=} {calculate_bartlett_sphericity(control[group])=}")
# %%
ryf_df = merged.loc[merged["athlete"] == "Ryf"]
loading_factors_ryf = {
    key: get_factors_for_grouping(ryf_df, group) for key, group in grouping.items()
}

for key, group in grouping.items():
    print(f"{key=} {calculate_bartlett_sphericity(ryf_df[group])=}")
# %%
bolt_df = merged.loc[merged["athlete"] == "Bolt"]
loading_factors_bolt = {
    key: get_factors_for_grouping(bolt_df, group) for key, group in grouping.items()
}
for key, group in grouping.items():
    print(f"{key=} {calculate_bartlett_sphericity(bolt_df[group])=}")
# %%
kipchoge_df = merged.loc[merged["athlete"] == "Kipchoge"]
loading_factors_kipchoge = {
    key: get_factors_for_grouping(kipchoge_df, group) for key, group in grouping.items()
}

for key, group in grouping.items():
    print(f"{key=} {calculate_bartlett_sphericity(kipchoge_df[group])=}")


# %%


def composite_reliability(factors: list[float]) -> float:
    factors = np.array(factors)
    error = 1 - factors**2  # calculare error for item i
    square_of_sums = (np.sum(factors)) ** 2
    composite = square_of_sums / (square_of_sums + np.sum(error))
    return composite


def composite_reliabilities(factor_loadings: dict[str, dict[str, float]]):
    return {
        col: composite_reliability(list(factors.values()))
        for col, factors in factor_loadings.items()
    }


cr_merged = composite_reliabilities(loading_factors_merged)
print(f"{cr_merged=}")
cr_control = composite_reliabilities(loading_factors_control)
print(f"{cr_control=}")
cr_bolt = composite_reliabilities(loading_factors_bolt)
print(f"{cr_bolt=}")
cr_ryf = composite_reliabilities(loading_factors_ryf)
print(f"{cr_ryf=}")
cr_kipchoge = composite_reliabilities(loading_factors_kipchoge)
print(f"{cr_kipchoge=}")


# %%


def average_variance_extracted(factors: list[float]) -> float:
    factors = np.array(factors)
    error = 1 - factors**2
    sum_of_squares = np.sum(factors**2)
    return sum_of_squares / (sum_of_squares + np.sum(error))


def average_variances_extracted(factor_loadings: dict[str, dict[str, float]]):
    return {
        col: average_variance_extracted(list(factors.values()))
        for col, factors in factor_loadings.items()
    }


ave_merged = average_variances_extracted(loading_factors_merged)
print(f"{ave_merged=}")
ave_control = average_variances_extracted(loading_factors_control)
print(f"{ave_control=}")
ave_bolt = average_variances_extracted(loading_factors_bolt)
print(f"{ave_bolt=}")
ave_ryf = average_variances_extracted(loading_factors_ryf)
print(f"{ave_ryf=}")
ave_kipchoge = average_variances_extracted(loading_factors_kipchoge)
print(f"{ave_kipchoge=}")
# %%


def compute_value_given_loadings(row: pd.Series, loadings: dict[str, float]) -> float:
    if any(pd.isnull(row[col]) for col in loadings):
        return float("NaN")
    value = 0
    for column, loading in loadings.items():
        value += row[column] * loading
    return value


def compute_loaded_dataframe(
    df: pd.DataFrame, loading_factors: dict[str, dict[str, float]]
) -> pd.DataFrame:
    df = df.copy()
    for factor_name, factor_dict in loading_factors.items():
        df[factor_name] = df.apply(
            lambda x: compute_value_given_loadings(x, factor_dict), axis=1
        )
    return df[list(loading_factors.keys())]


# %%
loaded = compute_loaded_dataframe(merged, loading_factors_merged)

merged.loc[merged["athlete"] == "", "athlete"] = "control"
merged = merged.join(pd.get_dummies(merged["athlete"]))
loaded.corr()
merged["has_athlete"] = merged["athlete"] != "control"
# %%

from pyprocessmacro import Process

for athlete in ["loaded_usain", "loaded_daniela", "loaded_eliud"]:
    print(f"{athlete=}")
    p = Process(data=loaded, model=1, x="limited", y="purchase", m=[athlete])
    p.summary()
# %%

merged_with_loaded = merged.join(loaded)
merged_with_loaded = merged_with_loaded.fillna(0)

p = Process(data=loaded, model=1, x="limited", y="purchase", m="consumer")
p.summary()
# %%
p = Process(
    data=merged_with_loaded, model=1, x="limited", y="purchase", m="has_athlete"
)
p.summary()

# %%
p = Process(
    data=merged_with_loaded, model=1, x="limited", y="consumer", m="has_athlete"
)
p.summary()


# %%
def covert_to_numeric(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return 0


merged_with_loaded["CLEANED WTP"] = (
    merged_with_loaded["CLEANED WTP"].fillna(0).apply(covert_to_numeric)
)
merged_with_loaded[["athlete", "CLEANED WTP"]].groupby("athlete")[
    ["CLEANED WTP"]
].mean()
# %%
print(
    "Female: ",
    merged_with_loaded.loc[merged_with_loaded["Female"]][["CLEANED WTP"]].mean(),
)
# %%
print(
    "Male: ",
    merged_with_loaded.loc[merged_with_loaded["Male"]][["CLEANED WTP"]].mean(),
)

# %%

from scipy.stats import pearsonr

input_columns = [
    "limited",
    "consumer",
    "Bolt",
    "Ryf",
    "Kipchoge",
    "loaded_usain",
    "loaded_daniela",
    "loaded_eliud",
    "control",
    AGE,
    CLEANED_INCOME,
    EDUCATION,
    "CLEANED KNOWING",
    "Female",
    "Male",
    "Non-binary/Other",
]
merged_with_loaded[[*input_columns, "CLEANED WTP", "purchase"]].corr(method="pearson")


# %%
def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


merged_with_loaded[[*input_columns, "CLEANED WTP", "purchase"]].corr(
    method=pearsonr_pval
)

# %%
import statsmodels.api as sm

X2 = sm.add_constant(merged_with_loaded[input_columns].astype(float))
est = sm.OLS(merged_with_loaded["CLEANED WTP"], X2)
est = est.fit()
print(est.summary())

# %%

X2 = sm.add_constant(merged_with_loaded[input_columns].astype(float))
est = sm.OLS(merged_with_loaded["purchase"], X2)
est = est.fit()
print(est.summary())
# %%
merged_with_loaded.to_excel("MT_EDIT.xlsx")

# %%
import statsmodels.api as sm
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.stats.mediation import Mediation

# Defining the probit link function
probit_link = links.probit()

# Defining the outcome model using GLM with a binomial family and a probit link function
outcome_model = sm.GLM.from_formula(
    "purchase ~ consumer + limited",
    data=loaded,
    family=families.Binomial(link=probit_link),
)

# Defining the mediator model using OLS
mediator_model = sm.OLS.from_formula("consumer ~ limited", data=loaded)

# Fitting the mediation model
med = Mediation(outcome_model, mediator_model, "limited", "consumer").fit()

# Displaying the summary of the mediation model
print(med.summary())

# %%
import statsmodels.api as sm
import statsmodels.genmod.families.links as links

probit = links.probit
outcome_model = sm.GLM.from_formula(
    "purchase ~ consumer + limited",
    data=loaded,
    family=sm.families.Binomial(link=probit()),
)
mediator_model = sm.OLS.from_formula("consumer ~ limited", data=loaded)
med = Mediation(outcome_model, mediator_model, "limited", "consumer").fit()
med.summary()
