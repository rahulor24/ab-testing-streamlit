import streamlit as st
from utils import clean_data
from utils import Visualization
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

tab1, tab2, tab3 = st.tabs(["Introduction", "EDA", "A/B Test"])

with tab1:
    st.title("Market A/B Testing Analysis")
    st.header("Overview")
    st.markdown("""This project evaluates the effectiveness of a marketing ad campaign using A/B testing methodology. By comparing a test group (users exposed to ads) with a control group (users exposed to Public Service Announcements, PSAs), we assess whether ad exposure leads to a statistically significant improvement in user conversions

**WORKFLOW:**
1. Data Preparation
- Imported and inspected the dataset for structure, missing values and duplicate values.
- Applied outlier detection on the Total Ads and Most Ads Hour columns using the Interquartile Range (IQR) method to remove extreme values that could bias results.
2. Exploratory Data Analysis (EDA)
- Visualized conversion across Test (Ad) and Control (PSA) groups.
- Examined the distribution of Total Ads viewed and Most Ads Hours.
- Analyzed conversion patterns and identified peak ad exposure times by day and hour.
3. A/B Testing
- Split the dataset into Group A (Ad) and Group B (PSA).
- Calculated conversion rates for both groups.
- Conducted an Independent T-test to evaluate statistical significance of observed differences.

**HYPOTHESIS**
- Null Hypothesis (H₀): No difference exists in conversion rates between the Ad group and the PSA group.
- Alternative Hypothesis (H₁): Conversion rates differ between the Ad group and the PSA group (with the expectation that ads increase conversions).
""")

    st.write("----")
    st.subheader("Cleaned Dataset Preview")
    df = clean_data("./dataset/marketing_AB.csv")

    filter= st.checkbox("Apply Filters to Dataset Preview", value=False)
    if filter:
        st.sidebar.header("Filters")
        test_group = st.sidebar.multiselect("Select Test Group", df["test group"].unique())
        day_filter = st.sidebar.multiselect("Select Days", df["most ads day"].unique())

        # Build mask
        mask = pd.Series(True, index=df.index)   # start with all True
        if test_group:
            mask &= df["test group"].isin(test_group)
        if day_filter:
            mask &= df["most ads day"].isin(day_filter)

        # Apply mask
        filtered_df = df[mask].reset_index(drop=True)
        st.dataframe(filtered_df)

    else:
        st.dataframe(df.reset_index(drop=True))

    st.subheader("Descriptive Statistics")
    st.write(df.describe().round(2))

    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_marketing_AB.csv',
        mime='text/csv',
    )

    st.write("----")

with tab2:
    st.title("Market A/B Testing Analysis")
    st.header("Exploratory Data Analysis (EDA)")

    # plot_type = st.radio("Choose Plot", ["Conversion by Test Group", "Mean Total Ads by Conversion Status for Each Test Group", "Conversion by Most Ads Day and Bifurcated by Test Group", "Mean Most Ads Hour by Conversion Status for Each Test Group"])

    # if plot_type == "Conversion by Test Group":
    if st.checkbox("Show Conversion by Test Group"):
        # Visualization of Conversion by Test Group
        st.subheader("Conversion by Test Group")
        fig = Visualization.conversion_by_test_group(df)
        st.pyplot(fig)
        st.write(f"""The visualization above illustrates the conversion rates between the Test Group (exposed to ads) and the Control Group (exposed to PSAs). It is evident that the Test Group exhibits a higher conversion rate of {6832*100/503015:.2f} compared to the Control Group (conversion rate: {226*100/20893:.2f}), suggesting that ad exposure positively influences user conversions.""")

    # elif plot_type == "Mean Total Ads by Conversion Status for Each Test Group":
    if st.checkbox("Show Mean Total Ads by Conversion Status for Each Test Group"):
        # Visualize mean total ads by conversion status for each test group
        st.subheader("Mean Total Ads by Conversion Status for Each Test Group")
        fig = Visualization.mean_total_ads_by_conversion(df)
        st.pyplot(fig)
        st.write("From above graph we can interpret that on an average for conversion around 33 ads are required for 'ad' group whereas only 28 (approx) ads are required for 'psa' test group.")

    # elif plot_type == "Conversion by Most Ads Day and Bifurcated by Test Group":
    if st.checkbox("Show Conversion by Most Ads Day and Bifurcated by Test Group"):
        # Visualization of Conversion by Most Ads Day and Bifurcated by Test Group
        st.subheader("Conversion by Most Ads Day and Bifurcated by Test Group")
        fig = Visualization.conversions_by_most_ads_day(df)
        st.pyplot(fig)
        st.write("In above graphs we can clearly see the huge difference in conversion for both test group but due to class imbalance it's not advisable to conclude any final statement rightnow.")

    # else:
    if st.checkbox("Show Mean Most Ads Hour by Conversion Status for Each Test Group"):
        # Visualization of Mean Most Ads Hour by Conversion Status
        st.subheader("Mean Most Ads Hour by Conversion Status for Each Test Group")
        fig = Visualization.mean_most_ads_hour_by_conversion(df)
        st.pyplot(fig)
        st.write("From above graph we can interpret that most ads hour doesn't have significant difference between conversion of 'ad' group and 'psa' group.")

    st.write("----")


with tab3:
    st.title("Market A/B Testing Analysis")
    # A/B test (two-sample t-test) on conversion rate between 'ad' and 'psa' groups
    # Null hypothesis: no difference in conversion rate between groups

    st.header("A/B Test on Conversion Rates Between 'Ad' and 'PSA' Groups")
    st.write("Conducting a Welch's t-test to compare conversion rates between the 'ad' and 'psa' groups.")
    st.write("Null hypothesis: no difference in conversion rate between groups")

    ad = df[df['test group'] == 'ad']['converted'].astype(int)
    psa = df[df['test group'] == 'psa']['converted'].astype(int)

    n_ad, n_psa = len(ad), len(psa)
    cr_ad, cr_psa = ad.mean(), psa.mean()
    diff = cr_ad - cr_psa

    # Welch's t-test (unequal variances)
    t_stat, p_value = ttest_ind(ad, psa, equal_var=False)

    # 95% CI for difference in proportions using normal approximation (large samples)
    se = np.sqrt(cr_ad * (1 - cr_ad) / n_ad + cr_psa * (1 - cr_psa) / n_psa)
    z95 = 1.96
    ci_lower, ci_upper = diff - z95 * se, diff + z95 * se

    st.write("#### A/B Test Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label= "Conversion Rate - Ad Group", value = f"{cr_ad:.2%}")

    with col2:
        st.metric(label= "Conversion Rate - PSA Group", value = f"{cr_psa:.2%}")

    st.markdown(f"""
Where:
- Group sizes: ad = {n_ad}, psa = {n_psa}
- Difference (ad - psa) = {diff:.4%}
- t-statistic = {t_stat:.4f}, p-value = {p_value:.5g}
- 95% CI for difference = ({ci_lower:.4%}, {ci_upper:.4%})
""")


    st.write(" ")
    alpha = 0.05
    st.markdown(f"""Significance level: alpha = 0.05
- Decision rule:
    - If p-value < alpha, reject H0 (evidence of difference in conversion rates)
    - If p-value >= alpha, fail to reject H0 (no evidence of difference)
""")

    if p_value < alpha:
        st.write(f"- Hence, reject H0 at alpha={alpha}: conversion rates differ between groups.")
    else:
        st.write(f"- Hence, failed to reject H0 at alpha={alpha}: no evidence of difference in conversion rates.")

    st.write("----")

    # Final conclusion and business recommendation
    st.header("Final Conclusion and Business Recommendation")
    conclusion = f"""
    **Final conclusion:**
    - Observed conversion rates: Ad = {cr_ad:.4%}, PSA = {cr_psa:.4%}
        Absolute difference in conversion rates (Ad - PSA) = {diff:.4%} (≈{(diff/cr_psa):.1%} relative lift).
    - Statistical test (Welch two-sample t-test): t = {t_stat:.4f}, p = {p_value:.5g}, alpha = {alpha}
        95% CI for difference = ({ci_lower:.4%}, {ci_upper:.4%}).

    **Interpretation:**
    - The p-value < alpha so we reject the null hypothesis of no difference.
    - The 95% CI is entirely positive, so the 'ad' variant shows a statistically significant higher conversion rate than 'psa'.
    - Effect size: the absolute uplift is ~{diff:.4%} (small in absolute terms), but relative uplift is ~{(diff/cr_psa):.1%}.

    **Business recommendation (preferred strategy):**
    - Prefer the 'ad' variant for rollout, subject to business validation:
        - Expected incremental conversions ≈ {int(diff * 100000)} per 100k users exposed.
        - Before full rollout, perform an ROI analysis: compare incremental revenue per conversion vs incremental cost of serving 'ad'.
        - Run a staged rollout (canary/holdout) while monitoring secondary metrics (retention, engagement, conversion quality, churn).

    **Next steps & caveats:**
    - Although results are statistically significant (large sample), evaluate practical significance and unit economics.
    - Conduct segment analysis to identify cohorts with larger lifts (targeted rollout).
    - Confirm results with a proportion z-test or Bayesian analysis and ensure no bias from preprocessing/outlier handling.
    - Monitor metrics post-rollout to catch unintended effects.

    """
    st.write(conclusion)

    st.write("----")
    st.success("End of A/B Testing Analysis Report")