import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
from sklearn.metrics import cohen_kappa_score

df = pd.read_excel('review.xlsx', sheet_name='Sheet1')
df = df.drop(['Prompt', 'LLM-1', 'LLM-2', 'LLM-3', 'LLM-4'], axis=1)


def fleiss_kappa_fn(df):
    results = {}
    question_suffixes = ['-1', '-2', '-3', '-4']
    categories = ['yes', 'no']  # Define the categories of interest

    for i, suffix in enumerate(question_suffixes):
        question_number = i + 1
        kappa_key = f'question_{question_number}_kappa'

        # Identify columns for the current question
        relevant_cols = [col for col in df.columns if col.endswith(suffix)]

        if not relevant_cols:
            print(f"Info for Question {question_number} (suffix '{suffix}'): No columns found.")
            results[kappa_key] = None
            continue

        if len(relevant_cols) < 2:
            print(
                f"Info for Question {question_number} (suffix '{suffix}'): Fewer than 2 raters ({len(relevant_cols)} found). Kappa requires at least 2 raters.")
            results[kappa_key] = None
            continue

        # Work on a copy for processing
        question_data_subset = df[relevant_cols].copy()

        # Drop rows (items) where ANY reviewer for THIS question has a missing (NaN) answer
        question_data_cleaned = question_data_subset.dropna()

        if question_data_cleaned.shape[0] < 2:  # Need at least 2 subjects (items/prompts)
            print(
                f"Info for Question {question_number} (suffix '{suffix}'): Fewer than 2 items with non-missing ratings from all raters. Kappa calculation skipped.")
            results[kappa_key] = None
            continue

        num_raters = question_data_cleaned.shape[1]

        # Prepare the (N_items x N_categories) table for Fleiss' Kappa
        # Each cell (i,j) is the number of raters who assigned item i to category j.
        # The sum of counts for each item across categories MUST be `num_raters`.

        rating_counts_list = []  # To store lists of counts for valid items

        for _original_idx, row_data in question_data_cleaned.iterrows():
            current_item_ratings_lower = [str(r).lower() for r in row_data]

            counts_for_this_item = [current_item_ratings_lower.count(cat) for cat in categories]

            # Only include this item if all its ratings fall into the defined categories ('yes' or 'no')
            if sum(counts_for_this_item) == num_raters:
                rating_counts_list.append(counts_for_this_item)
            # else:
            # This item has ratings other than 'yes'/'no' (e.g., 'maybe', blank strings not NaN)
            # or inconsistent counts. It's excluded to maintain data integrity for Fleiss Kappa.
            # print(f"Debug: Item at original index {_original_idx} for Q{question_number} excluded. Ratings: {current_item_ratings_lower}, Counts: {counts_for_this_item}, Sum: {sum(counts_for_this_item)}, Raters: {num_raters}")

        if not rating_counts_list or len(rating_counts_list) < 2:
            print(
                f"Info for Question {question_number} (suffix '{suffix}'): Fewer than 2 items remaining after ensuring all ratings are 'yes' or 'no'. Kappa calculation skipped.")
            results[kappa_key] = None
            continue

        final_agg_table_np = np.array(rating_counts_list)
        print(final_agg_table_np)

        try:
            # Note: fleiss_kappa can return NaN if, for example, all ratings fall into one category
            # (e.g., all raters say 'yes' for all items), leading to Pe = 1.
            kappa = fleiss_kappa(final_agg_table_np, method='fleiss')
            results[kappa_key] = kappa
        except Exception as e:
            print(f"Error calculating Fleiss' Kappa for Question {question_number} (suffix '{suffix}'): {e}")
            results[kappa_key] = None

    print(results)


def calculate_cohens_kappa_pairwise(df, rater1_col, rater2_col):
    if rater1_col not in df.columns or rater2_col not in df.columns:
        print(f"Error: One or both columns ('{rater1_col}', '{rater2_col}') not found in DataFrame.")
        return None

    # Create a temporary DataFrame with only the two raters' columns
    ratings_pair_df = df[[rater1_col, rater2_col]].copy()

    # Drop rows where either rater has a missing value for these specific columns
    ratings_pair_df.dropna(subset=[rater1_col, rater2_col], inplace=True)

    if ratings_pair_df.shape[0] < 2:  # Need at least 2 items with ratings from both
        print(f"Warning: Fewer than 2 common rated items for '{rater1_col}' and '{rater2_col}' after handling NaNs.")
        return None

    rater1_ratings = ratings_pair_df[rater1_col].astype(str).str.lower()
    rater2_ratings = ratings_pair_df[rater2_col].astype(str).str.lower()

    # Ensure there's more than one unique value in the combined ratings,
    # otherwise kappa might be undefined or 1.0 by some implementations if all agree on one value.
    # sklearn handles this, often returning NaN or 1.0 appropriately.
    unique_labels = pd.concat([rater1_ratings, rater2_ratings]).nunique()
    if unique_labels < 2 and ratings_pair_df.shape[0] > 0:  # All ratings are identical for one category
        # If all ratings are identical (e.g., everyone said 'yes' for all common items)
        # observed agreement is 1. Expected agreement might also be 1.
        # sklearn's cohen_kappa_score might return NaN or 0.0 in such cases, depending on version and specifics.
        # Or it might correctly assess. Let's allow sklearn to calculate.
        pass

    try:
        kappa = cohen_kappa_score(rater1_ratings, rater2_ratings)
        print(kappa)
    except Exception as e:
        print(f"Error calculating Cohen's Kappa for '{rater1_col}' and '{rater2_col}': {e}")

calculate_cohens_kappa_pairwise(df, 'DRH-1', 'ADM-1')
calculate_cohens_kappa_pairwise(df, 'DRH-2', 'ADM-2')
calculate_cohens_kappa_pairwise(df, 'DRH-3', 'ADM-3')
calculate_cohens_kappa_pairwise(df, 'DRH-4', 'ADM-4')