import pandas as pd

class GroupEstimate:
    def __init__(self, estimate):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.default_estimates = None

    def fit(self, X, y, default_category=None):
        # Combine X and y into one DataFrame
        df = pd.concat([X, y], axis=1)
        y_col = y.name

        # Compute group estimates (mean or median)
        if self.estimate == "mean":
            self.group_estimates = df.groupby(list(X.columns))[y_col].mean()
        elif self.estimate == "median":
            self.group_estimates = df.groupby(list(X.columns))[y_col].median()
        else:
            raise ValueError("estimate must be either 'mean' or 'median'")

        # Optional: store fallback estimates by a single default category
        if default_category:
            if self.estimate == "mean":
                self.default_estimates = df.groupby(default_category)[y_col].mean()
            elif self.estimate == "median":
                self.default_estimates = df.groupby(default_category)[y_col].median()
        else:
            self.default_estimates = None

    def predict(self, X_):
        # Convert input to DataFrame
        X_ = pd.DataFrame(X_, columns=self.group_estimates.index.names)

        # Merge with known group estimates
        merged = pd.merge(
            X_,
            self.group_estimates.reset_index(),
            on=self.group_estimates.index.names,
            how="left"
        )

        # Identify missing predictions
        missing_mask = merged[self.group_estimates.name].isna()

        # Fallback: use default category estimates if available
        if self.default_estimates is not None:
            default_col = self.default_estimates.index.name
            merged.loc[missing_mask, self.group_estimates.name] = (
                merged.loc[missing_mask, default_col].map(self.default_estimates)
            )

        # Count how many still missing
        missing = merged[self.group_estimates.name].isna().sum()
        if missing > 0:
            print(f"{missing} group(s) still not found in the training data.")

        return merged[self.group_estimates.name].values
