def generate_rolling_window_folds(dataframe, train_window, val_window):
    years = sorted(dataframe["year"].unique())

    folds = []

    for i in range(len(years) - train_window - val_window + 1):
        train_years = years[i: i + train_window]
        val_years = years[i + train_window: i + train_window + val_window]
        folds.append((train_years, val_years))

    return folds