
import pandas as pd

COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def load_train_data():
    file_path = "data/raw/train_FD001.txt"

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    return df


def add_rul_column(df):
    # Find the final cycle for each engine
    max_cycle = df.groupby("engine_id")["cycle"].max()

    # Calculate RUL for every row
    df["RUL"] = (
    df["engine_id"].map(max_cycle) - df["cycle"]
).clip(upper=125)

    return df


if __name__ == "__main__":
    data = load_train_data()
    data = add_rul_column(data)

    print("Dataset shape:", data.shape)
    print(data[["engine_id", "cycle", "RUL"]].head())

    print("\nLast rows of engine 1:")
    print(data[data["engine_id"] == 1][["engine_id", "cycle", "RUL"]].tail())
