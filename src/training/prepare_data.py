import pandas as pd


def main():
    df = pd.read_csv("data/processed/dataset.csv")

    df = df.sort_values("timestamp")

    # target = next day's AQI
    df["target_aqi"] = df["aqi"].shift(-1)

    df = df.dropna()

    df.to_csv("data/processed/train_dataset.csv", index=False)

    print("Training dataset saved:", df.shape)


if __name__ == "__main__":
    main()
