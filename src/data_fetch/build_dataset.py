import pandas as pd
import glob

def main():
    files = glob.glob("data/raw/*.csv")
    if not files:
        print("No raw data files found.")
        return

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv("data/processed/dataset.csv", index=False)
    print("Saved processed dataset:", df.shape)


if __name__ == "__main__":
    main()
