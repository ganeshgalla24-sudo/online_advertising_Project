from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.stats_analysis import run_statistics
from src.model import run_model

if __name__ == "__main__":
    df = run_preprocessing("data/raw/online_advertising_performance_data.csv")
    run_eda(df)
    run_statistics(df)
    run_model(df)