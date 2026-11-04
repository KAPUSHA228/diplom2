from pathlib import Path
import pandas as pd
import numpy as np


class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODELS_DIR = ARTIFACTS_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    EXPERIMENTS_OUTPUT_DIR = BASE_DIR / "experiments_output"
    ANALYSIS_DATA_DIR = BASE_DIR / "analysis_data"

    RANDOM_SEED = 42
    DEFAULT_N_CLUSTERS = 3
    RISK_THRESHOLD = 0.5
    SHAP_TOP_N = 5
    CV_FOLDS = 5
    TEST_SIZE = 0.2

    # Пути создаются автоматически при импорте
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    ANALYSIS_DATA_DIR.mkdir(exist_ok=True)

    def __init__(self):
        # Автоматическое создание всех необходимых папок
        for directory in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.EXPERIMENTS_OUTPUT_DIR,
            self.EXPERIMENTS_OUTPUT_DIR / "runs",
            self.EXPERIMENTS_OUTPUT_DIR / "summary",
            self.EXPERIMENTS_OUTPUT_DIR / "figures",
            self.ANALYSIS_DATA_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()


def optimize_dtypes(df: pd.DataFrame, categorical_threshold: float = 0.5) -> pd.DataFrame:
    """
    Оптимизирует типы данных DataFrame для экономии памяти и ускорения.

    Args:
        df: исходный DataFrame
        categorical_threshold: если доля уникальных значений < порога, конвертируем в category

    Returns:
        pd.DataFrame: оптимизированный DataFrame
    """
    df = df.copy()

    for col in df.columns:
        col_type = df[col].dtype

        # 1. Category для строк с небольшим числом уникальных значений
        if col_type == "object":
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < categorical_threshold:
                df[col] = df[col].astype("category")

        # 2. Целочисленные → минимальный int
        elif "int" in str(col_type):
            _info = np.iinfo
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= 0:  # Беззнаковые
                if c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype("uint8")
                elif c_max <= np.iinfo(np.uint16).max:
                    df[col] = df[col].astype("uint16")
                elif c_max <= np.iinfo(np.uint32).max:
                    df[col] = df[col].astype("uint32")
            else:  # Со знаком
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype("int8")
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype("int16")
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype("int32")

        # 3. Float → float32 (достаточно для ML)
        elif "float" in str(col_type):
            df[col] = df[col].astype("float32")

    return df
