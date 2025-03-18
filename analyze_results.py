import pandas as pd

# Список CSV-файлов
csv_files = [
    "optuna1800.csv",
    "optuna3600.csv",
    "optuna5400.csv",
    "flaml_r1800.csv",
    "flaml3600.csv",
    "flaml_5400.csv",
    "cnn_model.csv"
]

def load_and_combine_results(csv_files):
    """
    Считывает и объединяет данные из всех CSV-файлов.
    """
    combined_df = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(file)
        df['source'] = file  # Добавляем столбец с названием источника
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df
	
def analyze_results(df):
    """
    Анализирует результаты, выводит сводку и находит лучшие модели.
    """
    print("Общая информация о данных:")
    print(df.describe())  # Общая статистика

    print("\nЛучшие модели по точности:")
    print(df.nlargest(3, 'accuracy')[['best_model_name', 'accuracy', 'source']])

    print("\nЛучшие модели по F1-метрике:")
    print(df.nlargest(3, 'f1_score')[['best_model_name', 'f1_score', 'source']])

    print("\nМодель с минимальным временем обучения:")
    print(df.nsmallest(1, 'training_time')[['best_model_name', 'training_time', 'source']])

if __name__ == "__main__":
    # Загрузка данных
    combined_results = load_and_combine_results(csv_files)
    
    # Анализ данных
    analyze_results(combined_results)