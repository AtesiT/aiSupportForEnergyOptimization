import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ
# ========================

def generate_energy_data(days=30, freq='H'):
    """Генерация синтетических данных энергопотребления"""
    np.random.seed(42)
    
    # Базовое потребление с суточным циклом
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                               end=datetime.now(), freq=freq)
    n_points = len(timestamps)
    
    # Базовый уровень потребления (кВт)
    base_load = 5000  # Базовое потребление 5 МВт
    
    # Суточный цикл (ночь/день)
    daily_cycle = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    
    # Сезонность (рабочие/выходные)
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    weekend_effect = np.where(day_of_week >= 5, -1000, 0)
    
    # Случайные колебания
    random_noise = np.random.normal(0, 300, n_points)
    
    # Генерация аномалий (внезапные скачки потребления)
    anomalies = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.03), replace=False)
    anomalies[anomaly_indices] = np.random.uniform(1000, 4000, len(anomaly_indices))
    
    # Итоговое потребление
    consumption = base_load + daily_cycle + weekend_effect + random_noise + anomalies
    
    # Температура (для контекста)
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 3, n_points)
    
    # Производственные параметры
    production_rate = np.random.uniform(70, 100, n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'energy_kWh': consumption,
        'temperature_C': temperature,
        'production_rate': production_rate,
        'is_anomaly': (anomalies > 0).astype(int)
    })

# ========================
# 2. АНАЛИТИЧЕСКИЙ МОДУЛЬ
# ========================

class EnergyAnomalyDetector:
    """Детектор аномалий энергопотребления"""
    
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.model = IsolationForest(contamination=0.05, random_state=42)
        
    def create_features(self, df):
        """Создание признаков для ML модели"""
        df = df.copy()
        
        # Статистики за скользящее окно
        df['rolling_mean'] = df['energy_kWh'].rolling(window=self.window_size).mean()
        df['rolling_std'] = df['energy_kWh'].rolling(window=self.window_size).std()
        df['z_score'] = (df['energy_kWh'] - df['rolling_mean']) / df['rolling_std']
        
        # Производные признаки
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Отклонение от типичного паттерна
        typical_by_hour = df.groupby('hour')['energy_kWh'].mean()
        df['deviation_from_typical'] = df.apply(
            lambda row: row['energy_kWh'] - typical_by_hour[row['hour']], axis=1
        )
        
        return df
    
    def detect_anomalies(self, df):
        """Обнаружение аномалий с помощью ML"""
        feature_cols = ['energy_kWh', 'z_score', 'deviation_from_typical', 
                       'production_rate', 'temperature_C']
        
        # Подготовка данных
        df_features = self.create_features(df)
        feature_data = df_features[feature_cols].fillna(0)
        
        # Обучение модели и предсказание
        predictions = self.model.fit_predict(feature_data)
        df_features['ml_anomaly'] = np.where(predictions == -1, 1, 0)
        
        # Правило на основе z-score (для сравнения)
        df_features['rule_anomaly'] = np.where(
            abs(df_features['z_score'].fillna(0)) > 3, 1, 0
        )
        
        # Комбинированный результат
        df_features['is_detected'] = np.where(
            (df_features['ml_anomaly'] == 1) | (df_features['rule_anomaly'] == 1), 1, 0
        )
        
        return df_features

# ========================
# 3. МОДУЛЬ РЕКОМЕНДАЦИЙ
# ========================

class EnergyAdvisor:
    """Генератор рекомендаций по энергосбережению"""
    
    def __init__(self, electricity_cost=0.08):
        self.electricity_cost = electricity_cost  # $/kWh
        
    def generate_recommendation(self, row, baseline):
        """Генерация конкретной рекомендации"""
        current_power = row['energy_kWh']
        deviation = current_power - baseline
        excess_kwh = max(0, deviation)
        
        if excess_kwh == 0:
            return None
        
        # Расчет потенциальной экономии за час
        hourly_saving = excess_kwh * self.electricity_cost
        
        # Рекомендации в зависимости от времени суток
        hour = row['timestamp'].hour
        
        if hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Ночное время
            recommendations = [
                f"Переведите ненужное оборудование в спящий режим",
                f"Оптимизируйте работу вентиляции на 30%",
                f"Отключите освещение в неиспользуемых зонах"
            ]
        elif hour in [6, 7, 8, 9, 10, 11]:  # Утро
            recommendations = [
                f"Проверьте график запуска мощного оборудования",
                f"Скорректируйте уставки температуры на 2°C",
                f"Объедините производственные партии для экономии"
            ]
        else:  # День/вечер
            recommendations = [
                f"Запустите диагностику компрессора #3",
                f"Проверьте давление в системе (текущее отклонение: {deviation:.0f} кВт)",
                f"Рассмотрите возможность отложить не критичные процессы на 2 часа"
            ]
        
        # Выбор рекомендации
        rec_idx = min(int(deviation / 500), len(recommendations) - 1)
        
        return {
            'timestamp': row['timestamp'],
            'current_power_kW': current_power,
            'baseline_kW': baseline,
            'excess_kW': deviation,
            'hourly_cost_saving': hourly_saving,
            'recommendation': recommendations[rec_idx],
            'confidence': min(0.95, 0.7 + deviation / 2000)
        }

# ========================
# 4. ВИЗУАЛИЗАЦИЯ
# ========================

def visualize_results(df, recommendations):
    """Визуализация результатов анализа"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График 1: Потребление энергии с аномалиями
    axes[0, 0].plot(df['timestamp'], df['energy_kWh'], label='Потребление', alpha=0.7)
    axes[0, 0].plot(df['timestamp'], df['rolling_mean'], label='Базовый уровень', linestyle='--')
    
    anomaly_points = df[df['is_detected'] == 1]
    if not anomaly_points.empty:
        axes[0, 0].scatter(anomaly_points['timestamp'], anomaly_points['energy_kWh'], 
                          color='red', s=50, label='Аномалии', zorder=5)
    
    axes[0, 0].set_title('Энергопотребление с детекцией аномалий')
    axes[0, 0].set_ylabel('кВт·ч')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Z-score
    axes[0, 1].plot(df['timestamp'], df['z_score'].fillna(0))
    axes[0, 1].axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Порог (z=3)')
    axes[0, 1].axhline(y=-3, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Z-score отклонения от нормы')
    axes[0, 1].set_ylabel('Z-score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Распределение аномалий по часам
    if not anomaly_points.empty:
        hour_counts = anomaly_points['timestamp'].dt.hour.value_counts().sort_index()
        axes[1, 0].bar(hour_counts.index, hour_counts.values)
        axes[1, 0].set_title('Распределение аномалий по часам суток')
        axes[1, 0].set_xlabel('Час')
        axes[1, 0].set_ylabel('Количество аномалий')
        axes[1, 0].set_xticks(range(0, 24, 2))
    
    # График 4: Потенциальная экономия
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        axes[1, 1].bar(range(len(rec_df)), rec_df['hourly_cost_saving'])
        axes[1, 1].set_title('Потенциальная экономия по рекомендациям')
        axes[1, 1].set_xlabel('Рекомендация #')
        axes[1, 1].set_ylabel('Экономия ($/час)')
    
    plt.tight_layout()
    plt.savefig('energy_analysis_report.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================
# 5. ОСНОВНАЯ ЛОГИКА
# ========================

def main():
    print("=" * 60)
    print("СИСТЕМА ПОДДЕРЖКИ РЕШЕНИЙ ДЛЯ ОПТИМИЗАЦИИ ЭНЕРГОПОТРЕБЛЕНИЯ")
    print("=" * 60)
    
    # 1. Генерация данных
    print("\n1. Загрузка и подготовка данных...")
    data = generate_energy_data(days=14, freq='H')
    print(f"   Загружено {len(data)} записей")
    print(f"   Период: {data['timestamp'].min()} - {data['timestamp'].max()}")
    
    # 2. Обнаружение аномалий
    print("\n2. Анализ энергопотребления...")
    detector = EnergyAnomalyDetector(window_size=24)
    analyzed_data = detector.detect_anomalies(data)
    
    anomalies = analyzed_data[analyzed_data['is_detected'] == 1]
    print(f"   Обнаружено аномалий: {len(anomalies)}")
    
    if not anomalies.empty:
        avg_deviation = (anomalies['energy_kWh'] - anomalies['rolling_mean']).mean()
        print(f"   Среднее отклонение: {avg_deviation:.0f} кВт")
    
    # 3. Генерация рекомендаций
    print("\n3. Формирование рекомендаций...")
    advisor = EnergyAdvisor(electricity_cost=0.085)
    recommendations = []
    
    for idx, row in anomalies.iterrows():
        baseline = row['rolling_mean'] if not pd.isna(row['rolling_mean']) else row['energy_kWh']
        rec = advisor.generate_recommendation(row, baseline)
        if rec:
            recommendations.append(rec)
    
    # 4. Вывод рекомендаций
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ ДЛЯ ОПЕРАТОРА:")
    print("=" * 60)
    
    if not recommendations:
        print("✓ Аномалий не обнаружено. Энергопотребление в норме.")
    else:
        total_potential_saving = sum([r['hourly_cost_saving'] for r in recommendations])
        annual_saving = total_potential_saving * 24 * 365 / len(data) * 24  # Экстраполяция
        
        for i, rec in enumerate(recommendations[:5], 1):  # Покажем первые 5
            print(f"\n⚠️  Аномалия #{i} ({rec['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
            print(f"   Текущее потребление: {rec['current_power_kW']:.0f} кВт")
            print(f"   Ожидаемое: {rec['baseline_kW']:.0f} кВт")
            print(f"   Избыточное потребление: {rec['excess_kW']:.0f} кВт")
            print(f"   Рекомендация: {rec['recommendation']}")
            print(f"   Потенциальная экономия: ${rec['hourly_cost_saving']:.2f}/час")
            print(f"   Уверенность системы: {rec['confidence']*100:.0f}%")
        
        print(f"\n📊 СВОДКА:")
        print(f"   Всего рекомендаций: {len(recommendations)}")
        print(f"   Общая потенциальная экономия: ${total_potential_saving:.2f}/час")
        print(f"   Прогноз годовой экономии: ${annual_saving:,.0f}")
        
        # Пример расчета для презентации
        print(f"\n📈 ПРИМЕР РАСЧЕТА ЭФФЕКТИВНОСТИ:")
        print(f"   Если система предотвращает 1 аномалию в день (500 кВт × 2 часа):")
        daily_saving = 500 * 2 * 0.085  # кВт * часы * $/кВт·ч
        print(f"   Ежедневная экономия: ${daily_saving:.2f}")
        print(f"   Годовая экономия: ${daily_saving * 365:,.0f}")
    
    # 5. Визуализация
    print("\n4. Генерация отчета...")
    visualize_results(analyzed_data, recommendations)
    
    # 6. Экспорт результатов
    if recommendations:
        report_df = pd.DataFrame(recommendations)
        report_df.to_csv('energy_recommendations.csv', index=False, encoding='utf-8-sig')
        print(f"\n📁 Результаты сохранены в файлы:")
        print(f"   - energy_recommendations.csv (рекомендации)")
        print(f"   - energy_analysis_report.png (визуализации)")
    
    print("\n" + "=" * 60)
    print("СИСТЕМА АНАЛИЗА ЗАВЕРШИЛА РАБОТУ")
    print("=" * 60)

# ========================
# ЗАПУСК ПРОГРАММЫ
# ========================

if __name__ == "__main__":
    main()