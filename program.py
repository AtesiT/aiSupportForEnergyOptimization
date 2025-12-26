import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import time
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. TEST DATA GENERATION
# ========================

def generate_energy_data(days=30, freq='H'):
    """Generate synthetic energy consumption data"""
    np.random.seed(42)
    
    # Base consumption with daily cycle
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                               end=datetime.now(), freq=freq)
    n_points = len(timestamps)
    
    # Base consumption level (kW)
    base_load = 5000  # 5 MW base load
    
    # Daily cycle (night/day)
    daily_cycle = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    
    # Seasonality (weekdays/weekends)
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    weekend_effect = np.where(day_of_week >= 5, -1000, 0)
    
    # Random fluctuations
    random_noise = np.random.normal(0, 300, n_points)
    
    # Generate anomalies (sudden consumption spikes)
    anomalies = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.03), replace=False)
    anomalies[anomaly_indices] = np.random.uniform(1000, 4000, len(anomaly_indices))
    
    # Final consumption
    consumption = base_load + daily_cycle + weekend_effect + random_noise + anomalies
    
    # Temperature (for context)
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 3, n_points)
    
    # Production parameters
    production_rate = np.random.uniform(70, 100, n_points)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'energy_kWh': consumption,
        'temperature_C': temperature,
        'production_rate': production_rate,
        'is_anomaly': (anomalies > 0).astype(int)
    })

# ========================
# 2. ANALYTICAL MODULE
# ========================

class EnergyAnomalyDetector:
    """Energy consumption anomaly detector"""
    
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.model = IsolationForest(contamination=0.05, random_state=42)
        
    def create_features(self, df):
        """Create features for ML model"""
        df = df.copy()
        
        # Rolling window statistics
        df['rolling_mean'] = df['energy_kWh'].rolling(window=self.window_size).mean()
        df['rolling_std'] = df['energy_kWh'].rolling(window=self.window_size).std()
        df['z_score'] = (df['energy_kWh'] - df['rolling_mean']) / df['rolling_std']
        
        # Derived features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Deviation from typical pattern
        typical_by_hour = df.groupby('hour')['energy_kWh'].mean()
        df['deviation_from_typical'] = df.apply(
            lambda row: row['energy_kWh'] - typical_by_hour[row['hour']], axis=1
        )
        
        return df
    
    def detect_anomalies(self, df):
        """Detect anomalies using ML"""
        feature_cols = ['energy_kWh', 'z_score', 'deviation_from_typical', 
                       'production_rate', 'temperature_C']
        
        # Prepare data
        df_features = self.create_features(df)
        feature_data = df_features[feature_cols].fillna(0)
        
        # Train model and predict
        predictions = self.model.fit_predict(feature_data)
        df_features['ml_anomaly'] = np.where(predictions == -1, 1, 0)
        
        # Rule-based detection (z-score)
        df_features['rule_anomaly'] = np.where(
            abs(df_features['z_score'].fillna(0)) > 3, 1, 0
        )
        
        # Combined result
        df_features['is_detected'] = np.where(
            (df_features['ml_anomaly'] == 1) | (df_features['rule_anomaly'] == 1), 1, 0
        )
        
        return df_features

# ========================
# 3. RECOMMENDATION MODULE
# ========================

class EnergyAdvisor:
    """Energy saving recommendation generator"""
    
    def __init__(self, electricity_cost=0.08):
        self.electricity_cost = electricity_cost  # $/kWh
        
    def generate_recommendation(self, row, baseline):
        """Generate specific recommendation"""
        current_power = row['energy_kWh']
        deviation = current_power - baseline
        excess_kwh = max(0, deviation)
        
        if excess_kwh == 0:
            return None
        
        # Calculate potential hourly savings
        hourly_saving = excess_kwh * self.electricity_cost
        
        # Time-based recommendations
        hour = row['timestamp'].hour
        
        if hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night hours
            recommendations = [
                "Switch non-essential equipment to standby mode",
                "Optimize ventilation system operation by 30%",
                "Turn off lighting in unused areas"
            ]
        elif hour in [6, 7, 8, 9, 10, 11]:  # Morning hours
            recommendations = [
                "Check startup schedule for high-power equipment",
                "Adjust temperature setpoints by 2°C",
                "Consolidate production batches for efficiency"
            ]
        else:  # Day/Evening hours
            recommendations = [
                "Run diagnostic on compressor #3",
                f"Check system pressure (current deviation: {deviation:.0f} kW)",
                "Consider postponing non-critical processes by 2 hours"
            ]
        
        # Select recommendation based on deviation size
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
# 4. VISUALIZATION
# ========================

def visualize_results(df, recommendations):
    """Visualize analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Chart 1: Energy consumption with anomalies
    axes[0, 0].plot(df['timestamp'], df['energy_kWh'], label='Consumption', alpha=0.7)
    axes[0, 0].plot(df['timestamp'], df['rolling_mean'], label='Baseline', linestyle='--')
    
    anomaly_points = df[df['is_detected'] == 1]
    if not anomaly_points.empty:
        axes[0, 0].scatter(anomaly_points['timestamp'], anomaly_points['energy_kWh'], 
                          color='red', s=50, label='Anomalies', zorder=5)
    
    axes[0, 0].set_title('Energy Consumption with Anomaly Detection')
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Z-score
    axes[0, 1].plot(df['timestamp'], df['z_score'].fillna(0))
    axes[0, 1].axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Threshold (z=3)')
    axes[0, 1].axhline(y=-3, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Z-score Deviation from Normal')
    axes[0, 1].set_ylabel('Z-score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Anomaly distribution by hour
    if not anomaly_points.empty:
        hour_counts = anomaly_points['timestamp'].dt.hour.value_counts().sort_index()
        axes[1, 0].bar(hour_counts.index, hour_counts.values)
        axes[1, 0].set_title('Anomaly Distribution by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Number of Anomalies')
        axes[1, 0].set_xticks(range(0, 24, 2))
    
    # Chart 4: Potential savings
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        axes[1, 1].bar(range(len(rec_df)), rec_df['hourly_cost_saving'])
        axes[1, 1].set_title('Potential Savings from Recommendations')
        axes[1, 1].set_xlabel('Recommendation #')
        axes[1, 1].set_ylabel('Savings ($/hour)')
    
    plt.tight_layout()
    plt.savefig('energy_analysis_report.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================
# 5. ADDITIONAL: DEMO MODE
# ========================

def simulate_realtime_monitoring():
    """Real-time monitoring demo mode"""
    print("\n🎮 DEMO MODE: Real-time Monitoring Simulation")
    print("Simulating incoming data every 5 seconds...")
    
    for i in range(10):
        # Generate "current" reading
        current_time = datetime.now()
        base_power = 5000
        current_power = base_power + np.random.normal(0, 200)
        
        # Random anomaly
        if np.random.random() < 0.2:
            anomaly_size = np.random.uniform(800, 2000)
            current_power += anomaly_size
            print(f"\n🚨 ALERT! Anomaly detected at {current_time.strftime('%H:%M:%S')}")
            print(f"   Current consumption: {current_power:.0f} kW")
            print(f"   Deviation: +{anomaly_size:.0f} kW")
            
            # Simple recommendation
            if anomaly_size > 1500:
                print(f"   RECOMMENDATION: Check main compressor and furnace operation")
            else:
                print(f"   RECOMMENDATION: Adjust ventilation system settings")
            
            # Savings calculation
            potential_saving = anomaly_size * 0.085  # 0.085 $/kWh
            print(f"   Potential savings: ${potential_saving:.2f}/hour")
        else:
            print(f"   {current_time.strftime('%H:%M:%S')}: Normal consumption ({current_power:.0f} kW)")
        
        time.sleep(0.5)  # Simulate delay
    
    print("\n✅ Demo mode completed")

# ========================
# 6. MAIN LOGIC
# ========================

def main():
    print("=" * 60)
    print("ENERGY OPTIMIZATION DECISION SUPPORT SYSTEM")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1. Loading and preparing data...")
    data = generate_energy_data(days=14, freq='H')
    print(f"   Loaded {len(data)} records")
    print(f"   Period: {data['timestamp'].min()} - {data['timestamp'].max()}")
    
    # 2. Detect anomalies
    print("\n2. Analyzing energy consumption...")
    detector = EnergyAnomalyDetector(window_size=24)
    analyzed_data = detector.detect_anomalies(data)
    
    anomalies = analyzed_data[analyzed_data['is_detected'] == 1]
    print(f"   Anomalies detected: {len(anomalies)}")
    
    if not anomalies.empty:
        avg_deviation = (anomalies['energy_kWh'] - anomalies['rolling_mean']).mean()
        print(f"   Average deviation: {avg_deviation:.0f} kW")
    
    # 3. Generate recommendations
    print("\n3. Generating recommendations...")
    advisor = EnergyAdvisor(electricity_cost=0.085)
    recommendations = []
    
    for idx, row in anomalies.iterrows():
        baseline = row['rolling_mean'] if not pd.isna(row['rolling_mean']) else row['energy_kWh']
        rec = advisor.generate_recommendation(row, baseline)
        if rec:
            recommendations.append(rec)
    
    # 4. Output recommendations
    print("\n" + "=" * 60)
    print("OPERATOR RECOMMENDATIONS:")
    print("=" * 60)
    
    if not recommendations:
        print("✓ No anomalies detected. Energy consumption is normal.")
    else:
        total_potential_saving = sum([r['hourly_cost_saving'] for r in recommendations])
        annual_saving = total_potential_saving * 24 * 365 / len(data) * 24  # Extrapolation
        
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            print(f"\n⚠️  Anomaly #{i} ({rec['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
            print(f"   Current consumption: {rec['current_power_kW']:.0f} kW")
            print(f"   Expected: {rec['baseline_kW']:.0f} kW")
            print(f"   Excess consumption: {rec['excess_kW']:.0f} kW")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Potential savings: ${rec['hourly_cost_saving']:.2f}/hour")
            print(f"   System confidence: {rec['confidence']*100:.0f}%")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total recommendations: {len(recommendations)}")
        print(f"   Total potential savings: ${total_potential_saving:.2f}/hour")
        print(f"   Projected annual savings: ${annual_saving:,.0f}")
        
        # Example calculation for presentation
        print(f"\n📈 SAMPLE EFFICIENCY CALCULATION:")
        print(f"   If system prevents 1 anomaly per day (500 kW × 2 hours):")
        daily_saving = 500 * 2 * 0.085  # kW * hours * $/kWh
        print(f"   Daily savings: ${daily_saving:.2f}")
        print(f"   Annual savings: ${daily_saving * 365:,.0f}")
    
    # 5. Visualization
    print("\n4. Generating report...")
    visualize_results(analyzed_data, recommendations)
    
    # 6. Export results
    if recommendations:
        report_df = pd.DataFrame(recommendations)
        report_df.to_csv('energy_recommendations.csv', index=False, encoding='utf-8')
        print(f"\n📁 Results saved to files:")
        print(f"   - energy_recommendations.csv (recommendations)")
        print(f"   - energy_analysis_report.png (visualizations)")
    
    # 7. Demo mode
    simulate_realtime_monitoring()
    
    print("\n" + "=" * 60)
    print("SYSTEM ANALYSIS COMPLETED")
    print("=" * 60)

# ========================
# PROGRAM EXECUTION
# ========================

if __name__ == "__main__":
    main()