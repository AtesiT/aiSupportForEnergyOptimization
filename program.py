import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import csv
import warnings
warnings.filterwarnings('ignore')

# ========================
# 1. TEST DATA GENERATION
# ========================

def generate_energy_data(days=30, freq='H'):
    """Generate synthetic energy consumption data"""
    np.random.seed(42)
    
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                               end=datetime.now(), freq=freq)
    n_points = len(timestamps)
    
    # Base consumption level (kW)
    base_load = 5000
    
    # Daily cycle (night/day)
    daily_cycle = 2000 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    
    # Seasonality (weekdays/weekends)
    day_of_week = np.array([ts.weekday() for ts in timestamps])
    weekend_effect = np.where(day_of_week >= 5, -1000, 0)
    
    # Random fluctuations
    random_noise = np.random.normal(0, 300, n_points)
    
    # Generate anomalies
    anomalies = np.zeros(n_points)
    anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.03), replace=False)
    anomalies[anomaly_indices] = np.random.uniform(1000, 4000, len(anomaly_indices))
    
    # Final consumption
    consumption = base_load + daily_cycle + weekend_effect + random_noise + anomalies
    
    # Additional parameters
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 3, n_points)
    production_rate = np.random.uniform(70, 100, n_points)
    
    # Equipment data
    equipment_names = ['Compressor_A', 'Compressor_B', 'Pump_1', 'Pump_2', 'Furnace_1']
    equipment_data = {}
    
    for equipment in equipment_names:
        if 'Compressor' in equipment:
            base = np.random.uniform(300, 500, n_points)
        elif 'Pump' in equipment:
            base = np.random.uniform(100, 250, n_points)
        else:  # Furnace
            base = np.random.uniform(800, 1200, n_points)
        
        # Add some anomalies to equipment
        equipment_anomaly = np.zeros(n_points)
        anomaly_chance = np.random.random(n_points) < 0.02
        equipment_anomaly[anomaly_chance] = base[anomaly_chance] * np.random.uniform(0.1, 0.3, anomaly_chance.sum())
        
        equipment_data[equipment] = base + equipment_anomaly
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_kWh': consumption,
        'temperature_C': temperature,
        'production_rate': production_rate,
        'is_anomaly': (anomalies > 0).astype(int)
    })
    
    # Add equipment columns
    for equipment, data in equipment_data.items():
        df[f'{equipment}_kW'] = data
    
    return df, equipment_names

# ========================
# 2. ANALYTICAL MODULE
# ========================

class EnergyAnomalyDetector:
    """Energy consumption anomaly detector"""
    
    def __init__(self, window_size=24):
        self.window_size = window_size
        self.model = IsolationForest(contamination=0.05, random_state=42)
        
    def detect_anomalies(self, df):
        """Detect anomalies using ML"""
        df = df.copy()
        
        # Calculate rolling statistics
        df['rolling_mean'] = df['energy_kWh'].rolling(window=self.window_size).mean()
        df['rolling_std'] = df['energy_kWh'].rolling(window=self.window_size).std()
        df['z_score'] = (df['energy_kWh'] - df['rolling_mean']) / df['rolling_std']
        
        # Add time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Prepare features for ML
        feature_cols = ['energy_kWh', 'z_score', 'temperature_C', 'production_rate']
        feature_data = df[feature_cols].fillna(0)
        
        # Train model and predict
        predictions = self.model.fit_predict(feature_data)
        df['ml_anomaly'] = np.where(predictions == -1, 1, 0)
        
        # Rule-based detection
        df['rule_anomaly'] = np.where(abs(df['z_score'].fillna(0)) > 3, 1, 0)
        
        # Combined detection
        df['is_detected'] = np.where(
            (df['ml_anomaly'] == 1) | (df['rule_anomaly'] == 1), 1, 0
        )
        
        # Calculate cost
        df['hourly_cost_usd'] = df['energy_kWh'] * 0.085  # $0.085 per kWh
        
        return df

# ========================
# 3. RECOMMENDATION MODULE
# ========================

class EnergyAdvisor:
    """Energy saving recommendation generator"""
    
    def __init__(self, electricity_cost=0.085):
        self.electricity_cost = electricity_cost
        
    def generate_recommendations(self, df):
        """Generate recommendations for anomalies"""
        anomalies = df[df['is_detected'] == 1]
        recommendations = []
        
        for idx, row in anomalies.iterrows():
            baseline = row['rolling_mean'] if not pd.isna(row['rolling_mean']) else row['energy_kWh']
            deviation = row['energy_kWh'] - baseline
            
            if deviation <= 0:
                continue
            
            # Calculate savings
            hourly_saving = deviation * self.electricity_cost
            
            # Generate recommendation based on time and deviation
            hour = row['hour']
            
            if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                recommendation = "Switch non-essential equipment to standby mode"
            elif hour in [6, 7, 8, 9, 10, 11]:
                recommendation = "Check startup schedule for high-power equipment"
            else:
                recommendation = f"Investigate consumption spike of {deviation:.0f} kW"
            
            recommendations.append({
                'timestamp': row['timestamp'],
                'hour': hour,
                'current_power_kW': row['energy_kWh'],
                'expected_power_kW': baseline,
                'deviation_kW': deviation,
                'hourly_saving_usd': hourly_saving,
                'daily_saving_usd': hourly_saving * 24,
                'annual_saving_usd': hourly_saving * 8760,
                'recommendation': recommendation,
                'priority': 'High' if deviation > 1000 else 'Medium'
            })
        
        return pd.DataFrame(recommendations)

# ========================
# 4. SINGLE CSV EXPORTER
# ========================

class SingleCSVExporter:
    """Export all data to a single structured CSV file with sections"""
    
    @staticmethod
    def create_section_header(writer, header):
        """Create a section header in CSV"""
        writer.writerow([])  # Empty row
        writer.writerow([f"=== {header.upper()} ==="])
        writer.writerow([])  # Empty row
    
    @staticmethod
    def export_all_to_single_csv(df, recommendations, equipment_names, filename='energy_analysis_report.csv'):
        """Export all data to a single CSV file with multiple sections"""
        
        # Open CSV file for writing
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # ============================================
            # SECTION 1: ANALYSIS SUMMARY
            # ============================================
            writer.writerow(["ENERGY ANALYSIS REPORT"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([f"Analysis Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"])
            
            SingleCSVExporter.create_section_header(writer, "EXECUTIVE SUMMARY")
            
            # Calculate summary statistics
            total_energy = df['energy_kWh'].sum()
            avg_energy = df['energy_kWh'].mean()
            max_energy = df['energy_kWh'].max()
            min_energy = df['energy_kWh'].min()
            total_cost = total_energy * 0.085
            
            anomaly_count = df['is_detected'].sum()
            anomaly_percent = (anomaly_count / len(df)) * 100
            
            writer.writerow(["Metric", "Value", "Unit"])
            writer.writerow(["Total Records", len(df), "entries"])
            writer.writerow(["Total Energy Consumption", f"{total_energy:,.0f}", "kWh"])
            writer.writerow(["Average Hourly Consumption", f"{avg_energy:.0f}", "kW"])
            writer.writerow(["Peak Consumption", f"{max_energy:.0f}", "kW"])
            writer.writerow(["Minimum Consumption", f"{min_energy:.0f}", "kW"])
            writer.writerow(["Total Energy Cost", f"${total_cost:,.0f}", "USD"])
            writer.writerow(["Anomalies Detected", anomaly_count, "events"])
            writer.writerow(["Anomaly Rate", f"{anomaly_percent:.1f}", "%"])
            
            # ============================================
            # SECTION 2: COST SAVINGS ANALYSIS
            # ============================================
            SingleCSVExporter.create_section_header(writer, "COST SAVINGS ANALYSIS")
            
            if not recommendations.empty:
                total_hourly_saving = recommendations['hourly_saving_usd'].sum()
                total_annual_saving = recommendations['annual_saving_usd'].sum()
                
                writer.writerow(["Metric", "Value", "Unit"])
                writer.writerow(["Total Potential Hourly Savings", f"${total_hourly_saving:.2f}", "USD/hour"])
                writer.writerow(["Total Potential Daily Savings", f"${total_hourly_saving*24:.2f}", "USD/day"])
                writer.writerow(["Total Potential Annual Savings", f"${total_annual_saving:,.0f}", "USD/year"])
                writer.writerow(["Estimated ROI (if system cost $50,000)", f"{50000/total_annual_saving:.1f}", "years"])
            else:
                writer.writerow(["No significant anomalies detected for cost savings calculation"])
            
            # ============================================
            # SECTION 3: HOURLY DATA
            # ============================================
            SingleCSVExporter.create_section_header(writer, "HOURLY CONSUMPTION DATA")
            
            # Prepare hourly data
            hourly_df = df[['timestamp', 'energy_kWh', 'temperature_C', 'production_rate', 
                           'rolling_mean', 'z_score', 'is_detected', 'hourly_cost_usd']].copy()
            hourly_df['timestamp'] = hourly_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            hourly_df.columns = ['Timestamp', 'Energy (kWh)', 'Temperature (°C)', 'Production Rate (%)',
                                'Expected (kWh)', 'Z-Score', 'Anomaly Flag', 'Hourly Cost (USD)']
            
            # Write hourly data headers
            writer.writerow(hourly_df.columns.tolist())
            
            # Write hourly data rows
            for _, row in hourly_df.iterrows():
                writer.writerow(row.tolist())
            
            # ============================================
            # SECTION 4: ANOMALIES AND RECOMMENDATIONS
            # ============================================
            SingleCSVExporter.create_section_header(writer, "ANOMALIES & RECOMMENDATIONS")
            
            if not recommendations.empty:
                # Prepare recommendations data
                rec_df = recommendations.copy()
                rec_df['timestamp'] = rec_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Write recommendations headers
                rec_headers = ['Timestamp', 'Hour', 'Current (kW)', 'Expected (kW)', 'Deviation (kW)',
                              'Hourly Saving (USD)', 'Annual Saving (USD)', 'Recommendation', 'Priority']
                writer.writerow(rec_headers)
                
                # Write recommendations rows
                for _, row in rec_df.iterrows():
                    writer.writerow([
                        row['timestamp'],
                        int(row['hour']),
                        f"{row['current_power_kW']:.0f}",
                        f"{row['expected_power_kW']:.0f}",
                        f"{row['deviation_kW']:.0f}",
                        f"${row['hourly_saving_usd']:.2f}",
                        f"${row['annual_saving_usd']:,.0f}",
                        row['recommendation'],
                        row['priority']
                    ])
            else:
                writer.writerow(["No actionable anomalies detected during analysis period"])
            
            # ============================================
            # SECTION 5: EQUIPMENT ANALYSIS
            # ============================================
            SingleCSVExporter.create_section_header(writer, "EQUIPMENT ANALYSIS")
            
            # Calculate equipment statistics
            equipment_stats = []
            for equipment in equipment_names:
                col_name = f'{equipment}_kW'
                if col_name in df.columns:
                    total_power = df[col_name].sum()
                    avg_power = df[col_name].mean()
                    max_power = df[col_name].max()
                    percent_of_total = (total_power / df['energy_kWh'].sum()) * 100
                    
                    equipment_stats.append({
                        'Equipment': equipment,
                        'Total (kWh)': f"{total_power:,.0f}",
                        'Average (kW)': f"{avg_power:.0f}",
                        'Peak (kW)': f"{max_power:.0f}",
                        '% of Total': f"{percent_of_total:.1f}%",
                        'Cost (USD)': f"${total_power * 0.085:,.0f}"
                    })
            
            if equipment_stats:
                # Write equipment headers
                writer.writerow(['Equipment', 'Total Consumption (kWh)', 'Average Power (kW)', 
                                'Peak Power (kW)', '% of Total', 'Estimated Cost (USD)'])
                
                # Write equipment rows
                for stat in equipment_stats:
                    writer.writerow([
                        stat['Equipment'],
                        stat['Total (kWh)'],
                        stat['Average (kW)'],
                        stat['Peak (kW)'],
                        stat['% of Total'],
                        stat['Cost (USD)']
                    ])
            
            # ============================================
            # SECTION 6: DAILY SUMMARY
            # ============================================
            SingleCSVExporter.create_section_header(writer, "DAILY SUMMARY")
            
            # Calculate daily statistics
            df['date'] = df['timestamp'].dt.date
            daily_stats = df.groupby('date').agg({
                'energy_kWh': ['sum', 'mean', 'max'],
                'is_detected': 'sum',
                'hourly_cost_usd': 'sum'
            }).round(2)
            
            # Flatten columns
            daily_stats.columns = ['Total (kWh)', 'Average (kW)', 'Peak (kW)', 'Anomalies', 'Daily Cost (USD)']
            daily_stats = daily_stats.reset_index()
            
            # Write daily headers
            writer.writerow(['Date', 'Total Energy (kWh)', 'Average Power (kW)', 
                            'Peak Power (kW)', 'Anomalies', 'Daily Cost (USD)'])
            
            # Write daily rows
            for _, row in daily_stats.iterrows():
                writer.writerow([
                    row['date'].strftime('%Y-%m-%d'),
                    f"{row['Total (kWh)']:,.0f}",
                    f"{row['Average (kW)']:.0f}",
                    f"{row['Peak (kW)']:.0f}",
                    int(row['Anomalies']),
                    f"${row['Daily Cost (USD)']:,.0f}"
                ])
            
            # ============================================
            # SECTION 7: CALCULATION REFERENCE
            # ============================================
            SingleCSVExporter.create_section_header(writer, "CALCULATION REFERENCE")
            
            writer.writerow(["Parameter", "Value", "Description"])
            writer.writerow(["Electricity Cost", "$0.085/kWh", "Industrial tariff rate"])
            writer.writerow(["Anomaly Threshold", "Z-Score > 3", "Standard deviation threshold"])
            writer.writerow(["Analysis Window", "24 hours", "Rolling average calculation period"])
            writer.writerow(["Hours per Year", "8,760", "For annual calculations"])
            writer.writerow(["Days per Year", "365", "For annual calculations"])
            
            # ============================================
            # SECTION 8: KEY INSIGHTS
            # ============================================
            SingleCSVExporter.create_section_header(writer, "KEY INSIGHTS")
            
            # Generate insights
            insights = [
                f"1. Peak consumption occurs between {df.loc[df['energy_kWh'].idxmax(), 'hour']}:00 and {(df.loc[df['energy_kWh'].idxmax(), 'hour'] + 1) % 24}:00 hours",
                f"2. {anomaly_percent:.1f}% of readings showed abnormal consumption patterns",
                f"3. Average temperature during operation: {df['temperature_C'].mean():.1f}°C",
                f"4. Production rate averaged {df['production_rate'].mean():.1f}% of capacity"
            ]
            
            if not recommendations.empty:
                top_saving = recommendations.nlargest(1, 'hourly_saving_usd').iloc[0]
                insights.append(f"5. Highest saving opportunity: {top_saving['deviation_kW']:.0f} kW at {int(top_saving['hour'])}:00")
            
            for insight in insights:
                writer.writerow([insight])
            
            print(f"\n✅ All data exported to single file: {filename}")
            print(f"   File contains {len(df)} hourly records, {len(recommendations)} recommendations")
            
            return filename

# ========================
# 5. VISUALIZATION
# ========================

def create_visualizations(df, recommendations):
    """Create visualization charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chart 1: Energy consumption
    axes[0, 0].plot(df['timestamp'], df['energy_kWh'], label='Actual', alpha=0.7)
    axes[0, 0].plot(df['timestamp'], df['rolling_mean'], label='Expected', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Energy Consumption Analysis')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Anomaly detection
    anomaly_points = df[df['is_detected'] == 1]
    axes[0, 1].plot(df['timestamp'], df['z_score'].fillna(0), alpha=0.7)
    axes[0, 1].axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Threshold (z=3)')
    axes[0, 1].axhline(y=-3, color='r', linestyle='--', alpha=0.5)
    if not anomaly_points.empty:
        axes[0, 1].scatter(anomaly_points['timestamp'], anomaly_points['z_score'].fillna(0), 
                          color='red', s=20, label='Anomalies', zorder=5)
    axes[0, 1].set_title('Anomaly Detection (Z-Score)')
    axes[0, 1].set_ylabel('Z-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Daily pattern
    hourly_avg = df.groupby('hour')['energy_kWh'].mean()
    axes[1, 0].bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
    axes[1, 0].set_title('Average Daily Consumption Pattern')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Power (kW)')
    axes[1, 0].set_xticks(range(0, 24, 3))
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Chart 4: Potential savings
    if not recommendations.empty:
        savings_by_hour = recommendations.groupby('hour')['hourly_saving_usd'].sum()
        axes[1, 1].bar(savings_by_hour.index, savings_by_hour.values, alpha=0.7, color='green')
        axes[1, 1].set_title('Potential Savings by Hour')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Savings ($/hour)')
        axes[1, 1].set_xticks(range(0, 24, 3))
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('energy_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================
# 6. MAIN PROGRAM
# ========================

def main():
    print("=" * 70)
    print("ENERGY OPTIMIZATION DECISION SUPPORT SYSTEM")
    print("Single CSV Export Version")
    print("=" * 70)
    
    # 1. Generate data
    print("\n📊 1. Generating synthetic energy data...")
    data, equipment_names = generate_energy_data(days=30, freq='H')
    print(f"   • Time period: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
    print(f"   • Total records: {len(data)} hourly readings")
    print(f"   • Equipment monitored: {', '.join(equipment_names)}")
    
    # 2. Detect anomalies
    print("\n🔍 2. Analyzing consumption patterns...")
    detector = EnergyAnomalyDetector(window_size=24)
    analyzed_data = detector.detect_anomalies(data)
    
    anomalies = analyzed_data[analyzed_data['is_detected'] == 1]
    print(f"   • Anomalies detected: {len(anomalies)}")
    
    if not anomalies.empty:
        avg_deviation = (anomalies['energy_kWh'] - anomalies['rolling_mean']).mean()
        print(f"   • Average deviation: {avg_deviation:.0f} kW")
    
    # 3. Generate recommendations
    print("\n💡 3. Generating energy saving recommendations...")
    advisor = EnergyAdvisor(electricity_cost=0.085)
    recommendations = advisor.generate_recommendations(analyzed_data)
    
    print(f"   • Recommendations generated: {len(recommendations)}")
    
    if not recommendations.empty:
        total_savings = recommendations['annual_saving_usd'].sum()
        print(f"   • Total potential annual savings: ${total_savings:,.0f}")
    
    # 4. Export to single CSV
    print("\n💾 4. Exporting all data to single CSV file...")
    exporter = SingleCSVExporter()
    csv_file = exporter.export_all_to_single_csv(
        analyzed_data, 
        recommendations, 
        equipment_names,
        filename='energy_decision_support_report.csv'
    )
    
    # 5. Create visualizations
    print("\n📈 5. Creating visualization charts...")
    create_visualizations(analyzed_data, recommendations)
    
    # 6. Print summary
    print("\n" + "=" * 70)
    print("REPORT SUMMARY")
    print("=" * 70)
    
    total_energy = analyzed_data['energy_kWh'].sum()
    total_cost = total_energy * 0.085
    
    print(f"📅 Analysis Period: {analyzed_data['timestamp'].min().date()} to {analyzed_data['timestamp'].max().date()}")
    print(f"⚡ Total Energy: {total_energy:,.0f} kWh")
    print(f"💰 Total Cost: ${total_cost:,.0f}")
    print(f"🚨 Anomalies: {len(anomalies)} events ({len(anomalies)/len(analyzed_data)*100:.1f}%)")
    
    if not recommendations.empty:
        annual_savings = recommendations['annual_saving_usd'].sum()
        savings_percent = (annual_savings / total_cost) * 100
        print(f"💵 Potential Annual Savings: ${annual_savings:,.0f}")
        print(f"📊 Savings Potential: {savings_percent:.1f}% of current cost")
        
        # Top recommendation
        top_rec = recommendations.nlargest(1, 'hourly_saving_usd').iloc[0]
        print(f"🎯 Top Opportunity: {top_rec['deviation_kW']:.0f} kW at {int(top_rec['hour'])}:00")
        print(f"   Recommendation: {top_rec['recommendation']}")
        print(f"   Potential: ${top_rec['annual_saving_usd']:,.0f}/year")
    
    print(f"\n📁 Report File: {csv_file}")
    print("📊 Visualization: energy_visualization.png")
    print("=" * 70)
    print("✅ DSS Analysis Complete. Ready for operator review.")
    print("=" * 70)

# ========================
# RUN PROGRAM
# ========================

if __name__ == "__main__":
    main()