import os
import sys
from pathlib import Path
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# --- [ PATH SETUP AND IMPORT FIXES ] ---
project_root = Path(__file__).resolve().parent
framework_path = project_root / "EnergyForecaster"

# Add project root and framework folder to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(framework_path))

from EnergyForecaster.EnergyForecasterMain import EnergyForecaster
from EnergyForecaster.DataController import DataController
from EnergyForecaster.ProcessController import Process

# --- [ PATH DEFINITIONS ] ---
data_path = project_root / "data"
project_path = project_root / "benchmark_project"
h5_file = project_path / "benchmark_project.h5"

weather_csv = data_path / "weather.csv"
cons_csv = data_path / "consumption.csv"

os.makedirs(project_path / "data", exist_ok=True)
os.makedirs(project_path / "models", exist_ok=True)

assert weather_csv.exists(), f"❌ Missing: {weather_csv}"
assert cons_csv.exists(), f"❌ Missing: {cons_csv}"

# --- [ CREATE .h5 IF NEEDED ] ---
if not h5_file.exists():
    print("🟡 Creating placeholder .h5 file...")
    with h5py.File(h5_file, 'w') as f:
        f.attrs['__SIGNATURE__'] = 'Energy Forecaster Framework'
    print("✅ .h5 file created")

    dummy = type("EF", (), {})()
    dummy.data_statistics = None
    dummy.preprocessor = None
    dummy.results_statistics = None
    dummy.results_visualizer = None
    dummy.data_visualizer = None
    dummy.process_controller = None

    dc = DataController(dummy, str(project_path))
    dc.import_csv(str(weather_csv), h5_name="weather")
    dc.import_csv(str(cons_csv), h5_name="consumption")
    print("✅ CSVs imported and stored")

# --- [ INIT FRAMEWORK ] ---
ef = EnergyForecaster(str(project_path))
print("✅ EnergyForecaster initialized")

# --- [ LOAD DATASETS (no renaming) ] ---
dt_weather = ef.data_controller.get_dataset('weather', in_line=True)
dt_consumption = ef.data_controller.get_dataset('consumption', in_line=True)

# --- [ PARSE TIMESTAMPS ] ---
dt_weather.data['time'] = dt_weather.data['time'].astype(str)
dt_consumption.data['utc_timestamp'] = dt_consumption.data['utc_timestamp'].astype(str)

# dt_weather.to_timestamp('time', '%Y-%m-%d %H:%M:%S', 'Europe/Athens', assign='inplace')
# dt_consumption.to_timestamp('utc_timestamp', '%Y-%m-%dT%H:%M:%SZ', 'UTC', assign='inplace')

# --- [ SCALE + TARGET SETUP ] ---
dt_weather.make_scale('time')
dt_consumption.make_scale('utc_timestamp')

# Fill NaNs in temperature and consumption using linear interpolation
dt_weather.fill_linear('temperature', assign='inplace')
dt_consumption.fill_linear('ES_load_actual_entsoe_transparency', assign='inplace')

dt_weather.attach_scale('temperature', 'time')
dt_consumption.attach_scale('ES_load_actual_entsoe_transparency', 'utc_timestamp')
dt_consumption.make_target('ES_load_actual_entsoe_transparency')

# --- [ SAVE TO MEMORY + FILE ] ---
ef.data_controller.datasets['weather'] = dt_weather
ef.data_controller.datasets['consumption'] = dt_consumption

ef.data_controller.update_dataset('weather')
ef.data_controller.update_dataset('consumption')
print("✅ Datasets updated")

# --- [ CREATE PROCESS ] ---
proc = Process(name="benchmark_proc", EF=ef)
ef.data_controller.set_process(proc)
print(f"✅ Process '{proc.name}' initialized")


# --- [ INSERT DATA USING ORIGINAL COLUMN NAMES ] ---
proc.insert_data('weather', ['temperature'])
proc.insert_data('consumption', ['ES_load_actual_entsoe_transparency'])

# --- [ REGISTER MODELS IF NEEDED ] ---
model_names = ef.data_controller.get_model_names()
if 'mlp_100' not in model_names:
    ef.data_controller._set_model('mlp_100', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500), fit_params={})
if 'random_forest' not in model_names:
    ef.data_controller._set_model('random_forest', RandomForestRegressor(n_estimators=100), fit_params={})

# --- [ ADD MODELS TO PROCESS ] ---
for model_name in ['mlp_100', 'random_forest']:
    try:
        proc.add_model(model_name)
    except KeyError:
        print(f"⚠️ Model '{model_name}' not found in your model definitions.")

# --- [ TRAIN MODELS ] ---
proc.fit_models(n_epochs=10, use_torch_validation=True)

# --- [ EVALUATE ] ---
print("📊 Evaluation Results:")
for model_name in proc.models:
    try:
        print(f"  ✅ MAPE ({model_name}):", proc.mape(model_name, data_part='test'))
        print(f"  ✅ RMSE ({model_name}):", proc.rmse(model_name, data_part='test'))
    except Exception as e:
        print(f"  ❌ Evaluation failed for {model_name}: {e}")

# --- [ PLOT FORECASTS ] ---
for model_name in proc.models:
    try:
        proc.plot_forecasts(model_name, data_part='test', start=0, steps=48, alpha=0.05)
    except Exception as e:
        print(f"  ❌ Plotting failed for {model_name}: {e}")
