import sys
import os
import importlib.util
import h5py

# --- [ SETUP PATHS ] ---
base_path = "/home/stmischos/PycharmProjects/EnergyForecastingExperiments"
framework_path = f"{base_path}/EnergyForecaster/EnergyForecaster.py"
data_path = f"{base_path}/data"
project_path = f"{base_path}/benchmark_project"
h5_file = f"{project_path}/benchmark_project.h5"

# Manually create an empty .h5 file if it doesn't exist
if not os.path.exists(h5_file):
    print("🛠️ Creating minimal .h5 file with signature...")
    with h5py.File(h5_file, "w") as f:
        f.attrs['__SIGNATURE__'] = 'EnergyForecaster'  # required
        f.attrs['name'] = 'benchmark_project'           # probably used for labeling
        f.create_group("datasets")
        f.create_group("models")
        f.create_group("processes")
    print("✅ .h5 file created with required signature.")

# --- [ LOAD CLASS MANUALLY FROM .py ] ---
sys.path.append(f"{base_path}/EnergyForecaster")
spec = importlib.util.spec_from_file_location("EnergyForecaster", framework_path)
ef_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ef_module)
EnergyForecaster = ef_module.EnergyForecaster
DataController = ef_module.DataController  # extract controller directly

# --- [ ENSURE PROJECT FOLDER EXISTS ] ---
if not os.path.exists(project_path):
    os.makedirs(project_path)

# --- [ IF .H5 MISSING: DO A MANUAL CSV IMPORT FIRST ] ---
if not os.path.exists(h5_file):
    print("🟡 .h5 file missing. Importing CSVs via temporary DataController...")
    dummy_ef = type("DummyEF", (), {})()  # fake EF context
    dummy_data = DataController(dummy_ef, project_path)
    dummy_data.import_csv(f"{data_path}/weather_2_template.csv", h5_name='weather_2')
    dummy_data.import_csv(f"{data_path}/consumption_2_template.csv", h5_name='consumption_2')
    print("✅ .h5 file created via import.")
else:
    print("✅ .h5 file already exists. Skipping import.")

# --- [ NOW CREATE FULL FORECASTER INSTANCE ] ---
ef = EnergyForecaster(project_path)

# --- [ CONTINUE AS NORMAL ] ---
dt_weather = ef.data_controller.get_dataset('weather_2')
dt_consumption = ef.data_controller.get_dataset('consumption_2')

dt_weather.to_timestamp('datetime', '%Y-%m-%d %H:%M:%S', 'Europe/Athens', assign='inplace')
dt_consumption.to_timestamp('datetime', '%Y-%m-%d %H:%M:%S', 'Europe/Athens', assign='inplace')

dt_weather.make_scale('datetime')
dt_consumption.make_scale('datetime')
dt_weather.attach_scale('temperature', 'datetime')
dt_consumption.attach_scale('consumption', 'datetime')
dt_consumption.make_target('consumption')

ef.data_controller.update_dataset('weather_2')
ef.data_controller.update_dataset('consumption_2')

proc = ef.process_controller.create_process('benchmark_proc')
proc.insert_data('weather_2', ['temperature'])
proc.insert_data('consumption_2', ['consumption'])

proc.add_model('mlp_100')
proc.add_model('random_forest')

proc.fit_models(n_epochs=10, use_torch_validation=True)

print('MAPE (mlp_100):', proc.mape('mlp_100'))
print('RMSE (random_forest):', proc.rmse('random_forest'))

proc.plot_forecasts('mlp_100', data_part='test', start=0, steps=48, alpha=0.05)
proc.plot_forecasts('random_forest', data_part='test', start=0, steps=48, alpha=0.05)
