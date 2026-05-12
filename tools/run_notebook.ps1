# Execute the main notebook end-to-end with all required env vars set.
# Outputs are written back into the .ipynb in place.
# Uses the Python 3.11 venv at .venv311 because PySpark 3.5.1 has worker-pickle
# incompatibilities with Python 3.13.

$ErrorActionPreference = 'Continue'

$VenvPy = "c:/AB_Projects2/.venv311/Scripts/python.exe"

# --- Environment for PySpark on Windows ---
$env:JAVA_HOME = "C:\Users\HarishChandra\AppData\Local\Programs\Microsoft\jdk-17.0.10.7-hotspot"
$env:HADOOP_HOME = "C:\AB_Projects2\.hadoop"
$env:Path = "$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;" + $env:Path
$env:PYSPARK_PYTHON = $VenvPy
$env:PYSPARK_DRIVER_PYTHON = $VenvPy
$env:PYTHONIOENCODING = "utf-8"

Write-Output "JAVA_HOME    = $env:JAVA_HOME"
Write-Output "HADOOP_HOME  = $env:HADOOP_HOME"
Write-Output "PYSPARK_PY   = $env:PYSPARK_PYTHON"
Write-Output "VENV PYTHON  = $VenvPy"
Write-Output ""
java -version 2>&1 | Select-Object -First 1
& $VenvPy --version
Write-Output ""
Write-Output "Starting notebook execution..."
Write-Output "Working dir: $(Get-Location)"
Write-Output ""

# nbconvert in-place execution, generous per-cell timeout for CV-tuning cells.
& $VenvPy -m jupyter nbconvert `
    --to notebook `
    --execute `
    --inplace `
    --ExecutePreprocessor.timeout=1500 `
    --ExecutePreprocessor.kernel_name=venv311 `
    notebooks/Bank_Distributed_ML_Project.ipynb 2>&1

$exit = $LASTEXITCODE
Write-Output ""
Write-Output "nbconvert exited with code $exit"
exit $exit
