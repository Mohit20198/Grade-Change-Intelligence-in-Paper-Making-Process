$root = "c:\Users\nehau\OneDrive\Desktop\honeywell"
cd $root

Write-Host "1. Resetting Git to clear any accidental .env history..."
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue

Write-Host "2. Creating .gitignore and .env.example..."
@"
.env
__pycache__/
*.pyc
output/*.csv
output/*.pkl
output/*.db
*.zip
"@ | Out-File -Encoding utf8 .gitignore

@"
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
"@ | Out-File -Encoding utf8 .env.example

Write-Host "3. Deleting scratch and log files..."
$filesToDelete = @(
    "simulator\scratch_trim.py", "simulator\scratch_trim2.py", "simulator\scratch_trim3.py", "simulator\scratch_trim4.py",
    "simulator\scratch_inspect.py",
    "simulator\debug_apples.py", "simulator\debug.txt",
    "simulator\dashboard_test.txt", "simulator\streamlit_test.txt", "simulator\re_test.txt",
    "simulator\apples_output.txt", "simulator\apples_300s_output.txt",
    "simulator\lead_time_output.txt", "simulator\pattern_output.txt", "simulator\output.txt", "simulator\train_output.txt",
    "honeywell_hackathon_simulator.zip",
    "simulator\explainability_test.txt"
)
foreach ($f in $filesToDelete) {
    if (Test-Path $f) { 
        Write-Host "Deleting $f"
        Remove-Item -Force $f 
    }
}

Write-Host "Removing __pycache__ folders..."
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

Write-Host "4. Reorganizing files into folders..."
$dirs = @("feature_engineering", "correlation_discovery", "modeling", "explainability", "recommendation", "dashboard", "docs", "output", "simulator_new")
foreach ($d in $dirs) {
    if (!(Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

Move-Item "simulator\feature_engineering.py", "simulator\audit_features.py" -Destination "feature_engineering\" -ErrorAction SilentlyContinue
Move-Item "simulator\correlation_discovery.py", "simulator\generate_risk_matrix.py" -Destination "correlation_discovery\" -ErrorAction SilentlyContinue
Move-Item "simulator\train_lgbm.py", "simulator\train_lgbm_ablated.py", "simulator\lead_time_300s.py", "simulator\pattern_check.py", "simulator\compute_stab_reduction*.py", "simulator\apples_300s.py" -Destination "modeling\" -ErrorAction SilentlyContinue
Move-Item "simulator\explainability.py" -Destination "explainability\" -ErrorAction SilentlyContinue
Move-Item "simulator\recommendation_engine.py", "simulator\rationale_layer.py" -Destination "recommendation\" -ErrorAction SilentlyContinue
Move-Item "simulator\dashboard.py" -Destination "dashboard\" -ErrorAction SilentlyContinue

# Move docs
Move-Item "architecture_flowchart*" -Destination "docs\" -ErrorAction SilentlyContinue

# Move remaining simulator files
Move-Item "simulator\simulator.py", "simulator\transfer_functions.py", "simulator\event_scheduler.py", "simulator\disturbances.py", "simulator\validate_and_plot.py", "simulator\event_reporting.py" -Destination "simulator_new\" -ErrorAction SilentlyContinue

# Output folder
if (Test-Path "simulator\output") {
    Move-Item "simulator\output\*" -Destination "output\" -ErrorAction SilentlyContinue
}

# Move .env and requirements to root
Move-Item "simulator\.env" -Destination "." -ErrorAction SilentlyContinue
Move-Item "simulator\requirements.txt" -Destination "." -ErrorAction SilentlyContinue

# Cleanup old simulator
Remove-Item -Recurse -Force "simulator" -ErrorAction SilentlyContinue
Rename-Item "simulator_new" "simulator"

Write-Host "Done organizing."
