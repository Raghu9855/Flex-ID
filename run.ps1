# ============================================================
# FLEX-ID — Master Launcher Script (PowerShell)
# Run from: C:\Users\HP\Flex-ID\
#
# Usage:
#   .\run.ps1 api              → Start FastAPI backend server
#   .\run.ps1 test             → Run automated test suite (pytest)
#   .\run.ps1 server_fedavg    → Start FedAvg server (4 clients, 10 rounds)
#   .\run.ps1 server_fedprox   → Start FedProx server
#   .\run.ps1 server_krum      → Start Krum server
#   .\run.ps1 server_median    → Start Median server
#   .\run.ps1 client 0         → Start honest client 0
#   .\run.ps1 attack 0 backdoor → Start malicious client 0
#   .\run.ps1 evaluate         → Evaluate + plot results
#   .\run.ps1 explain          → Run SHAP explainability
#   .\run.ps1 partition 4      → Re-partition for N clients
#   .\run.ps1 install          → Install all dependencies
# ============================================================

param(
    [Parameter(Position=0)] [string]$Command = "help",
    [Parameter(Position=1)] [string]$Arg1 = "",
    [Parameter(Position=2)] [string]$Arg2 = "flip"
)

$VENV_PYTHON = "$PSScriptRoot\venv\Scripts\python.exe"
if (Test-Path $VENV_PYTHON) {
    $PYTHON = $VENV_PYTHON
    $PY_ARGS = @()
} else {
    $PYTHON = "py"
    $PY_ARGS = @("-3.13")
}
$ML_DIR = "$PSScriptRoot\ml"

switch ($Command) {

    "api" {
        Write-Host "[FLEX-ID] Starting FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
        & $PYTHON $PY_ARGS -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    }

    "test" {
        Write-Host "[FLEX-ID] Executing automated pytest test suite..." -ForegroundColor Cyan
        & $PYTHON $PY_ARGS -m pytest -v
    }

    "benchmark" {
        Write-Host "[FLEX-ID] Running API performance and latency load benchmark..." -ForegroundColor Magenta
        & $PYTHON $PY_ARGS "$PSScriptRoot\scripts\load_test.py" --concurrency 5 --requests 50
    }

    "install" {
        Write-Host "[FLEX-ID] Installing all packages..." -ForegroundColor Cyan
        & $PYTHON $PY_ARGS -m pip install -r "$PSScriptRoot\requirements.txt"
        Write-Host "[FLEX-ID] Done." -ForegroundColor Green
    }

    "partition" {
        $N = if ($Arg1) { $Arg1 } else { "4" }
        Write-Host "[FLEX-ID] Creating $N client partitions..." -ForegroundColor Cyan
        & $PYTHON $PY_ARGS "$ML_DIR\2_create_partitions.py" --num_clients $N
    }

    "server_fedavg" {
        $N = if ($Arg1) { $Arg1 } else { "4" }
        Write-Host "[FLEX-ID] Starting FedAvg server ($N clients, 10 rounds)..." -ForegroundColor Green
        & $PYTHON $PY_ARGS "$ML_DIR\4_server.py" --strategy fedavg --rounds 10 --num_clients $N
    }

    "server_fedprox" {
        $N = if ($Arg1) { $Arg1 } else { "4" }
        Write-Host "[FLEX-ID] Starting FedProx server ($N clients, 10 rounds)..." -ForegroundColor Green
        & $PYTHON $PY_ARGS "$ML_DIR\4_server.py" --strategy fedprox --rounds 10 --proximal_mu 0.1 --num_clients $N
    }

    "server_krum" {
        $N = if ($Arg1) { $Arg1 } else { "4" }
        Write-Host "[FLEX-ID] Starting Krum (Multi-Krum) server ($N clients)..." -ForegroundColor Green
        & $PYTHON $PY_ARGS "$ML_DIR\4_server.py" --aggregation multikrum --rounds 10 --num_clients $N --attack
    }

    "server_median" {
        $N = if ($Arg1) { $Arg1 } else { "4" }
        Write-Host "[FLEX-ID] Starting Coordinate Median server ($N clients)..." -ForegroundColor Green
        & $PYTHON $PY_ARGS "$ML_DIR\4_server.py" --aggregation median --rounds 10 --num_clients $N --attack
    }

    "client" {
        $CID = if ($Arg1) { $Arg1 } else { "0" }
        Write-Host "[FLEX-ID] Starting honest client $CID ..." -ForegroundColor Yellow
        & $PYTHON $PY_ARGS "$ML_DIR\client.py" --cid $CID
    }

    "attack" {
        $CID    = if ($Arg1) { $Arg1 } else { "0" }
        $AType  = $Arg2
        Write-Host "[FLEX-ID] Starting MALICIOUS client $CID (attack=$AType) ..." -ForegroundColor Red
        & $PYTHON $PY_ARGS "$ML_DIR\client_attack.py" --cid $CID --attack_type $AType --scale 0.3
    }

    "evaluate" {
        Write-Host "[FLEX-ID] Evaluating models..." -ForegroundColor Cyan
        $fa  = "$PSScriptRoot\results\fedavgeachround\round-10-weights.pkl"
        $fp  = "$PSScriptRoot\results\fedproxeachround\round-10-weights.pkl"
        & $PYTHON $PY_ARGS "$ML_DIR\compare_results.py" --fedavg $fa --fedprox $fp --mode no_attack
        & $PYTHON $PY_ARGS "$ML_DIR\plot_history.py"
    }

    "explain" {
        Write-Host "[FLEX-ID] Running SHAP explainability..." -ForegroundColor Cyan
        & $PYTHON $PY_ARGS "$ML_DIR\explain_model.py" --round 10 --num_clients 4
    }

    default {
        Write-Host @"

FLEX-ID Launcher
----------------
Usage:  .\run.ps1 <command> [args]

Backend & Test Commands:
  api                       Start FastAPI backend server on http://localhost:8000
  test                      Run automated pytest suite (19 tests)
  benchmark                 Run concurrent load & latency benchmark

Federated Learning Commands:
  install                   Install all Python dependencies
  partition [N]             Create partitions for N clients (default: 4)
  server_fedavg  [N]        Start FedAvg server
  server_fedprox [N]        Start FedProx server
  server_krum    [N]        Start Multi-Krum server (attack mode)
  server_median  [N]        Start Coordinate Median server (attack mode)
  client         <cid>      Start honest client with ID <cid>
  attack         <cid> <type>  Start malicious client (types: flip/noise/backdoor/byzantine/adaptive)
  evaluate                  Compare FedAvg vs FedProx + plot curves
  explain                   Run federated SHAP explanation

"@ -ForegroundColor Cyan
    }
}
