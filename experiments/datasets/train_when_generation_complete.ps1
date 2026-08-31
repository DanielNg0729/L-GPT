param(
    [int]$ExpectedRows = 7922,
    [int]$PollSeconds = 30
)

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$training = Join-Path $root 'robustness\v2\catalogue_synonym_training.jsonl'
$python = Join-Path $root '.venv-v2\Scripts\python.exe'
$stdout = Join-Path $root 'robustness\v2\results\attribute_encoder_training.log'
$stderr = Join-Path $root 'robustness\v2\results\attribute_encoder_training.err.log'

while (-not (Test-Path -LiteralPath $training) -or (Get-Content -LiteralPath $training | Measure-Object -Line).Lines -lt $ExpectedRows) {
    Start-Sleep -Seconds $PollSeconds
}

& $python -m robustness.v2.train_attribute_encoder *>> $stdout
if ($LASTEXITCODE -ne 0) {
    "Fine-tuning failed with exit code $LASTEXITCODE" | Add-Content -LiteralPath $stderr
    exit $LASTEXITCODE
}
