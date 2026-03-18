param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$Raw,
    [Parameter(Mandatory=$true)][string]$Kg,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$MaxExamples = 0
)

$cmd = @(
    'python', '-m', 'sgens.cli', 'run-dataset',
    '--dataset', $Dataset,
    '--raw', $Raw,
    '--kg', $Kg,
    '--output-dir', $OutputDir
)

if ($MaxExamples -gt 0) {
    $cmd += @('--max-examples', "$MaxExamples")
}

Write-Host "Running:" ($cmd -join ' ')
& $cmd[0] $cmd[1..($cmd.Length-1)]
