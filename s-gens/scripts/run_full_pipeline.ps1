param(
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$Raw,
    [Parameter(Mandatory=$true)][string]$Kg,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Preset = 'paper',
    [int]$MaxExamples = 0,
    [string]$ExperimentName = '',
    [string]$EvalBackend = 'lexical',
    [string]$EvalModel = '',
    [switch]$TrainGnn,
    [int]$GnnEpochs = 5,
    [switch]$TrainRetriever,
    [string]$RetrieverModelName = 'distilbert-base-uncased',
    [int]$RetrieverEpochs = 1,
    [switch]$UseTrainedRetrieverForEval,
    [switch]$UseOllamaQueryGeneration,
    [string]$OllamaModel = 'llama3.1:8b',
    [string]$OllamaUrl = 'http://127.0.0.1:11434',
    [switch]$RunRagAnswer,
    [string]$RagQuery = '',
    [int]$RagTopK = 5,
    [string]$RagOutput = ''
)

$runCmd = @(
    'python', '-m', 'sgens.cli', 'run-dataset',
    '--dataset', $Dataset,
    '--raw', $Raw,
    '--kg', $Kg,
    '--output-dir', $OutputDir,
    '--preset', $Preset
)
if ($MaxExamples -gt 0) { $runCmd += @('--max-examples', "$MaxExamples") }
if ($ExperimentName -ne '') { $runCmd += @('--experiment-name', $ExperimentName) }
if ($UseOllamaQueryGeneration) {
    $runCmd += @('--query-generator', 'ollama', '--ollama-model', $OllamaModel, '--ollama-url', $OllamaUrl)
}

Write-Host "Running S-Gens pipeline:" ($runCmd -join ' ')
& $runCmd[0] $runCmd[1..($runCmd.Length-1)]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$runRoot = if ($ExperimentName -ne '') { Join-Path $OutputDir $ExperimentName } else { Join-Path $OutputDir ("{0}_{1}" -f $Dataset, $Preset) }
$documents = Join-Path $runRoot 'normalized\documents.json'
$original = Join-Path $runRoot 'normalized\original.json'
$triplets = Join-Path $runRoot 'sgens_run\artifacts\triplets.json'
$gnnOut = Join-Path $runRoot 'checkpoints\gnn'
$retrieverOut = Join-Path $runRoot 'checkpoints\retriever'

if ($TrainGnn) {
    $gnnCmd = @(
        'python', '-m', 'sgens.cli', 'train-gnn',
        '--triplets', $triplets,
        '--kg', $Kg,
        '--output-dir', $gnnOut,
        '--epochs', "$GnnEpochs"
    )
    Write-Host "Training Siamese GNN:" ($gnnCmd -join ' ')
    & $gnnCmd[0] $gnnCmd[1..($gnnCmd.Length-1)]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($TrainRetriever) {
    $trainCmd = @(
        'python', '-m', 'sgens.cli', 'train-retriever',
        '--triplets', $triplets,
        '--documents', $documents,
        '--output-dir', $retrieverOut,
        '--model-name', $RetrieverModelName,
        '--epochs', "$RetrieverEpochs"
    )
    Write-Host "Training dense retriever:" ($trainCmd -join ' ')
    & $trainCmd[0] $trainCmd[1..($trainCmd.Length-1)]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$effectiveEvalBackend = $EvalBackend
$effectiveEvalModel = $EvalModel
if ($UseTrainedRetrieverForEval) {
    $effectiveEvalBackend = 'dense'
    $effectiveEvalModel = $retrieverOut
}

$evalCmd = @(
    'python', '-m', 'sgens.cli', 'evaluate-retriever',
    '--backend', $effectiveEvalBackend,
    '--documents', $documents,
    '--original', $original,
    '--save-to-run-root', $runRoot
)
if ($effectiveEvalModel -ne '') { $evalCmd += @('--model', $effectiveEvalModel) }

Write-Host "Evaluating retriever:" ($evalCmd -join ' ')
& $evalCmd[0] $evalCmd[1..($evalCmd.Length-1)]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($RunRagAnswer) {
    if ($RagQuery -eq '') {
        Write-Error 'RagQuery is required when RunRagAnswer is set.'
        exit 1
    }
    $ragCmd = @(
        'python', '-m', 'sgens.cli', 'answer',
        '--query', $RagQuery,
        '--documents', $documents,
        '--retriever-backend', $effectiveEvalBackend,
        '--top-k', "$RagTopK",
        '--ollama-model', $OllamaModel,
        '--ollama-url', $OllamaUrl
    )
    if ($effectiveEvalModel -ne '') { $ragCmd += @('--retriever-model', $effectiveEvalModel) }
    if ($RagOutput -ne '') { $ragCmd += @('--output', $RagOutput) }
    Write-Host "Running RAG answer:" ($ragCmd -join ' ')
    & $ragCmd[0] $ragCmd[1..($ragCmd.Length-1)]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
