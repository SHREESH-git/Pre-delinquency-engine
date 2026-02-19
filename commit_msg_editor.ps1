param([string]$path)
$content = Get-Content $path
$content = $content -replace 'Initial commit', 'Implemented Alerting Mechanism'
$content | Set-Content $path
