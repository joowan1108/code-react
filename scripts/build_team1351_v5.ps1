$ErrorActionPreference = "Stop"

$imageTag = "team1351:v5"
$localAlias = "team1351_v5:latest"
$tarPath = "team1351_v5.tar"
$archivePath = "team1351_v5.tar.gz"

Write-Host "Building linux/amd64 Docker image: $imageTag"
docker build --platform linux/amd64 -t $imageTag -t $localAlias .

Write-Host "Saving image archive: $archivePath"
if (Test-Path $tarPath) {
    Remove-Item -LiteralPath $tarPath -Force
}
if (Test-Path $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

docker save -o $tarPath $imageTag
python -c "import gzip, pathlib, shutil; src=pathlib.Path(r'$tarPath'); dst=pathlib.Path(r'$archivePath'); f_in=src.open('rb'); f_out=gzip.open(dst, 'wb', compresslevel=6); shutil.copyfileobj(f_in, f_out); f_in.close(); f_out.close()"
Remove-Item -LiteralPath $tarPath -Force

Write-Host "Created $archivePath"
Write-Host "Verify with: docker load -i $archivePath"
