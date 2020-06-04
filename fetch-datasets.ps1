$UrlBase = "http://files.grouplens.org/datasets/movielens/"
$DataDir = 'data'
$ErrorActionPreference = 'Stop'

function Fetch-DataSet
{
    Param(
        [Parameter(Position=0, Mandatory=$true)]
        [string]
        $TestFile,
        [Parameter(Position=1, Mandatory=$true)]
        [string]
        $ZipName
    )

    $tft = Join-Path $DataDir $TestFile
    if (-Not (Test-Path $tft)) {
        Write-Host "$TestFile not found"
        $zpath = Join-Path $DataDir $ZipName
        $zurl = "$UrlBase$ZipName"
        Write-Host "Downloading $zurl"
        Invoke-WebRequest -OutFile $zpath $zurl
        Write-Host "Extracting $zpath"
        Expand-Archive $zpath -DestinationPath $DataDir
    } else {
        Write-Host "$TestFile already exists"
    }
}

if (-Not (Test-Path $DataDir)) {
    New-Item -ItemType Directory $DataDir
}
foreach ($arg in $args) {
    switch ($arg) {
        "ml-100k" {
            Fetch-DataSet "ml-100k/u.data" "ml-100k.zip"
        }
        "ml-1m" {
            Fetch-DataSet "ml-1m/ratings.dat" "ml-1m.zip"
        }
        "ml-10m" {
            Fetch-DataSet "ml-10M100K/ratings.dat" "ml-10m.zip"
        }
        "ml-20m" {
            Fetch-DataSet "ml-20m/ratings.csv" "ml-20m.zip"
        }
        default {
            Write-Host "unknown data set $arg"
        }
    }
}
