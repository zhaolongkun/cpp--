$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
  param([string]$Message)
  Write-Host "[Step] $Message" -ForegroundColor Cyan
}

function Fail {
  param([string]$Message)
  throw "[Error] $Message"
}

function Require-Path {
  param(
    [string]$Path,
    [string]$Label
  )
  if (-not (Test-Path -LiteralPath $Path)) {
    Fail "$Label not found: $Path"
  }
}

function Import-VcVarsEnvironment {
  param([string]$BatchPath)

  Require-Path -Path $BatchPath -Label "vcvars64.bat"
  Write-Step "Importing MSVC environment from vcvars64.bat"

  $envDump = & cmd.exe /d /c "call `"$BatchPath`" >nul && set"
  if ($LASTEXITCODE -ne 0) {
    Fail "vcvars64.bat failed with exit code $LASTEXITCODE"
  }

  foreach ($line in $envDump) {
    if ($line -match "^([^=]+)=(.*)$") {
      $name = $matches[1]
      $value = $matches[2]
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}

function Invoke-Tool {
  param(
    [string]$FilePath,
    [string[]]$Arguments = @(),
    [string]$WorkingDirectory = ""
  )

  $oldLocation = Get-Location
  try {
    if ($WorkingDirectory) {
      Set-Location -LiteralPath $WorkingDirectory
    }

    Write-Host ("[Run] {0} {1}" -f $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
      Fail "Command failed with exit code ${exitCode}: $FilePath $($Arguments -join ' ')"
    }
  }
  finally {
    if ($WorkingDirectory) {
      Set-Location -LiteralPath $oldLocation
    }
  }
}

function Sync-DllToExeDir {
  param(
    [string]$SourcePath,
    [string]$DestinationDir
  )

  $targetPath = Join-Path $DestinationDir (Split-Path $SourcePath -Leaf)
  $sourceItem = Get-Item -LiteralPath $SourcePath

  $copyRequired = $true
  if (Test-Path -LiteralPath $targetPath) {
    $targetItem = Get-Item -LiteralPath $targetPath
    if ($sourceItem.Length -eq $targetItem.Length -and $sourceItem.LastWriteTimeUtc -eq $targetItem.LastWriteTimeUtc) {
      $copyRequired = $false
    }
  }

  if (-not $copyRequired) {
    return
  }

  try {
    Copy-Item -LiteralPath $SourcePath -Destination $DestinationDir -Force
  }
  catch {
    if (Test-Path -LiteralPath $targetPath) {
      $targetItem = Get-Item -LiteralPath $targetPath
      if ($sourceItem.Length -eq $targetItem.Length) {
        Write-Warning "DLL is locked by another process, but an identical file already exists: $targetPath"
        return
      }
    }
    Fail "Failed to copy DLL to executable directory: $SourcePath"
  }
}

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$workspaceRoot = Split-Path (Split-Path $projectRoot -Parent) -Parent
$asciiRoot = Join-Path $workspaceRoot "cpp_control_ascii"
$buildDir = Join-Path $asciiRoot "build\nmake-msvc-onnx-release"
$exePath = Join-Path $buildDir "tracker.exe"

$vcvarsPath = "D:\VisualStudio2019\VC\Auxiliary\Build\vcvars64.bat"
$opencvDir = "C:\Users\Administrator\miniconda3\envs\py310\Library\cmake\x64\vc17\lib"
$ortRoot = Join-Path $workspaceRoot "third_party\onnxruntime_nuget\Microsoft.ML.OnnxRuntime.1.24.1"
$ortIncludeDir = Join-Path $ortRoot "build\native\include"
$ortLibPath = Join-Path $ortRoot "runtimes\win-x64\native\onnxruntime.lib"
$ortDllPath = Join-Path $ortRoot "runtimes\win-x64\native\onnxruntime.dll"
$ortSharedDllPath = Join-Path $ortRoot "runtimes\win-x64\native\onnxruntime_providers_shared.dll"

Require-Path -Path $projectRoot -Label "Project root"
Require-Path -Path $asciiRoot -Label "ASCII project root"
Require-Path -Path (Join-Path $asciiRoot "CMakeLists.txt") -Label "CMakeLists.txt"
Require-Path -Path $opencvDir -Label "OpenCV_DIR"
Require-Path -Path $ortIncludeDir -Label "ONNX Runtime include dir"
Require-Path -Path $ortLibPath -Label "ONNX Runtime import library"
Require-Path -Path $ortDllPath -Label "onnxruntime.dll"
Require-Path -Path $ortSharedDllPath -Label "onnxruntime_providers_shared.dll"

Import-VcVarsEnvironment -BatchPath $vcvarsPath

$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$nmake = (Get-Command nmake.exe -ErrorAction Stop).Source

Write-Step "Configuring CMake project"
Invoke-Tool -FilePath $cmake -Arguments @(
  "-S", $asciiRoot,
  "-B", $buildDir,
  "-G", "NMake Makefiles",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DUSE_SYSTEM_OPENCV=ON",
  "-DOpenCV_DIR=$opencvDir",
  "-DUSE_ONNXRUNTIME=ON",
  "-DONNXRUNTIME_INCLUDE_DIR=$ortIncludeDir",
  "-DONNXRUNTIME_LIBRARY=$ortLibPath"
)

Write-Step "Building tracker with nmake"
Invoke-Tool -FilePath $nmake -Arguments @("/nologo") -WorkingDirectory $buildDir

Require-Path -Path $exePath -Label "tracker.exe"

Write-Step "Copying ONNX Runtime DLLs to executable directory"
Sync-DllToExeDir -SourcePath $ortDllPath -DestinationDir $buildDir
Sync-DllToExeDir -SourcePath $ortSharedDllPath -DestinationDir $buildDir

Require-Path -Path (Join-Path $buildDir "onnxruntime.dll") -Label "Copied onnxruntime.dll"
Require-Path -Path (Join-Path $buildDir "onnxruntime_providers_shared.dll") -Label "Copied onnxruntime_providers_shared.dll"

Write-Host ""
Write-Host "[Done] Build completed successfully." -ForegroundColor Green
Write-Host "[Info] tracker.exe: $exePath"
Write-Host "[Info] build dir:   $buildDir"
