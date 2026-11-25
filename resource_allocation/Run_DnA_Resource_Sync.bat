@echo off
REM ============================================================
REM  D&AA Resource Allocation -> OLT Resource Management sync
REM  Double-click to copy the Data & Advanced Analytics project
REM  allocations (Future Tech, Modern Cloud, Elevated Partnerships;
REM  excluding people listed in dna_sync_config.json) from 'D&A Resource Allocation Summaryv2.xlsx'
REM  (OLT Mgmt sheet) into 'Operations Team_ Resource Management.xlsx'
REM  (Project Resource (new) sheet).
REM  Shows a preview of the changes first and asks before writing.
REM ============================================================

setlocal

REM This .bat, the script and its config all sit together in the Resource_app
REM folder inside Team Information, so it runs for any user who has that folder
REM synced - no repo clone or fixed path.
set "PYTHON=C:\Program Files\Python312\python.exe"
set "SYNC_SCRIPT=%~dp0sync_dna_allocation.py"

echo.
echo  ======================================================================
echo   D^&AA Team Resourcing - Sync to OLT Resource Management
echo  ======================================================================
echo.
echo   WHAT THIS DOES
echo   Keeps the OLT view of Data ^& Advanced Analytics resourcing in line
echo   with the D^&AA team's own monthly allocation plan, so the D^&AA plan
echo   only has to be maintained in one place.
echo.
echo   FROM  D^&A Resource Allocation Summaryv2.xlsx  -  'OLT Mgmt' sheet
echo         (D^&AA team folder - the master D^&AA allocation plan)
echo   TO    Operations Team_ Resource Management.xlsx  -  'Project Resource (new)'
echo         (OLT Resource Management folder - Data ^& Advanced Analytics rows only)
echo.
echo   WHAT IS COPIED
echo   - Data ^& Advanced Analytics team members only, excluding anyone on
echo     the exclusion list in dna_sync_config.json (shown below)
echo   - Monthly FTE allocations from Oct-26 onwards, matched by month
echo   - Only the three E28 programs reported to the OLT:
echo       E28 Fut Tech AA        -^>  Future Tech
echo       E28 Modern Cloud       -^>  Modern Cloud  (all sub-projects added together)
echo       E28 Elevated Partners  -^>  Elevated Partnerships
echo.
echo   WHAT IS NOT CHANGED
echo   - Other teams, and other D^&AA projects in the OLT workbook
echo     (e.g. Project Pinnacle, Performance Op Model)
echo   - Past months, the leave registers and all other sheets
echo.
echo   HOW IT WORKS
echo   1. A preview of every change is listed below - nothing is written yet.
echo   2. You choose Y to apply the changes or N to cancel.
echo   3. On Y, a backup of the OLT workbook is saved locally first to
echo      %LOCALAPPDATA%\DnA_Resource_Sync\backups
echo  ======================================================================
echo.
echo  Checking D^&AA allocations against the OLT Resource Management workbook...
echo.

"%PYTHON%" "%SYNC_SCRIPT%"
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" goto :done
if not "%RC%"=="10" (
    echo.
    echo  ERROR: The comparison could not be completed. See the message above.
    goto :done
)

echo.
echo  Close 'Operations Team_ Resource Management.xlsx' in desktop Excel before continuing.
choice /c YN /n /m " Apply these changes to the OLT workbook? [Y/N] "
if errorlevel 2 (
    echo.
    echo  No changes made.
    goto :done
)

echo.
"%PYTHON%" "%SYNC_SCRIPT%" --apply
if errorlevel 1 (
    echo.
    echo  ERROR: The OLT workbook was not updated. See the message above.
    goto :done
)

echo.
echo  Done.

:done
echo.
pause
endlocal
