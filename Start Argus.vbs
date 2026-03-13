' Argus Launcher - Runs completely hidden
' Double-click this file to start Argus in the background
' The eye icon will appear in your system tray

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get the directory this script is in
scriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
' scripts are in \scripts, so project root is scriptDir
projectDir = scriptDir

' Build the path to the python script
' NOTE: We assume argus_tray.py is in scripts/ folder
pythonScript = projectDir & "/scripts/argus_tray.py"

' Debug: Check if script exists
If Not FSO.FileExists(pythonScript) Then
    MsgBox "Could not find script at: " & pythonScript, 16, "Argus Error"
    WScript.Quit
End If

' Try to find pythonw.exe (windowless python)
' First check local venv
venvPython = projectDir & "\venv\Scripts\pythonw.exe"
globalPython = "pythonw"

If FSO.FileExists(venvPython) Then
    ' Use venv python if it exists
    cmd = """" & venvPython & """ """ & pythonScript & """"
Else
    ' Fallback to system pythonw
    cmd = "pythonw """ & pythonScript & """"
End If

' Set working directory to project root so imports work
WshShell.CurrentDirectory = projectDir

' Run hidden (0 = hidden, False = don't wait)
' If this fails, pythonw might not be in PATH
On Error Resume Next
WshShell.Run cmd, 0, False

If Err.Number <> 0 Then
    MsgBox "Failed to launch Argus. Make sure Python is in your PATH.", 16, "Argus Error"
End If
