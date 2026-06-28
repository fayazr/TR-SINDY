; ============================================================
;  Turbulence Realm — SINDy  Inno Setup Script
;  Produces a Windows installer (Setup.exe) with:
;    - Disclaimer / no-liability license page
;    - Install location selection
;    - Start Menu + optional Desktop shortcuts
;    - Add/Remove Programs entry with uninstaller
; ============================================================

#define MyAppName        "Turbulence Realm SINDy"
#define MyAppVersion     "2.2.0"
#define MyAppPublisher   "Turbulence Realm"
#define MyAppURL         "https://www.turbulencerealm.com"
#define MyAppExeName     "TurbulenceRealmSINDy.exe"

[Setup]
; Basic metadata
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}

; Install location — Program Files on 64-bit Windows
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Output
OutputBaseFilename=TurbulenceRealmSINDy-2.2.0-Setup
OutputDir=installer
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Look & feel
SetupIconFile=logo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardStyle=modern
PrivilegesRequired=admin

; License / disclaimer page (shown before install)
LicenseFile=DISCLAIMER.txt
InfoBeforeFile=DISCLAIMER.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Bundle the entire PyInstaller output folder
Source: "build\dist\TurbulenceRealmSINDy\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
; Start Menu uninstall shortcut
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; Optional desktop shortcut
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch the app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up generated files on uninstall
Type: filesandordirs; Name: "{app}"
