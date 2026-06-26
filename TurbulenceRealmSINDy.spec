# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Turbulence Realm - SINDy
# One-folder build with aggressive excludes so the bloated Anaconda
# scientific stack (torch/tf/keras/grpc/jupyter/...) is not pulled in.

block_cipher = None

EXCLUDES = [
    'PyQt5', 'PySide6', 'PySide2',
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'tensorflow_intel', 'keras', 'tensorboard',
    'cupy', 'jax', 'jaxlib',
    'grpc', 'google', 'google.api_core', 'google.protobuf',
    'IPython', 'ipykernel', 'jupyter', 'jupyter_client', 'notebook',
    'nbconvert', 'nbformat', 'qtconsole', 'jedi', 'numba', 'llvmlite',
    'sympy', 'numexpr', 'tables', 'h5py', 'bokeh', 'plotly', 'dask',
    'sphinx', 'pytest', 'wx', 'tkinter', 'test',
]

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'sklearn.linear_model',
        'sklearn.utils._typedefs',
        'scipy.ndimage',
        'matplotlib.backends.backend_qtagg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TurbulenceRealmSINDy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TurbulenceRealmSINDy',
)
