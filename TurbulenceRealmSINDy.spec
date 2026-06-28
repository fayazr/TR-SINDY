# -*- mode: python ; coding: utf-8 -*-
# One-folder build: without ML
# Uses relative paths so it works on Linux, Windows, and macOS.

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=['src'],
    binaries=[],
    datas=[('config.json', '.'), ('logo.png', '.')],
    hiddenimports=['sklearn.linear_model', 'sklearn.utils._typedefs', 'scipy.ndimage', 'matplotlib.backends.backend_qtagg', 'tr_sindy_app._logging', 'tr_sindy_app._provenance', 'tr_sindy_app.settings_dialog'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6', 'PySide2', 'tensorflow', 'tensorflow_intel', 'keras', 'tensorboard', 'cupy', 'jax', 'jaxlib', 'grpc', 'google', 'google.api_core', 'google.protobuf', 'IPython', 'ipykernel', 'jupyter', 'jupyter_client', 'notebook', 'nbconvert', 'nbformat', 'qtconsole', 'jedi', 'numba', 'llvmlite', 'sympy', 'numexpr', 'tables', 'bokeh', 'plotly', 'dask', 'sphinx', 'pytest', 'wx', 'tkinter', 'test', 'torch', 'torchvision', 'torchaudio', 'onnx', 'onnxruntime', 'vtk', 'skimage', 'skimage.io', 'skimage.color', 'skimage.transform', 'skimage.measure', 'skimage.feature', 'skimage.morphology', 'skimage.filters', 'skimage.exposure', 'skimage.restoration', 'skimage.data', 'skimage.draw', 'skimage.metrics', 'langchain', 'langchain_community', 'langchain_core', 'nltk', 'astropy', 'astropy_iers_data', 'sqlalchemy', 'imageio', 'pywt', 'lxml', 'lxml.objectify', 'cryptography', 'nvidia', 'nvidia.nccl', 'mako', 'mako.codegen', 'av', 'panel', 'holoviews', 'datashader', 'param', 'pyviz_comms', 'opentelemetry', 'openai', 'anthropic', 'transformers', 'tokenizers', 'sentence_transformers', 'huggingface_hub', 'datasets', 'accelerate', 'peft', 'diffusers', 'safetensors', 'tokenizers', 'tiktoken', 'chromadb', 'pinecone', 'faiss', 'weaviate', 'pymilvus', 'qdrant_client', 'redis', 'pymongo', 'MySQLdb', 'psycopg2', 'asyncpg', 'aiomysql', 'sqlmodel', 'tortoise', 'beanie', 'motor'],
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
