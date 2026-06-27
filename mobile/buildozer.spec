[app]

# (str) Title of your application
title = Turbulence Realm SINDy

# (str) Package name
package.name = trsindy

# (str) Package domain (needed for android/ios package)
package.domain = com.turbulencerealm

# (str) Source code where the main.py live
source.dir = .

# (str) Source extension
source.ext = py

# (list) Source files (let buildozer auto-discover)
source.include_exts = py,png,jpg,kv,atlas,json

# (list) List of inclusions using pattern matching
source.include_patterns = tr_sindy_mobile/*,main.py

# (str) Application versioning
version = 2.2.0

# (list) Application requirements
# Core: kivy, numpy, opencv
requirements = python3,kivy,numpy,opencv

# (str) Custom application folder
# (leave empty to use default)
source.folder = .

# (str) Presplash of the application
presplash.filename = %(source.dir)s/presplash.png

# (str) Icon of the application
icon.filename = %(source.dir)s/icon.png

# (str) Supported orientation
orientation = landscape

# (list) List of service to declare
services =

# (str) Android logcat filter
android.logcat_filters = *:S python:D

# (bool) Enable AndroidX
android.gradle_dependencies = androidx.appcompat:appcompat:1.3.1

# (bool) Skip android permissions
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# (int) Android API to use
android.api = 31

# (int) Minimum API required
android.minapi = 24

# (int) Android NDK version
android.ndk = 23b

# (bool) Use private storage
android.private_storage = True

# (str) Android NDK path (leave empty for auto)
# android.ndk_path =

# (str) Android SDK path (leave empty for auto)
# android.sdk_path =

# (str) Python-for-Android branch
p4a.branch = master

# (str) Python-for-Android url
# p4a.url =

# (str) Python version
python.version = 3.10

# ---
# Build configuration
# ---

# (str) Build directory
build_dir = .buildozer

# (str) Bin directory
bin_dir = bin

# (bool) Copy lib instead of making a symlink
# android.copy_libs = 1

# ---
# iOS (not yet supported for this app)
# ---
[ios]

# (str) Name of the iOS app
app_name = Turbulence Realm SINDy

# ---
# Buildozer
# ---
[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug)
log_level = 1

# (int) Warning level
warn_on_root = 1
