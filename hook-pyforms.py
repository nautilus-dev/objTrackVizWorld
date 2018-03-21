hiddenimports = [
    "pyforms",
    "pyforms.gui"
    "pyforms.gui.Controls"
    "pyforms.gui.Controls.*"
]


from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files('pyforms', subdir="gui/Controls")