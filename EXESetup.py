from cx_Freeze import setup, Executable

setup(
    name="Palm Oil Fruit Classifier",
    version="0.1",
    description="Classify palm oil fruits using a trained model",
    executables=[Executable("GUI_Config.py", base="Win32GUI")]
)
