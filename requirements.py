import os

packages = {
    'PIL': 'Pillow',
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'requests': 'requests',
    'extcolors': 'extcolors',
    'sklearn': 'scikit-learn',
    'colorthief': 'colorthief',
    'colour': 'colour-science'
}

for key, val in packages.items():
    try:
        __import__(key)
    except ImportError:
        os.system("pip install " + val)
