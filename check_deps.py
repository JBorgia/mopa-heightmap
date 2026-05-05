try:
    import stl
    print("numpy-stl: INSTALLED", stl.__version__ if hasattr(stl, '__version__') else "")
except ImportError as e:
    print("numpy-stl: NOT INSTALLED:", e)

try:
    import rembg
    print("rembg: INSTALLED")
except ImportError as e:
    print("rembg: NOT INSTALLED:", e)

try:
    import birefnet
    print("birefnet: INSTALLED")
except ImportError as e:
    print("birefnet: NOT INSTALLED:", e)
