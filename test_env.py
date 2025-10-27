# 进入 Python 交互并逐条运行（或把下面写成 test_env.py 然后 python test_env.py）
import importlib, sys
modules = ["torch","bitsandbytes","faiss","sentence_transformers"]
for m in modules:
    try:
        mod = importlib.import_module(m)
        print(f"{m} imported. version:", getattr(mod, "__version__", "no __version__"))
    except Exception as e:
        print(f"{m} IMPORT ERROR:", repr(e))
print("sys.executable:", sys.executable)

