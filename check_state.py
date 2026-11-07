import os

base = r"D:\Alex"
for entry in sorted(os.listdir(base)):
    p = os.path.join(base, entry)
    if os.path.isdir(p):
        sub = sorted(os.listdir(p))
        print(f"DIR {entry}: {sub}")
    else:
        print(f"FILE {entry}: {os.path.getsize(p)} bytes")
