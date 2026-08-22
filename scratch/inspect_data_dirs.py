import os
import json

print("data/processed contents:")
if os.path.exists("data/processed"):
    for f in os.listdir("data/processed"):
        print(" ", f)

print("\ndata/indexes contents:")
if os.path.exists("data/indexes"):
    for f in os.listdir("data/indexes"):
        print(" ", f)
