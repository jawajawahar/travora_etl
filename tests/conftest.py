import os

# Tests never call the public road-routing server: legs are estimated offline.
os.environ.setdefault("TRAVORA_ROAD_ROUTING", "0")
