import cv2

print("opencv", cv2.__version__)
for name in [
    "TrackerKCF_create",
    "TrackerCSRT_create",
    "TrackerMOSSE_create",
    "legacy_TrackerMOSSE_create",
    "legacy_TrackerMedianFlow_create",
    "TrackerMedianFlow_create",
]:
    owner = cv2
    if name.startswith("legacy_"):
        owner = getattr(cv2, "legacy", None)
        actual = name.removeprefix("legacy_")
    else:
        actual = name
    print(name, bool(owner is not None and hasattr(owner, actual)))
