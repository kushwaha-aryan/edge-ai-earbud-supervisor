#!/usr/bin/env python
"""Editable simulated-earbud response mapping for demo_yamnet.py.

Each project trigger class maps to a placeholder response name. The demo
prints every response with the label (SIMULATED): it never plays sound and
never contacts real services - edit this table to express the intended Earbud
Supervisor policy.

Edit only the response values; do NOT change the keys, because they must stay
an exact key set over the 17 project class names (demo_yamnet.py aborts if
any class is missing here).
"""

EARBUD_RESPONSES: dict[str, str] = {
    "Aircraft": "Max ANC Mode",
    "Alarm": "Alert Mode",
    "Baby_Crying": "Transparency Mode",
    "Car_Engine": "Max ANC Mode",
    "Dog_Bark": "Transparency Mode",
    "Doorbell": "Alert Mode",
    "Drilling": "Max ANC Mode",
    "Footsteps": "Max ANC Mode",
    "Glass_Breaking": "Safety Transparency Mode",
    "Gunshot": "Safety Transparency Mode",
    "Help_Shouting": "Safety Transparency Mode",
    "Jackhammer": "Max ANC Mode",
    "Knocking": "Transparency Mode",
    "Motorcycle": "Max ANC Mode",
    "Siren": "Alert Mode",
    "Train": "Max ANC Mode",
    "Vehicle_Horn": "Alert Mode",
}