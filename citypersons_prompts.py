"""
citypersons_prompts.py
-----------------------
Pedestrian-specific hierarchical prompt pools for the HFR-VLM framework.
These replace the generic surveillance prompts with CityPersons-aware
descriptions that:
  - Capture pedestrian behaviour, posture, and context
  - Avoid generating identity-revealing text (faces, clothing details, etc.)
  - Preserve safety-relevant semantic information (crossing, group, cyclist)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CityPersonsPromptConfig:
    """
    Three-level prompt hierarchy tuned for urban pedestrian scenes.

    Layer A: Scene-level (broad urban context)
    Layer B: Pedestrian group / behaviour level
    Layer C: Individual detection level (fine-grained, no PII)
    """

    layer_a: List[str] = field(default_factory=lambda: [
        "Describe the overall urban street scene without identifying individuals.",
        "What is the general pedestrian activity level in this city scene?",
        "Summarise the traffic and pedestrian situation shown.",
        "Describe the public space environment and any crowd conditions.",
        "What type of urban area is depicted — commercial, residential, transit hub?",
        "Describe the road layout, sidewalks, and pedestrian infrastructure visible.",
        "What time of day and weather conditions appear in this street scene?",
        "Describe the overall safety context of this urban environment.",
    ])

    layer_b: List[str] = field(default_factory=lambda: [
        "How many people are present and what is their general movement pattern?",
        "Are pedestrians crossing the road, waiting, or walking along the pavement?",
        "Describe any groups of people without identifying individuals.",
        "What pedestrian safety behaviours are observable (e.g. using crosswalk)?",
        "Describe any cyclists or riders present in the scene.",
        "Are there people in unusual postures or situations that may require attention?",
        "Describe the pedestrian density and flow direction.",
        "What interactions between pedestrians and vehicles are visible?",
        "Are there any accessibility-relevant observations (e.g. wheelchair users)?",
        "Describe pedestrian proximity to vehicles or road hazards.",
    ])

    layer_c: List[str] = field(default_factory=lambda: [
        "Describe the approximate height and build of detected persons without faces.",
        "What clothing type (jacket, coat, etc.) is worn, without colour identification?",
        "Describe the posture of the nearest pedestrian (walking, standing, running).",
        "Is the pedestrian carrying any objects? Describe without identifying them.",
        "Describe the pedestrian's direction of travel relative to the road.",
        "Is the pedestrian on the pavement, in a cycle lane, or on the road?",
        "Describe any safety-relevant equipment visible (helmet, high-vis vest).",
        "Is the pedestrian using a mobile device or distracted in any way?",
        "Describe the distance of the detected person from the nearest vehicle.",
        "What is the estimated age group (child, adult, elderly) without further detail?",
        "Describe if the pedestrian appears to be in motion or stationary.",
        "Is there anything unusual about the pedestrian's behaviour or position?",
    ])


# PII terms specific to pedestrian surveillance
PEDESTRIAN_PII_TERMS = [
    "face", "facial", "eyes", "hair colour", "hair color", "skin",
    "name", "identity", "recognise", "recognize", "identify",
    "race", "ethnicity", "gender", "sex",
    "licence plate", "license plate",
    "phone number", "tattoo", "scar",
    "nationality", "religion",
]
