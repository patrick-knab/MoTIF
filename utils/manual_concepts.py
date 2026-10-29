"""Manual (non-VLM) concept vocabularies used as the fallback when agentic concepts are disabled.

These are the hand-written concept sets shipped with the original code release. They are
stored as proper Python lists (one entry per concept) and deduplicated while preserving
the original order. The HMDB-51 list intentionally keeps the single-word form of the
original release (e.g. "look", "up", "down") so that runs match the released vocabulary.

`get_manual_concepts(dataset)` returns a copy of the list for a dataset key used by the
training scripts (``breakfast``, ``ucf101``, ``hmdb51``, ``something2``).
"""

from __future__ import annotations

from typing import Dict, List

MANUAL_CONCEPTS: Dict[str, List[str]] = {
    "breakfast": [
        "grind", "fill", "boil", "pour", "steep", "brew", "tamp", "insert", "steam", "froth",
        "stir", "sip", "add", "slice", "toast", "butter", "spread", "cut", "assemble", "grate",
        "chop", "peel", "core", "squeeze", "pit", "mash", "crack", "whisk", "beat", "fry",
        "scramble", "flip", "mix", "cook", "drizzle", "serve", "drain", "grill", "preheat", "bake",
        "warm", "wash", "rinse", "blend", "measure", "set", "open", "close", "take", "put",
        "remove", "pack", "dry", "wipe", "sit", "stand", "carry", "pick", "blow", "taste",
        "adjust", "reach", "place", "seal", "unwrap", "unscrew", "scoop", "zest", "juice", "start",
        "stop", "turn", "heat", "cool", "toss", "shake", "tap", "knock", "press", "release",
        "slide", "rotate", "fold", "unfold", "wring", "sprinkle", "arrange", "sort", "stack",
        "unstack", "hide", "reveal", "cover", "uncover", "balance", "tilt", "catch", "throw",
        "drop", "roll", "spin", "twist", "poke", "pinch", "pull", "push", "drag", "scrub", "brush",
        "comb", "shave", "zip", "button", "tie", "untie", "snap", "clap", "wave", "point", "nod",
        "gesture", "smile", "frown", "laugh", "coffee", "kettle", "water", "tea", "milk", "sugar",
        "cereal", "yogurt", "granola", "fruit", "bread", "bagel", "cheese", "tomato", "cucumber",
        "onion", "herb", "banana", "apple", "orange", "avocado", "egg", "bacon", "sausage", "ham",
        "pan", "stove", "oven", "pastry", "croissant", "strawberry", "blender", "ice", "batter",
        "syrup", "cinnamon", "honey", "jar", "plate", "cup", "spoon", "fork", "knife", "tongs",
        "lid", "package", "container", "carton", "bottle", "pantry", "fridge", "cupboard",
        "counter", "sink", "dish", "towel", "timer", "mug", "bowl", "spatula", "ladle", "grater",
        "peeler", "colander", "sieve", "cuttingboard", "tray", "ovenmitt", "scale", "thermometer",
        "stool", "chair", "table", "napkin", "freezer", "hood", "burner", "flame", "plug",
        "socket", "switch", "knob", "handle", "stirrer", "measuringcup", "measuringspoon",
        "recipe", "cookbook", "ingredient", "serving", "leftover", "waste", "soap", "sponge",
        "detergent", "faucet", "garbage", "recycle", "bin",
    ],
    "ucf101": [
        "jump", "swing", "skip", "throw", "catch", "dribble", "bounce", "kick", "pass", "hit",
        "serve", "smash", "block", "spike", "dive", "swim", "climb", "grab", "pull", "hang",
        "push", "sit", "ride", "pedal", "balance", "stop", "start", "steer", "mount", "dismount",
        "gallop", "control", "lift", "curl", "press", "squat", "deadlift", "jab", "hook",
        "uppercut", "dodge", "wrestle", "grapple", "flip", "perform", "walk", "handstand", "run",
        "sprint", "shoot", "turn", "grind", "row", "paddle", "surf", "stand", "tuck", "enter",
        "splash", "wave", "clap", "raise", "spin", "dance", "breakdance", "strike", "parry",
        "fight", "reload", "aim", "release", "bowl", "pitch", "skateboard", "snowboard", "ski",
        "trampoline", "yoga", "sword", "gun", "archery", "hockey", "basketball", "volleyball",
        "soccer", "rugby", "baseball", "cricket", "rope", "ball", "bat", "racket", "puck", "stick",
        "net", "goal", "pool", "lane", "wall", "ladder", "bar", "dumbbell", "barbell", "mat",
        "beam", "hurdle", "bicycle", "helmet", "horse", "reins", "rail", "skis", "kayak", "canoe",
        "surfboard", "gloves", "boxing", "stage", "microphone", "instrument", "music", "sheet",
        "player", "opponent", "teammate", "referee", "coach", "dancer", "athlete", "gymnast",
        "swimmer", "skater", "snowboarder", "skateboarder", "rower", "surfer", "archer", "shooter",
        "bow", "club", "frisbee", "arrow", "target", "goalpost", "jersey", "uniform", "cap", "pad",
        "netting", "court", "field", "track", "floor", "platform", "water", "sand", "snow", "ice",
        "gym", "stadium", "arena", "ring", "hoop", "basket", "scoreboard", "timer",
    ],
    "hmdb51": [
        "bow", "fight", "sword", "walk", "run", "sprint", "jog", "stand", "up", "sit", "down",
        "jump", "hop", "leap", "fall", "roll", "crouch", "bend", "stretch", "turn", "around",
        "look", "left", "right", "nod", "head", "shake", "smile", "laugh", "frown", "yawn", "talk",
        "mouth", "words", "sing", "chew", "eat", "with", "hands", "utensils", "drink", "from",
        "cup", "bottle", "sip", "blow", "kiss", "hug", "wave", "hand", "point", "reach", "grab",
        "object", "release", "throw", "catch", "toss", "ball", "kick", "hit", "punch", "block",
        "push", "pull", "lift", "carry", "drag", "drop", "climb", "crawl", "swim", "dive",
        "surface", "float", "balance", "ride", "bicycle", "pedal", "brake", "steer", "mount",
        "horse", "dismount", "gallop", "skateboard", "skate", "sled", "ski", "snowboard", "slide",
        "backward", "shoot", "basketball", "dribble", "bounce", "serve", "tennis", "swing",
        "racket", "bat", "baseball", "frisbee", "juggle", "spin", "leg", "high", "low", "flip",
        "somersault", "cartwheel", "handstand", "headstand", "touch", "face", "wash", "comb",
        "hair", "brush", "teeth", "shave", "apply", "makeup", "put", "on", "hat", "take", "off",
        "jacket", "button", "shirt", "zip", "tie", "shoelace", "untie", "open", "door", "close",
        "knock", "enter", "room", "exit", "chair", "lie", "wake", "sleep", "start", "cross",
        "finish", "line",
    ],
    "something2": [
        "push", "pull", "lift", "drop", "hold", "carry", "throw", "catch", "slide", "drag", "roll",
        "spin", "rotate", "flip", "fold", "unfold", "wrap", "unwrap", "tie", "untie", "fasten",
        "unfasten", "tighten", "loosen", "break", "cut", "slice", "chop", "tear", "peel",
        "crumple", "flatten", "bend", "stretch", "shake", "stir", "pour", "scoop", "sprinkle",
        "stack", "unstack", "assemble", "disassemble", "open", "close", "lock", "unlock", "press",
        "tap", "swipe", "scroll", "zoom in", "zoom out", "point", "touch", "wave", "clap", "knock",
        "snap", "swing", "juggle", "bounce", "balance", "topple", "insert", "remove", "fill",
        "empty", "mix", "separate", "spill", "scatter", "gather", "cover", "uncover", "hide",
        "reveal", "lean", "tilt", "climb", "crawl", "jump", "hop", "walk", "run", "sprint",
        "stumble", "fall", "get up", "sit", "stand", "kneel", "crouch", "bow", "dance",
        "spin dance", "nod", "shake head", "smile", "frown", "laugh", "cry", "shout", "whisper",
        "speak", "yawn", "sneeze", "cough", "sleep", "wake", "eat", "chew", "bite", "sip", "drink",
        "spit", "blow", "smell", "taste", "write", "draw", "erase", "paint", "type", "click",
        "drag mouse", "plug", "unplug", "connect", "disconnect", "turn on", "turn off", "start",
        "stop", "accelerate", "decelerate", "pretend to push", "pretend to pull",
        "pretend to pour", "pretend to eat", "pretend to drink", "pretend to throw",
        "pretend to catch", "pretend to type", "pretend to swipe", "pretend to scroll",
        "pretend to climb", "pretend to fall", "pretend to hug", "pretend to kiss",
        "pretend to wave", "pretend to play guitar", "pretend to drive", "pretend to steer",
        "pretend to read", "pretend to sleep", "pretend to wake", "pretend to write",
        "pretend to draw", "pretend to paint", "pretend to clean", "pretend to cook",
        "pretend to stir", "pretend to measure", "pretend to weigh", "pretend to look around",
        "pretend to search", "pretend to point", "pretend to balance", "pretend to open",
        "pretend to close", "pretend to lock", "pretend to unlock", "pretend to kick",
        "pretend to punch", "pretend to block", "pretend to dodge", "pretend to jump rope",
        "pretend to row", "pretend to paddle", "pretend to shoot arrow", "pretend to load gun",
        "pretend to fire gun", "pretend to throw ball", "pretend to dribble",
        "pretend to shoot basket", "pretend to swing bat", "pretend to serve",
        "pretend to catch fish", "pretend to steer wheel", "pretend to honk",
        "pretend to use controller", "pretend to play piano", "pretend to play drums",
        "pretend to dance", "pretend to sing", "pretend to clap", "pretend to salute",
        "pretend to bow", "pretend to shake hands", "object", "container", "box", "cup", "bowl",
        "plate", "spoon", "knife", "fork", "chopstick", "pen", "pencil", "paper", "book", "phone",
        "remote", "laptop", "keyboard", "mouse", "bag", "backpack", "toy", "ball", "fruit",
        "apple", "orange", "banana", "grape", "vegetable", "carrot", "cucumber", "tomato",
        "bottle", "can", "lid", "cap", "key", "door", "window", "wall", "floor", "table", "chair",
        "shelf", "hand", "finger", "arm", "face", "person", "other", "background", "surface",
        "inside", "outside", "top", "bottom", "left", "right", "upward", "downward", "hot", "cold",
        "wet", "dry", "clean", "dirty", "full", "broken", "fixed", "smooth", "rough", "heavy",
        "light", "fragile", "durable", "rollable", "stackable", "squeezable", "pourable",
        "spillable", "openable", "closeable", "edible", "drinkable",
    ],
}


def get_manual_concepts(dataset: str) -> List[str]:
    """Return the manual concept list for ``dataset`` (raises KeyError for unknown datasets)."""
    key = dataset.lower()
    if key not in MANUAL_CONCEPTS:
        raise KeyError(
            f"No manual concept list for dataset {dataset!r}; "
            f"expected one of {sorted(MANUAL_CONCEPTS)}"
        )
    return list(MANUAL_CONCEPTS[key])
