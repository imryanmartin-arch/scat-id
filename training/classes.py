# Jackson County, Oregon scat classes -> iNaturalist taxon IDs.
# Photos are pulled from anywhere (scat looks the same everywhere), but only
# species that actually live around Medford are classes.
CLASSES = [
    # key,          display name,                taxon ids
    ("deer",       "Black-tailed deer",          [42220]),   # Odocoileus hemionus (incl. black-tail)
    ("elk",        "Roosevelt elk",              [204114]),
    ("bear",       "Black bear",                 [41638]),
    ("cougar",     "Cougar",                     [42007]),
    ("coyote",     "Coyote",                     [42051]),
    ("bobcat",     "Bobcat",                     [41976]),
    ("grayfox",    "Gray fox",                   [42076]),
    ("raccoon",    "Raccoon",                    [41663]),
    ("ringtail",   "Ringtail",                   [41676]),
    ("opossum",    "Virginia opossum",           [42652]),
    ("skunk",      "Skunk",                      [41871]),
    ("porcupine",  "Porcupine",                  [44026]),
    ("rabbit",     "Rabbit / hare",              [43095]),
    ("otter",      "River otter",                [41777]),
    ("wolf",       "Gray wolf",                  [42048]),
    ("turkey",     "Wild turkey",                [906]),
    ("grouse",     "Grouse / quail",             [890, 1240, 1278]),
    ("dog",        "Domestic dog",               [47144]),
    ("cattle",     "Cattle",                     [74113]),
    ("horse",      "Horse / mule",               [209233]),
]

# "Not scat": things a hunter might point the camera at by mistake.
# (taxon_id, evidence term_value or None, max observations)
NEGATIVES = [
    (40151, 26, 500),    # mammal tracks
    (47170, None, 450),  # fungi (mushrooms look like scat)
    (136329, None, 300), # Pinales (cones, needles)
    (311249, None, 150), # mosses
    (54743, None, 150),  # lichens
]

MAX_OBS_PER_CLASS = 1200
LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
