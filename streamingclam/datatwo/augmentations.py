import albumentationsxl as A

augmentations = A.Compose(
    [
        A.Flip(p=0.5),
        A.HueSaturationValue(
           hue_shift_limit=20,
           sat_shift_limit=20,
           val_shift_limit=20,
           p=0.2),
        A.Rotate(p=0.5),
    ],
)

#augmentations = None