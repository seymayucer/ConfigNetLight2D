# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import namedtuple
import inspect

"""for celeba clean augmented dataset
# V_mean	S_mean	Cr_mean
hair_color			
Black Hair	0.159687	0.350653	0.517305
Blond Hair	0.587065	0.447917	0.571579
Brown Hair	0.302940	0.546138	0.553557
Gray Hair	0.525238	0.247917	0.537637
"""

"""
	V_mean_S	S_mean_S	Cr_mean_S
skin_group			
Skin Tone 1	0.865812	0.293642	0.589465
Skin Tone 2	0.808378	0.371466	0.606398
Skin Tone 3	0.742096	0.425965	0.612668
Skin Tone 4	0.661814	0.475354	0.613644
Skin Tone 5	0.558040	0.511155	0.604469
Skin Tone 6	0.419688	0.527130	0.582609

"""
ControllableAttributeConfig = namedtuple(
    "ControllableAttributeConfig",
    "driven_attribute ignored_attributes facemodel_param_name facemodel_param_value facemodel_param_value_other",
)


class ControllabilityMetricConfigs:
    @staticmethod
    def all_configs():
        all_attributes = inspect.getmembers(
            ControllabilityMetricConfigs, lambda a: not inspect.isroutine(a)
        )
        configs = [
            x
            for x in all_attributes
            if not (x[0].startswith("__") and x[0].endswith("__"))
        ]

        return configs

    type1_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type1",
        ignored_attributes=[
            "skintype_Type2",
            "skintype_Type3",
            "skintype_Type4",
            "skintype_Type5",
            "skintype_Type6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.865812,	0.293642,	0.589465),
        facemodel_param_value_other=(0.808378,	0.371466,	0.606398),
    )

    type2_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type2",
        ignored_attributes=[
            "skintype_Type1",
            "skintype_Type3",
            "skintype_Type4",
            "skintype_Type5",
            "skintype_Type6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.808378,	0.371466,	0.606398),
        facemodel_param_value_other=(0.742096,	0.425965,	0.612668),
    )

    type3_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type3",
        ignored_attributes=[
            "skintype_Type1",
            "skintype_Type2",
            "skintype_Type4",
            "skintype_Type5",
            "skintype_Type6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.742096,	0.425965,	0.612668),
        facemodel_param_value_other=(0.661814,	0.475354,	0.613644),
    )

    type4_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type4",
        ignored_attributes=[
            "skintype_Type1",
            "skintype_Type2",
            "skintype_Type3",
            "skintype_Type5",
            "skintype_Type6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.661814,	0.475354,	0.613644),
        facemodel_param_value_other=(0.558040,	0.511155,	0.604469),
    )
    type5_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type5",
        ignored_attributes=[
            "skintype_Type1",
            "skintype_Type2",
            "skintype_Type3",
            "skintype_Type4",
            "skintype_Type6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.558040,	0.511155,	0.604469),
        facemodel_param_value_other=(0.419688,	0.527130,	0.582609),
    )

    type6_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_Type6",
        ignored_attributes=[
            "skintype_Type1",
            "skintype_Type2",
            "skintype_Type3",
            "skintype_Type4",
            "skintype_Type5",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(0.419688,	0.527130,	0.582609),
        facemodel_param_value_other=(0.558040,	0.511155,	0.604469),
    )

    black_hair_config = ControllableAttributeConfig(
        driven_attribute="hair_color_Black_Hair",
        ignored_attributes=[
            "hair_color_Blond_Hair",
            "hair_color_Brown_Hair",
            "hair_color_Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(0.159687, 0.350653, 0.517305),
        facemodel_param_value_other=(0.302940,	0.546138,	0.553557),
    )

    blond_hair_config = ControllableAttributeConfig(
        driven_attribute="hair_color_Blond_Hair",
        ignored_attributes=[
            "hair_color_Black_Hair",
            "hair_color_Brown_Hair",
            "hair_color_Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
           0.587065,	0.447917,	0.571579
        ),  # [0.58878942 0.56902791 0.42737119]
        facemodel_param_value_other=(0.525238,	0.247917,	0.537637),
    )

    brown_hair_config = ControllableAttributeConfig(
        driven_attribute="hair_color_Brown_Hair",
        ignored_attributes=[
            "hair_color_Blond_Hair",
            "hair_color_Black_Hair",
            "hair_color_Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
          0.302940,	0.546138,	0.553557
        ),  # [0.31239411 0.55426495 0.52791491]
        facemodel_param_value_other=(0.159687,	0.350653,	0.517305),
    )

    gray_hair_config = ControllableAttributeConfig(
        driven_attribute="hair_color_Gray_Hair",
        ignored_attributes=[
            "hair_color_Blond_Hair",
            "hair_color_Brown_Hair",
            "hair_color_Black_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
          0.525238,	0.247917,	0.537637
        ),  # [0.53267241 0.53701314 0.25010427]
        facemodel_param_value_other=( 0.587065,	0.447917,	0.571579),
    )

   