# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import namedtuple
import inspect

"""
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
        facemodel_param_value=(0.829955, 0.259177, 0.580118),
        facemodel_param_value_other=(0.784082, 0.376347, 0.604226),
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
        facemodel_param_value=(0.784082, 0.376347, 0.604226),
        facemodel_param_value_other=(0.695231, 0.467287, 0.612526),
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
        facemodel_param_value=(0.695231, 0.467287, 0.612526),
        facemodel_param_value_other=(0.580811, 0.520703, 0.605280),
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
        facemodel_param_value=(0.580811, 0.520703, 0.605280),
        facemodel_param_value_other=(0.465637, 0.526611, 0.586355),
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
        facemodel_param_value=(0.465637, 0.526611, 0.586355),
        facemodel_param_value_other=(0.360264, 0.473636, 0.562969),
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
        facemodel_param_value=(0.360264, 0.473636, 0.562969),
        facemodel_param_value_other=(0.465637, 0.526611, 0.586355),
    )

    black_hair_config = ControllableAttributeConfig(
        driven_attribute="hair_color_Black_Hair",
        ignored_attributes=[
            "hair_color_Blond_Hair",
            "hair_color_Brown_Hair",
            "hair_color_Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(0.188568, 0.250595, 0.511412),
        facemodel_param_value_other=(0.573767, 0.319474, 0.548478),
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
            0.573767,
            0.319474,
            0.548478,
        ),  # [0.58878942 0.56902791 0.42737119]
        facemodel_param_value_other=(0.188568, 0.250595, 0.511412),
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
            0.234597,
            0.558910,
            0.527588,
        ),  # [0.31239411 0.55426495 0.52791491]
        facemodel_param_value_other=(0.188568, 0.250595, 0.511412),
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
            0.611758,
            0.241252,
            0.537658,
        ),  # [0.53267241 0.53701314 0.25010427]
        facemodel_param_value_other=(0.188568, 0.250595, 0.511412),
    )

    # black_color_config = ControllableAttributeConfig(
    #     driven_attribute="Dark_Skin",
    #     ignored_attributes=["typ"],
    #     facemodel_param_name="skin_color",
    #     facemodel_param_value=(
    #         0.549,
    #         0.597,
    #         0.4963,
    #     ),  # Dark_Skin [0.54996947 0.59787681 0.49632458]
    #     facemodel_param_value_other=(
    #         0.707,
    #         0.600,
    #         0.39,
    #     ),
    # )

    # light_color_config = ControllableAttributeConfig(
    #     driven_attribute="Light_Skin",
    #     ignored_attributes=["Dark_Skin"],
    #     facemodel_param_name="skin_color",
    #     facemodel_param_value=(
    #         0.707,
    #         0.600,
    #         0.39,
    #     ),  # Light_Skin [0.70755964 0.6006187  0.39943934]
    #     facemodel_param_value_other=(
    #         0.549,
    #         0.597,
    #         0.4963,
    #     ),
    # )
    # black_hair_config = ControllableAttributeConfig(
    #     driven_attribute = "Black_Hair",
    #     ignored_attributes = ["Blond_Hair", "Brown_Hair", "Gray_Hair"],
    #     facemodel_param_name = "head_hair_color",
    #     facemodel_param_value = (0, 1, 0),
    #     facemodel_param_value_other = (0, 0.1, 0.1)
    # )

    # blond_hair_config = ControllableAttributeConfig(
    #     driven_attribute = "Blond_Hair",
    #     ignored_attributes = ["Black_Hair", "Brown_Hair", "Gray_Hair"],
    #     facemodel_param_name = "head_hair_color",
    #     facemodel_param_value = (0, 0.1, 0.1),
    #     facemodel_param_value_other = (0, 1, 0)
    # )

    # brown_hair_config = ControllableAttributeConfig(
    #     driven_attribute = "Brown_Hair",
    #     ignored_attributes = ["Blond_Hair", "Black_Hair", "Gray_Hair"],
    #     facemodel_param_name = "head_hair_color",
    #     facemodel_param_value = (0, 0.6, 0.5),
    #     facemodel_param_value_other = (0, 0.1, 0.1)
    # )

    # gray_hair_config = ControllableAttributeConfig(
    #     driven_attribute = "Gray_Hair",
    #     ignored_attributes = ["Blond_Hair", "Brown_Hair", "Black_Hair"],
    #     facemodel_param_name = "head_hair_color",
    #     facemodel_param_value = (0.7, 0.7, 0),
    #     facemodel_param_value_other = (0.0, 0.7, 0)
    # )

    # mouth_open_config = ControllableAttributeConfig(
    #     driven_attribute = "Mouth_Slightly_Open",
    #     ignored_attributes = ["Narrow_Eyes", "Smiling"],
    #     facemodel_param_name = "blendshape_values",
    #     facemodel_param_value = {"jaw_opening": 0.2},
    #     facemodel_param_value_other = {"jaw_opening": -0.05}
    # )

    # smile_config = ControllableAttributeConfig(
    #     driven_attribute = "Smiling",
    #     ignored_attributes = ["Narrow_Eyes", "Mouth_Slightly_Open"],
    #     facemodel_param_name = "blendshape_values",
    #     facemodel_param_value = {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0},
    #     facemodel_param_value_other = {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0}
    # )

    # squint_config = ControllableAttributeConfig(
    #     driven_attribute = "Narrow_Eyes",
    #     ignored_attributes = ["Smiling", "Mouth_Slightly_Open"],
    #     facemodel_param_name = "blendshape_values",
    #     facemodel_param_value = {"EyeBLinkLeft": 0.7, "EyeBLinkRight": 0.7},
    #     facemodel_param_value_other = {"EyeWideLeft": 1.0, "EyeWideRight": 1.0}
    # )

    # mustache_config = ControllableAttributeConfig(
    #     driven_attribute = "Mustache",
    #     ignored_attributes = ["No_Beard", "Goatee", "Sideburns"],
    #     facemodel_param_name = "beard_style_embedding",
    #     # "beard_Wavy_f"
    #     facemodel_param_value = [
    #         0.8493434358437133,
    #         3.087059026013613,
    #         0.46986106722598997,
    #         -1.3821969829871341,
    #         -0.33103870587106415,
    #         -0.03649891754263812,
    #         0.049692808518749985,
    #         0.10727920600451613,
    #         -0.32365312847867017
    #     ],
    #     # "beard_none"
    #     facemodel_param_value_other = [
    #         -1.1549744366277825,
    #         -0.15234213575276162,
    #         -0.3302730721199086,
    #         -0.47053537289207514,
    #         -0.158377484760156,
    #         0.3357074575072504,
    #         -0.44934623275285585,
    #         0.013085621430078971,
    #         -0.0021044358910661896
    #     ]
    # )
