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
	
            V_mean_S	V_std_S	S_mean_S	S_std_S	Cr_mean_S	Cr_std_S
skin_group						
Skin Tone 1	0.927383,0.044418,0.252556,0.057728,0.585349,3.877831
Skin Tone 2	0.884012,0.055293,0.335849,0.061203,0.609150,3.570142
Skin Tone 3	0.823905,0.068941,0.388567,0.069273,0.617843,3.494607
Skin Tone 4	0.753302,0.084607,0.444590,0.083087,0.623231,3.751974
Skin Tone 5	0.650748,0.097353,0.481329,0.095233,0.616137,3.918589
Skin Tone 6	0.508124,0.102235,0.520104,0.103364,0.598684,4.028233

"""

# V_mean_S	S_mean_S	Cr_mean_S
# majority_skintype
# 0	0.937763,	0.220777,	0.575370
# 1	0.903074,	0.309669,	0.602593
# 2	0.847762,	0.367820,	0.614938
# 3	0.778475,	0.423459,	0.620422
# 4	0.680472,	0.484950,	0.619607
# 5	0.523411,	0.527622,	0.601773


ControllableAttributeConfig = namedtuple(
    "ControllableAttributeConfig",
    "driven_attribute ignored_attributes facemodel_param_name facemodel_param_value facemodel_param_value_other",
)
"""
17 august 23

V_mean_S	V_std_S	S_mean_S	S_std_S	Cr_mean_S	Cr_std_S
variable						
skintype_1	0.855453,0.135949,0.276197,0.098960,0.581058,4.832301
skintype_2	0.808546,0.144814,0.361028,0.093633,0.602349,4.782313
skintype_3	0.749417,0.150773,0.416414,0.094081,0.611127,5.133609
skintype_4	0.675483,0.155918,0.469878,0.101796,0.614196,5.794926
skintype_5	0.575255,0.156508,0.506131,0.115588,0.606380,6.370054
skintype_6	0.458280,0.146087,0.526468,0.125722,0.589138,6.129124


(26021, 69) (26021, 125) (26021, 125) (26021, 128) (26021, 128)
V_mean	V_std	S_mean	S_std	Cr_mean	Cr_std
variable						
Black_Hair	0.137353,	0.119011,	0.347207,	0.219011,	0.514795,	4.365853
Blond_Hair	0.539634,	0.168390,	0.451182,	0.153096,	0.566042,	5.504892
Brown_Hair	0.257701,	0.145443,	0.528483,	0.182997,	0.544831,	5.643669
Gray_Hair	0.487158,	0.141263,	0.242061,	0.118065,	0.532360,	5.200809
"""


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
        driven_attribute="skintype_1",
        ignored_attributes=[
            "skintype_2",
            "skintype_3",
            "skintype_4",
            "skintype_5",
            "skintype_6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.855453,
            0.135949,
            0.276197,
            0.098960,
            146.156,
            4.832301,
        ),
        facemodel_param_value_other=(
            0.458280,
            0.146087,
            0.526468,
            0.125722,
            147.967,
            6.129124,
        ),
    )

    type2_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_2",
        ignored_attributes=[
            "skintype_1",
            "skintype_3",
            "skintype_4",
            "skintype_5",
            "skintype_6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.808546,
            0.144814,
            0.361028,
            0.093633,
            150.92,
            4.782313,
        ),
        facemodel_param_value_other=(
            0.458280,
            0.146087,
            0.526468,
            0.125722,
            147.967,
            6.129124,
        ),
    )

    type3_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_3",
        ignored_attributes=[
            "skintype_1",
            "skintype_2",
            "skintype_4",
            "skintype_5",
            "skintype_6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.749417,
            0.150773,
            0.416414,
            0.094081,
            152.89,
            5.133609,
        ),
        facemodel_param_value_other=(
            0.458280,
            0.146087,
            0.526468,
            0.125722,
            147.967,
            6.129124,
        ),
    )

    type4_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_4",
        ignored_attributes=[
            "skintype_1",
            "skintype_2",
            "skintype_3",
            "skintype_5",
            "skintype_6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.675483,
            0.155918,
            0.469878,
            0.101796,
            153.57,
            5.794926,
        ),
        facemodel_param_value_other=(
            0.855453,
            0.135949,
            0.276197,
            0.098960,
            146.156,
            4.832301,
        ),
    )
    type5_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_5",
        ignored_attributes=[
            "skintype_1",
            "skintype_2",
            "skintype_3",
            "skintype_4",
            "skintype_6",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.575255,
            0.156508,
            0.506131,
            0.115588,
            151.82,
            6.370054,
        ),
        facemodel_param_value_other=(
            0.855453,
            0.135949,
            0.276197,
            0.098960,
            146.156,
            4.832301,
        ),
    )

    type6_skin_color_config = ControllableAttributeConfig(
        driven_attribute="skintype_6",
        ignored_attributes=[
            "skintype_1",
            "skintype_2",
            "skintype_3",
            "skintype_4",
            "skintype_5",
        ],
        facemodel_param_name="skin_color",
        facemodel_param_value=(
            0.458280,
            0.146087,
            0.526468,
            0.125722,
            147.96,
            6.129124,
        ),
        facemodel_param_value_other=(
            0.855453,
            0.135949,
            0.276197,
            0.098960,
            146.156,
            4.832301,
        ),
    )

    black_hair_config = ControllableAttributeConfig(
        driven_attribute="Black_Hair",
        ignored_attributes=[
            "Blond_Hair",
            "Brown_Hair",
            "Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
            0.137353,
            0.119011,
            0.347207,
            0.219011,
            131.31,
            4.365853,
        ),
        facemodel_param_value_other=(
            0.539634,
            0.168390,
            0.451182,
            0.153096,
            142.793421,
            5.504892,
        ),
    )

    blond_hair_config = ControllableAttributeConfig(
        driven_attribute="Blond_Hair",
        ignored_attributes=[
            "Black_Hair",
            "Brown_Hair",
            "Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
            0.539634,
            0.168390,
            0.451182,
            0.153096,
            142.793421,
            5.504892,
        ),
        facemodel_param_value_other=(
            0.257701,
            0.145443,
            0.528483,
            0.182997,
            138.04,
            5.643669,
        ),
    )

    brown_hair_config = ControllableAttributeConfig(
        driven_attribute="Brown_Hair",
        ignored_attributes=[
            "Blond_Hair",
            "Black_Hair",
            "Gray_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
            0.257701,
            0.145443,
            0.528483,
            0.182997,
            138.04,
            5.643669,
        ),  # [0.31239411 0.55426495 0.52791491]
        facemodel_param_value_other=(
            0.539634,
            0.168390,
            0.451182,
            0.153096,
            142.793421,
            5.504892,
        ),
    )

    gray_hair_config = ControllableAttributeConfig(
        driven_attribute="Gray_Hair",
        ignored_attributes=[
            "Blond_Hair",
            "Brown_Hair",
            "Black_Hair",
        ],
        facemodel_param_name="hair_color",
        facemodel_param_value=(
            0.487158,
            0.141263,
            0.242061,
            0.118065,
            135.24,
            5.200809,
        ),  # [0.53267241 0.53701314 0.25010427]
        facemodel_param_value_other=(
            0.539634,
            0.168390,
            0.451182,
            0.153096,
            142.793421,
            5.504892,
        ),
    )

    right_narrow_eyes_config = ControllableAttributeConfig(
        driven_attribute="Narrow_Eyes",
        ignored_attributes=["Big_Lips"],
        facemodel_param_name="right_eye_features",
        facemodel_param_value=(
            0.5920399,
            0.1100694,
            0.27706745,
            0.16528295,
            0.2302398,
            0.29087615,
            0.53507143,
            0.5502912,
            0.5048181,
            0.28353253,
            0.601653,
            0.02122175,
            0.28111592,
            0.18367511,
            0.3125438,
            0.67406243,
            0.67270565,
            0.6510665,
            0.63333124,
            0.4390739,
            0.19914114,
            0.26569512,
            0.69648015,
            0.64586294,
            0.65464705,
            0.619775,
            0.52423334,
            0.2797138,
            0.43687978,
            0.31864163,
            0.14185835,
            0.62731713,
            0.15920764,
            0.655575,
            0.25433207,
            0.23032057,
            0.616634,
            0.5202292,
            0.49023756,
            0.21859004,
            0.13285655,
            0.21777453,
            0.5983069,
            0.6930065,
            0.26593342,
            0.17437351,
            0.32700115,
            0.4776197,
            0.69010043,
            0.20746045,
            0.14089182,
            0.49043566,
            0.6009436,
            0.6737395,
            0.61239505,
            0.5424416,
            0.46397933,
            0.6640819,
            0.6483609,
            0.63772297,
            0.66322434,
            0.22528301,
            0.6596935,
            0.19922833,
            0.6832109,
            0.2938239,
            0.1768483,
            0.5080349,
            0.48598552,
            0.6668984,
            0.48517525,
            0.16083176,
            0.1206691,
            0.19729221,
            0.22999942,
            0.23911123,
            0.3033363,
            0.6412591,
            0.51992315,
            0.1936624,
            0.27583203,
            0.4442367,
            0.19800013,
            0.17466605,
            0.65977806,
            0.687429,
            0.15389183,
            0.5441554,
            0.23876846,
            0.28129843,
            0.13007246,
            0.6553308,
            0.17992727,
            0.617102,
            0.2017771,
            0.6287206,
            0.43616676,
            0.1502318,
            0.56114584,
            0.18334134,
            0.6560821,
            0.5262655,
            0.4716009,
            0.6372574,
            0.65006906,
            0.22567616,
            0.21167319,
            0.6500235,
            0.13690004,
            0.2674418,
            0.3101184,
            0.64540243,
            0.31938183,
            0.26377574,
            0.29798928,
            0.6186401,
            0.19791254,
            0.5743667,
            0.46174857,
            0.20269348,
            0.12791027,
            0.1631034,
            0.23370014,
            0.1900116,
            0.24277441,
        ),
        facemodel_param_value_other=(
            0.44992903,
            0.34380105,
            0.34479856,
            0.6383789,
            0.7152164,
            0.56802,
            0.32089984,
            0.3384638,
            0.41032597,
            0.64244324,
            0.5502117,
            0.22664568,
            0.6068487,
            0.66359305,
            0.80257905,
            0.17682172,
            0.17540154,
            0.15823445,
            0.14431423,
            0.3010866,
            0.6826978,
            0.7546781,
            0.4411156,
            0.15381314,
            0.16057779,
            0.14850476,
            0.46201122,
            0.77736187,
            0.27976102,
            0.62077296,
            0.5991657,
            0.14019488,
            0.6312698,
            0.5808567,
            0.74850994,
            0.71545124,
            0.16874385,
            0.32018533,
            0.438863,
            0.70334923,
            0.5897274,
            0.702272,
            0.32429776,
            0.19868349,
            0.34077615,
            0.6506718,
            0.42862794,
            0.0966944,
            0.38510832,
            0.69154817,
            0.5840336,
            0.3424748,
            0.32550132,
            0.17648402,
            0.33746943,
            0.31675312,
            0.11497784,
            0.3693007,
            0.15588363,
            0.66348773,
            0.36668304,
            0.37203404,
            0.16405345,
            0.683258,
            0.3775406,
            0.788658,
            0.65302277,
            0.4591964,
            0.3853865,
            0.4496406,
            0.40357888,
            0.6330213,
            0.5746348,
            0.6809533,
            0.7146417,
            0.7249171,
            0.79671913,
            0.30006555,
            0.11295375,
            0.6742925,
            0.77230346,
            0.3148057,
            0.68128175,
            0.6510314,
            0.16488415,
            0.19296059,
            0.6226719,
            0.32034925,
            0.72441447,
            0.694342,
            0.58616966,
            0.16165122,
            0.65837014,
            0.33742768,
            0.6365518,
            0.14128739,
            0.3290326,
            0.6155659,
            0.11349515,
            0.6631925,
            0.1615186,
            0.13309637,
            0.27594018,
            0.35143182,
            0.18594393,
            0.7105308,
            0.6960729,
            0.15731229,
            0.5847442,
            0.56988966,
            0.8047134,
            0.35568064,
            0.33106613,
            0.7606094,
            0.7925447,
            0.5943048,
            0.68159515,
            0.36524174,
            0.42782387,
            0.6452269,
            0.5801284,
            0.6368157,
            0.71906096,
            0.6491052,
            0.7314076,
        ),
    )

    left_narrow_eyes_config = ControllableAttributeConfig(
        driven_attribute="Narrow_Eyes",
        ignored_attributes=["Big_Lips"],
        facemodel_param_name="left_eye_features",
        facemodel_param_value=(
            0.616474,
            0.6730417,
            0.69864476,
            0.21235958,
            0.12658472,
            0.14786553,
            0.67508256,
            0.26271033,
            0.68307155,
            0.15391287,
            0.540135,
            0.6894515,
            0.6794379,
            0.15312977,
            0.5438359,
            0.63909876,
            0.23395842,
            0.15729508,
            0.14262332,
            0.6450155,
            0.7021486,
            0.7197217,
            0.6247824,
            0.1555154,
            0.1639877,
            0.70622545,
            0.61990243,
            0.63273734,
            0.18114308,
            0.7234984,
            0.54165536,
            0.14732133,
            0.6599288,
            0.55015457,
            0.19753335,
            0.6552454,
            0.14846493,
            0.13331394,
            0.44148624,
            0.66076964,
            0.71733785,
            0.5925731,
            0.6611807,
            0.29287118,
            0.7615028,
            0.14152023,
            0.5886979,
            0.6812749,
            0.20941071,
            0.25476253,
            0.20678928,
            0.336411,
            0.17245716,
            0.64108205,
            0.55454165,
            0.7179138,
            0.4832637,
            0.83212477,
            0.6546958,
            0.5738704,
            0.7148787,
            0.21664898,
            0.6961204,
            0.60186124,
            0.64507926,
            0.6761728,
            0.20831086,
            0.14693299,
            0.64953274,
            0.68948644,
            0.5906758,
            0.7478071,
            0.59488416,
            0.7157399,
            0.57743174,
            0.27137834,
            0.15366772,
            0.5874202,
            0.5509029,
            0.24500631,
            0.1978606,
            0.21954805,
            0.35362408,
            0.15709624,
            0.34503284,
            0.1803968,
            0.28450492,
            0.7162732,
            0.18188003,
            0.14608382,
            0.6188411,
            0.6590737,
            0.14438862,
            0.6765788,
            0.1701287,
            0.5672182,
            0.22854061,
            0.5523653,
            0.70782053,
            0.49360305,
            0.3148442,
            0.78102183,
            0.1906735,
            0.46992514,
            0.6178116,
            0.67428637,
            0.11837798,
            0.3387076,
            0.649414,
            0.6581675,
            0.17014708,
            0.15714367,
            0.16902596,
            0.16064024,
            0.39162487,
            0.19235173,
            0.68805456,
            0.19058742,
            0.19678558,
            0.592239,
            0.6808985,
            0.80659074,
            0.70912224,
            0.58599424,
            0.47408137,
        ),
        facemodel_param_value_other=(
            0.22412409,
            0.1466028,
            0.16264792,
            0.75500786,
            0.6363911,
            0.6683331,
            0.14765023,
            0.80706036,
            0.7048061,
            0.6829289,
            0.7273199,
            0.15668496,
            0.70631266,
            0.67953336,
            0.23335828,
            0.24760395,
            0.7816799,
            0.6840121,
            0.66108173,
            0.6162465,
            0.16474351,
            0.17777036,
            0.6113923,
            0.68386954,
            0.69739926,
            0.16796133,
            0.19916314,
            0.12426879,
            0.7217811,
            0.23573811,
            0.0768061,
            0.6725003,
            0.13849702,
            0.5479025,
            0.7414791,
            0.26402637,
            0.6742484,
            0.6470274,
            0.4931433,
            0.6203534,
            0.7196289,
            0.11696428,
            0.60949564,
            0.76851845,
            0.21512562,
            0.65642315,
            0.13303897,
            0.57856953,
            0.751294,
            0.7991523,
            0.7480161,
            0.7564687,
            0.7136364,
            0.66341674,
            0.55626005,
            0.17756344,
            0.06785537,
            0.6786581,
            0.5634805,
            0.12626913,
            0.17484383,
            0.7600454,
            0.16068278,
            0.5856808,
            0.5618133,
            0.7280774,
            0.7494882,
            0.6699855,
            0.14114828,
            0.5833234,
            0.5801927,
            0.20348285,
            0.2078453,
            0.72940016,
            0.10223808,
            0.8013021,
            0.68291956,
            0.15385005,
            0.55231595,
            0.7916049,
            0.32101354,
            0.7539129,
            0.8093278,
            0.6880878,
            0.63379127,
            0.72137743,
            0.8256041,
            0.17612542,
            0.72382164,
            0.67074394,
            0.1179371,
            0.62699413,
            0.6680485,
            0.72014916,
            0.7092286,
            0.547084,
            0.7771146,
            0.12841505,
            0.5884774,
            0.7468139,
            0.81025136,
            0.6334326,
            0.73260176,
            0.72930074,
            0.38450256,
            0.58177817,
            0.61618674,
            0.6804541,
            0.69934446,
            0.7463842,
            0.7095206,
            0.6852982,
            0.70813996,
            0.6908561,
            0.6863084,
            0.7348489,
            0.15606923,
            0.7318795,
            0.73864615,
            0.6751178,
            0.6880585,
            0.65510607,
            0.5942296,
            0.1345718,
            0.5138738,
        ),
    )

    big_nose_config = ControllableAttributeConfig(
        driven_attribute="Big_Nose",
        ignored_attributes=["Big_Lips"],
        facemodel_param_name="nose_features",
        facemodel_param_value=(
            0.23237285,
            0.30395752,
            0.6608439,
            0.7044698,
            0.18260251,
            0.605401,
            0.32594174,
            0.65364677,
            0.16970526,
            0.67403483,
            0.18775125,
            0.6703528,
            0.32190207,
            0.7317104,
            0.3023044,
            0.35627636,
            0.20534422,
            0.8018754,
            0.30575523,
            0.2967586,
            0.29326382,
            0.19224046,
            0.43180606,
            0.6112324,
            0.31152076,
            0.7734608,
            0.7219393,
            0.65371853,
            0.29779014,
            0.6635271,
            0.7614787,
            0.74517393,
            0.2951968,
            0.3110177,
            0.2937335,
            0.29782045,
            0.31362307,
            0.33190247,
            0.1701534,
            0.19950113,
            0.34765548,
            0.2604984,
            0.7397538,
            0.62025064,
            0.63066155,
            0.7325782,
            0.7404801,
            0.3341361,
            0.61596555,
            0.23235965,
            0.3048803,
            0.65324074,
            0.31101668,
            0.6674319,
            0.52941024,
            0.33387387,
            0.30427548,
            0.22400606,
            0.6313588,
            0.16337666,
            0.2980746,
            0.49076506,
            0.23004413,
            0.32296202,
            0.583574,
            0.5932692,
            0.31303903,
            0.358043,
            0.352435,
            0.43813765,
            0.20257746,
            0.6958395,
            0.50763303,
            0.30262747,
            0.76343596,
            0.1768459,
            0.5544738,
            0.3045491,
            0.6114174,
            0.6260381,
            0.34170341,
            0.19300312,
            0.7013855,
            0.77221304,
            0.3015212,
            0.6598251,
            0.68622047,
            0.16274534,
            0.35272038,
            0.23771809,
            0.3198429,
            0.7150248,
            0.17123713,
            0.6180136,
            0.3303612,
            0.6112246,
            0.6167652,
            0.625574,
            0.2987719,
            0.6287632,
            0.27565843,
            0.59240514,
            0.6350189,
            0.7939183,
            0.23025927,
            0.29359344,
            0.6941494,
            0.7056573,
            0.68140894,
            0.59288996,
            0.17637107,
            0.19896972,
            0.33759728,
            0.16392975,
            0.29380977,
            0.57263476,
            0.58017117,
            0.30454254,
            0.29581454,
            0.7688191,
            0.76096576,
            0.77946,
            0.71393013,
            0.61298823,
            0.6261098,
        ),
        facemodel_param_value_other=(
            0.73715603,
            0.8009799,
            0.7013167,
            0.69688094,
            0.24049199,
            0.12835005,
            0.8204139,
            0.7085527,
            0.6129231,
            0.68599784,
            0.2514405,
            0.75929004,
            0.81717587,
            0.69258744,
            0.8000604,
            0.78577745,
            0.7064305,
            0.83435047,
            0.802477,
            0.7964061,
            0.7943305,
            0.31215134,
            0.64010644,
            0.13150208,
            0.80750895,
            0.68482596,
            0.70335835,
            0.69968504,
            0.79697764,
            0.69334155,
            0.6960594,
            0.7069834,
            0.79489666,
            0.8069964,
            0.793945,
            0.7972506,
            0.8096143,
            0.82651913,
            0.60362047,
            0.66672146,
            0.80222285,
            0.7683746,
            0.79066503,
            0.20477885,
            0.67989254,
            0.7126907,
            0.7183246,
            0.8286739,
            0.13526267,
            0.7401584,
            0.8019111,
            0.6848332,
            0.8070849,
            0.65280205,
            0.10873906,
            0.8283227,
            0.80156356,
            0.7294974,
            0.6657656,
            0.59140205,
            0.7971297,
            0.08385456,
            0.7346087,
            0.8181995,
            0.1296977,
            0.807878,
            0.8090968,
            0.84642947,
            0.8355836,
            0.81113416,
            0.33707127,
            0.6778706,
            0.08800623,
            0.80036515,
            0.67595696,
            0.6345496,
            0.10582937,
            0.8017869,
            0.67842025,
            0.7019574,
            0.8346241,
            0.6801136,
            0.7260396,
            0.6903593,
            0.79972094,
            0.6777675,
            0.7290524,
            0.5899282,
            0.8428755,
            0.74470484,
            0.8153063,
            0.67771554,
            0.24297959,
            0.6685962,
            0.82415897,
            0.68734103,
            0.6728344,
            0.14134471,
            0.7972079,
            0.1441608,
            0.7823455,
            0.6541111,
            0.6762333,
            0.66937166,
            0.73797977,
            0.79413736,
            0.72136074,
            0.7985018,
            0.7386049,
            0.1227795,
            0.6327563,
            0.25719866,
            0.8116115,
            0.593823,
            0.7942567,
            0.11416161,
            0.11565337,
            0.801631,
            0.79545695,
            0.7363466,
            0.72231275,
            0.866378,
            0.6845937,
            0.65608436,
            0.729038,
        ),
    )

    big_lips_config = ControllableAttributeConfig(
        driven_attribute="Big_Lips",
        ignored_attributes=["Big_Nose"],
        facemodel_param_name="mouth_features",
        facemodel_param_value=(
            0.6305179,
            0.9038065,
            0.9027913,
            0.35117996,
            0.3176988,
            0.3242847,
            0.30487177,
            0.30441082,
            0.6272389,
            0.31535783,
            0.33149245,
            0.76861715,
            0.05098867,
            0.66400516,
            0.9136412,
            0.70478195,
            0.6336113,
            0.7446265,
            0.83651114,
            0.8147285,
            0.63171715,
            0.6354931,
            0.30442646,
            0.7773606,
            0.67218715,
            0.6796592,
            0.3055468,
            0.7545446,
            0.63494575,
            0.27246305,
            0.89185834,
            0.01932677,
            0.88002104,
            0.3199599,
            0.3051343,
            0.3272169,
            0.89951396,
            0.65740174,
            0.3250353,
            0.3333114,
            0.3071043,
            0.678123,
            0.9257821,
            0.28139886,
            0.6739668,
            0.598005,
            0.30090597,
            0.3043916,
            0.6658857,
            0.3191915,
            0.30174103,
            0.9038796,
            0.34710902,
            0.8063714,
            0.81909007,
            0.8205964,
            0.88916886,
            0.80222946,
            0.30834755,
            0.8937012,
            0.3403558,
            0.32958513,
            0.31069633,
            0.31699952,
            0.3114522,
            0.31879622,
            0.3041044,
            0.3251156,
            0.7429942,
            0.89800316,
            0.32441208,
            0.05782639,
            0.30666524,
            0.30258316,
            0.3144379,
            0.6543671,
            0.63496524,
            0.80857676,
            0.6423253,
            0.704595,
            0.7971133,
            0.9235428,
            0.31646514,
            0.75336295,
            0.7879664,
            0.89336526,
            0.3288484,
            0.31631956,
            0.3029868,
            0.6968194,
            0.32783118,
            0.8197326,
            0.8870954,
            0.3165228,
            0.6577428,
            0.31812683,
            0.30241635,
            0.30539665,
            0.31872845,
            0.31743717,
            0.30246243,
            0.32773176,
            0.30487406,
            0.28169727,
            0.30730662,
            0.3194176,
            0.30304825,
            0.3349846,
            0.8119465,
            0.68115896,
            0.30843326,
            0.03188063,
            0.31986716,
            0.31734034,
            0.30467242,
            0.3069509,
            0.31586045,
            0.8980863,
            0.7939926,
            0.6178741,
            0.6591196,
            0.88688546,
            0.8157957,
            0.31430894,
            0.6145224,
        ),
        facemodel_param_value_other=(
            0.36496735,
            0.8932333,
            0.892712,
            0.6173538,
            0.5793705,
            0.5892547,
            0.55952805,
            0.5584179,
            0.7003825,
            0.5763332,
            0.5858158,
            0.77424514,
            0.06169198,
            0.70730615,
            0.9032952,
            0.72739,
            0.36457255,
            0.75539553,
            0.82653344,
            0.80979526,
            0.3658411,
            0.3694831,
            0.55880326,
            0.7823682,
            0.7110786,
            0.71527475,
            0.5629248,
            0.7651101,
            0.36850974,
            0.5434554,
            0.88069886,
            0.03267036,
            0.8685053,
            0.5839714,
            0.5623802,
            0.57563114,
            0.88853735,
            0.38872623,
            0.58695436,
            0.59864235,
            0.56485057,
            0.71454155,
            0.91420686,
            0.5560611,
            0.71141076,
            0.33341885,
            0.555688,
            0.5586237,
            0.7102673,
            0.58132946,
            0.55706894,
            0.8911129,
            0.61468506,
            0.803418,
            0.8109992,
            0.81329274,
            0.87104964,
            0.8003498,
            0.577305,
            0.883559,
            0.6069646,
            0.5957468,
            0.5712891,
            0.57914114,
            0.5714424,
            0.58212537,
            0.55858475,
            0.58807564,
            0.7544282,
            0.88710654,
            0.5875733,
            0.070957,
            0.5648147,
            0.5572599,
            0.5767184,
            0.38500753,
            0.6909044,
            0.8040567,
            0.69547576,
            0.7268255,
            0.7943563,
            0.912872,
            0.57904845,
            0.7630621,
            0.7871259,
            0.8822087,
            0.5931539,
            0.57819074,
            0.5572017,
            0.72343236,
            0.591899,
            0.81370443,
            0.8765678,
            0.57707125,
            0.39017484,
            0.5801955,
            0.5571239,
            0.56269836,
            0.5805456,
            0.58053017,
            0.55680126,
            0.5906761,
            0.55905014,
            0.5562931,
            0.56820613,
            0.5830169,
            0.55708784,
            0.6006494,
            0.80717707,
            0.7157721,
            0.5688318,
            0.05436677,
            0.5819246,
            0.57893044,
            0.55944055,
            0.56573135,
            0.57600063,
            0.8873194,
            0.7929609,
            0.3524004,
            0.39323917,
            0.8754569,
            0.8091996,
            0.5754221,
            0.6789727,
        ),
    )
