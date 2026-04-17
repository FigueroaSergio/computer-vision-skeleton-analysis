

MODELS_CONFIG = [
    {
        "name": "ST_GCN-f10-L3",
        "type": "ST_GCN",
        "weights_path": "models/ST_CGN-f10-L3.h5",
        "n_frames": 10,
        "layers": 3
    },
       {
        "name": "ST_GCN-f15-L3",
        "type": "ST_GCN",
        "weights_path": "models/ST_CGN-f15-L3.h5",
        "n_frames": 15,
        "layers": 3
    },
    {
        "name": "ST_GCN-f10",
        "type": "ST_GCN",
        "weights_path": "models/ST_CGN-f10.h5",
        "n_frames": 10,
        "layers": 2
    },
    {
        "name": "ST_GCN-f15-L2",
        "type": "ST_GCN",
        "weights_path": "models/ST_CGN-f15-L2.h5",
        "n_frames": 15,
        "layers": 2
    },
    # SPIL --------------------------------
    {
        "name": "SPIL-f10",
        "type": "SPIL",
        "weights_path": "models/SPIL-f10.keras",
        "n_frames": 10,
        "n_points": 1024
    },
    {
        "name": "SPIL-f15",
        "type": "SPIL",
        "weights_path": "models/SPIL-f15.keras",
        "n_frames": 15,
        "n_points": 1024
    },
    # {
    #     "name": "SPIL-Base",
    #     "type": "SPIL",
    #     "weights_path": "models/SPIL.keras",
    #     "n_frames": 10,
    #     "n_points": 1024
    # },

    # CONV3D---------------------------------------------
    {
        "name": "PoseConv3D-Limbs",
        "type": "PoseConv3D",
        "weights_path": "models/Limbs_PoseConv3D-f10.keras",
        "n_frames": 10,
        "width": 128,
        "height": 128
    },
    {
        "name": "PoseConv3D-Limbs-f15",
        "type": "PoseConv3D",
        "weights_path": "models/Limbs_PoseConv3D-f15.keras",
        "n_frames": 15,
        "width": 128,
        "height": 128
    },
    # {
    #     "name": "PoseConv3D-Joints",
    #     "type": "PoseConv3D",
    #     "weights_path": "models/Joints_PoseConv3D.h5",
    #     "n_frames": 10,
    #     "width": 128,
    #     "height": 128
    # }
]
