DATA_NAME=$1
v2x_side=$2

python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./data/infos/${DATA_NAME} \
    --v2x-side ${v2x_side}

python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/${DATA_NAME} \
    --save-root ./datasets/${DATA_NAME} \
    --v2x-side ${v2x_side}

python tools/spd_data_converter/map_spd_to_nuscenes.py \
    --maps-root ./datasets/${DATA_NAME}/maps \
    --save-root ./datasets/${DATA_NAME} \
    --v2x-side ${v2x_side}