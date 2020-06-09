

image_dir = './tmp.png'
det_model_dir = 'model_train/detect_model'
det_param_dir = 'model_train/detect_params'
rec_model_dir = 'model_train/recognize_model'
rec_param_dir = 'model_train/recognize_params'


use_gpu = False
ir_optim = True
use_tensorrt = False
gpu_mem = 8000
det_algorithm = 'DB'
det_max_side_len = 960
det_db_thresh = 0.3
det_db_box_thresh = 0.5
det_db_unclip_ratio = 2.0
det_east_score_thresh = 0.8
det_east_cover_thresh = 0.1
det_east_nms_thresh = 0.2
rec_algorithm = 'CRNN'
rec_image_shape = '3, 32, 320'
rec_char_type = 'ch'
rec_batch_num = 30
rec_char_dict_path = 'model_data/vocabs.txt'
font_path = 'model_data/simfang.ttf'
output_pach = 'output'