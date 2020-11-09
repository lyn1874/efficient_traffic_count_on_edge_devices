import coremltools as ct
import utils.utils_npy as input_utils
import numpy as np

def predict_coreml():
    project_name = "coco"
    compound_coef = 0
    input_size = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    img_path = "frames/clip/frame_00000020.jpg"
    time_init = time.time()
    ori_imgs, framed_imgs, \
        framed_metas = input_utils.preprocess(img_path, max_size=input_size[compound_coef])
    x_npy = np.transpose(np.expand_dims(framed_imgs[0], 0), (0, 3, 1, 2)).astype('float32')
    model_ml = ct.models.MLModel("coreml_efficientdet_d0.mlmodel")
    ml_output = model_ml.predict({"input": x_npy})
    return ml_output