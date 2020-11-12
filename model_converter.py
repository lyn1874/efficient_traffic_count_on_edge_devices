import coremltools as ct
import yaml
import torch
import torch.onnx
import numpy as np
import os
import backbone as bb
import onnx
import onnxruntime
import imp
import time
import matplotlib.pyplot as plt
import utils.utils as input_utils
import prediction as model_arch
import vis_utils as vu
import utils.utils_npy as input_utils
import utils.utils as input_utils_tensor
import numpy as np
import efficientdet.utils as eff_utils
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class ModelConversion(object):
    def __init__(self):
        self.input_size = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.project_name = "coco"
        self.compound_coef = 0
        self.sample_batch_size = 1
        self.device = torch.device("cpu")
        self.onnx_name = "onnx_efficientdet_d%d.onnx" % self.compound_coef
        self.coreml_name = "coreml_efficientdet_d%d.mlmodel" % self.compound_coef
        params = yaml.safe_load(open(f'parameter/{self.project_name}.yml'))
        obj_list = params['obj_list']    
        weights_path = 'checkpoints/efficientdet-d%d.pth' % 0

        model = bb.EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), 
                                     scales=eval(params['anchors_scales']),
                                     onnx_export=True).to(self.device)

        model.backbone_net.model.set_swish(memory_efficient=False)
        model.load_state_dict(torch.load(weights_path))  # , map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.image_path="/project_scratch/bo/normal_data/aic2020/AIC20_track1/Dataset_A_Frame/"
        self.image_path += "cam_1_dawn/frame_00000304.jpg"

    def torch2onnx(self):
        if not os.path.isfile(self.onnx_name):
            dummy_input = torch.randn((self.sample_batch_size, 3, 
                                       self.input_size[self.compound_coef], 
                                       self.input_size[self.compound_coef]), dtype=torch.float32).to(self.device)

            # opset_version can be changed to 10 or other number, based on your need
            torch.onnx.export(self.model, dummy_input,
                              self.onnx_name,
                              verbose=False,
                              opset_version=11,
                              input_names = ['input'],  
                              output_names = ['output'])
            print("sucessfully converting the PyTorch model to ONNX model")
            print("next check the onnx model")
        model_onnx = onnx.load(self.onnx_name)
        # Check that the IR is well formed
        onnx.checker.check_model(model_onnx)
        # Print a human readable representation of the graph
        ort_session = onnxruntime.InferenceSession(self.onnx_name)

        time_init = time.time()
        ori_imgs, framed_imgs, \
            framed_metas = input_utils.preprocess(self.image_path, 
                                                  max_size=self.input_size[self.compound_coef])
        x = torch.from_numpy(framed_imgs[0]).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        regression, classification, anchors = ort_outs

        threshold = 0.3
        nms_threshold = 0.4
        preds = input_utils.postprocess_npy(anchors, regression, classification, threshold, nms_threshold)
        preds = input_utils.invert_affine_npy(framed_metas, preds)[0]
        print("inference time", 1/(time.time() - time_init))

        vu.show_bbox(ori_imgs[0], preds[:, :4], preds[:, 4], "pred", show=True)
        
    def torch2coreml(self):        
        dummy_input = torch.randn((self.sample_batch_size, 3, 
                                   self.input_size[self.compound_coef], 
                                   self.input_size[self.compound_coef]))
        for i in range(1):
            try:
                traced_model = torch.jit.trace(self.model, dummy_input)
            except:
                print("Trace error")
                pass
        traced_model = torch.jit.trace(self.model, dummy_input)
        print("sucessfully traced the model")

        time_init = time.time()
        ori_imgs, framed_imgs, \
            framed_metas = input_utils.preprocess(self.image_path, max_size=self.input_size[self.compound_coef])
        x = torch.from_numpy(framed_imgs[0]).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        trace_output = traced_model(x)
        regression, classification, anchors = trace_output

        threshold = 0.3
        nms_threshold = 0.4
        preds = input_utils_tensor.postprocess(anchors, regression, classification, threshold, nms_threshold)
        preds = input_utils_tensor.invert_affine(framed_metas, preds)[0]
        print("inference time", 1/(time.time() - time_init))
        vu.show_bbox(ori_imgs[0], preds["rois"], preds["class_ids"], "pred", show=True)
        
        @register_torch_op(override=True)
        def constant_pad_nd(context, node):
            inputs = _get_inputs(context, node, expected=3)
            # print(inputs[1].val, inputs[0].shape)
            new_pad = inputs[1].val.reshape((-1, 2))[::-1].reshape(-1).tolist()
            new_pad = [0]*(2*len(inputs[0].shape)-len(new_pad)) + new_pad
        #     print(new_pad, inputs[0].shape)
            padded = mb.pad(x=inputs[0], pad=np.array(new_pad), mode="constant", constant_val=float(0), name=node.name)
            context.add(padded)
        @register_torch_op(override=True)
        def select(context, node):
            inputs = _get_inputs(context, node, expected=3)
            _input = inputs[0]
            dim = inputs[1].val
            index = inputs[2].val
            assert dim.shape == ()
            assert index.shape == ()
        #     assert _input.val is None

            # NOTE:
            # Each index in @begin_array/@end_array corresponds to a dimension of @_input
            # Each val of those arrays corresponds to the start/end index to slice in that dimension
            begin_array = [0] * len(_input.shape)
            begin_array[dim] = index
            end_array = [s if isinstance(s, int) else 0 for s in _input.shape]
            end_mask = [True] * len(_input.shape)
            if index != -1:
                end_array[dim] = index + 1
                end_mask[dim] = False

            slice_by_index = mb.slice_by_index(
                x=_input,
                begin=begin_array,
                end=end_array,
                end_mask=end_mask,
                name=node.name + "_slice_by_index",
            )
            # Now we squeeze the dimension we have selected from to remove it
            squeeze = mb.squeeze(
                x=slice_by_index, axes=np.array([dim]), name=node.name + "_squeeze"
            )
            context.add(squeeze, node.name)
            
        @register_torch_op(override=True)
        def clamp(context, node):
            inputs = _get_inputs(context, node, expected=3)
            context.add(mb.clip(x=inputs[0], alpha=inputs[1], beta=inputs[2], name=node.name))
            
        ml_dummy = ct.TensorType(name='input', shape=dummy_input.shape)
        model_ml = ct.convert(traced_model, inputs=[ml_dummy]) # I think the output has some problems!
        model_ml.input_description["input"] = "Input image to be classified"
        # Set model author name
        model_ml.author = 'bobo'
       # Set a version for the model
        model_ml.version = "0.0"
        model_ml.save(self.coreml_name)

    