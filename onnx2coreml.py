import sys
import coremltools as ct
from onnx_coreml import convert
import onnx

compound_coef = 0
onnx_name = "onnx_efficientdet_d%d.onnx" % compound_coef
model_file = open(onnx_name, 'rb')
model_proto = onnx.onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto, image_input_names=['0'])
