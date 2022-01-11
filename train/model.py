import torch
import onnx
from onnx2pytorch import ConvertModel


def load_model(path_to_onnx_model, trainable_layers=[], batch_size=1):
    onnx_model = onnx.load(path_to_onnx_model)
    model = ConvertModel(onnx_model, experimental=True)  # pretrained_model

    # hack to enable batch_size > 1
    model.Constant_1047.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1048.shape = (batch_size, 2, 66)
    model.Constant_1049.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1050.shape = (batch_size, 2, 66)
    model.Constant_1051.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1052.shape = (batch_size, 2, 66)
    model.Constant_1053.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1054.shape = (batch_size, 2, 66)
    model.Constant_1057.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1058.shape = (batch_size, 2, 66)
    model.Constant_1059.constant = torch.tensor((batch_size, 2, 66))
    model.Reshape_1060.shape = (batch_size, 2, 66)
    model.Elu_907.inplace = False

    def reinitialise_weights(layer_weight):
        model.layer_weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(layer_weight))

    for name, layer in model.named_modules():
        if name in trainable_layers:
            reinitialise_weights(layer.weight)
            layer.bias.data.fill_(0.01)
        else:
            layer.requires_grad_(False)
            layer.train(False)

    return model
