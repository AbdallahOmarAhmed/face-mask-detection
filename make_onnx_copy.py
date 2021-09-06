
import torchvision

from torch import nn
import torch.onnx

from models.faceboxes import FaceBoxes

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def load_detection_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
def export_onnx(model,input_size,path):
    rand_input = torch.randn(input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      rand_input,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)

                      export_params=True,  # store the trained parameter weights inside the model file
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names




model = torchvision.models.resnet18()
n = model.fc.in_features
model.fc = nn.Linear(n, 1)
model.load_state_dict(torch.load("weights/modelTR.pth"))
model.eval()

net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
net = load_detection_model(net, 'weights/FaceBoxes.pth', False)
net.eval()

input_size = (1,3,256,256)
export_onnx(model,input_size,"weights/maskDetectionModel.onnx")
export_onnx(net,input_size,"weights/faceDetectionModel.onnx")