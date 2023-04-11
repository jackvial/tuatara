import os
import glob
import torch
from PIL import Image
from torchvision import transforms as T

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


preprocess = T.Compose(
    [
        T.Resize((32, 128), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)


model_fp32 = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=True).eval()
# compiled_model = torch.compile(model)
print("model_fp32: ", model_fp32)

model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

f=print_size_of_model(model_fp32,"fp32")
q=print_size_of_model(model_int8,"int8")
print("{0:.2f} times smaller".format(f/q))

im_path = "/Users/jackvial/Code/CPlusPlus/tuatara/images/art-01107.jpg"
image = preprocess(Image.open(im_path).convert("RGB")).unsqueeze(0)
# Greedy decoding
pred = model_int8(image).softmax(-1)
label, _ = model_int8.tokenizer.decode(pred)
raw_label, raw_confidence = model_int8.tokenizer.decode(pred, raw=True)
print(raw_label)

traced_model = torch.jit.trace(model_int8, torch.rand(image.shape))
traced_model.save("../weights/parseq_int8_torchscript.pt")
