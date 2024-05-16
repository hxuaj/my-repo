# -*- coding: utf-8 -*-
import timm
import os
# from torchvision.io import read_image
from torchvision.transforms import v2, InterpolationMode
from PIL import Image
import torch
import json
import io
from flask import Flask, jsonify, request
import base64
import sys
import gc


class ViT_Flask(object):
    def __init__(self):
        super(ViT_Flask, self).__init__()
        # model and config
        self.root_path = os.path.dirname(os.getcwd())
        self.model_dir = os.path.join(self.root_path, 'model', 'vit_timm')
        self.cfg_path = os.path.join(self.model_dir, 'config.json')
        self.checkpoint_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 21k label json
        # self.label_filename = "imagenet-22k-id2label.json"
        self.label_filename = "imagenet-22k-label-cn.json"
        self.label_path = os.path.join(self.root_path, self.label_filename)

        # inference configs
        self.batch_size = 128
        self.top_k = 1 # return top_k number of predictions per image

        # load model when initialize
        # self.model, self.transforms_trimm = self.load_model()

    def load_model(self):
        """
        Load ViT model and configs.
        File paths:
        root
        ├── model
        │     └── vit_timm
        │          ├── config.json
        │          └── pytorch_model.bin
        └── flask_depo
        │     └── app.py
        └── imagenet-22k-id2label.json
        Output:
        - model, ViT model pretrained in ImageNet 21k.
        - transforms_timm, images preprocess transform
        """

        model = timm.create_model('vit_base_patch8_224.augreg_in21k', 
                            # pretrained=True, 
                            # pretrained_cfg=cfg_path,
                            checkpoint_path=self.checkpoint_path)
        
        model.eval() # set the model in evaluation mode
        model.to(self.device)
        data_config = timm.data.resolve_model_data_config(model)
        transforms_timm = timm.data.create_transform(**data_config, is_training=False)

        # Compose(
        #     Resize(size=248, interpolation=bicubic, max_size=None, antialias=True)
        #     CenterCrop(size=(224, 224))
        #     ToTensor()
        #     Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
        # )
        transforms_timm = v2.Compose([v2.ToDtype(torch.float32, scale=True), 
                                    #   v2.Resize(248, interpolation=InterpolationMode.BICUBIC, 
                                    #             max_size=None, antialias=True),
                                      v2.CenterCrop(size=(224, 224)), 
                                    #   v2.ToTensor(),
                                      v2.Normalize(mean=[0.5000, 0.5000, 0.5000], 
                                                   std=[0.5000, 0.5000, 0.5000])])

        return model, transforms_timm

    def preprocess(self, image_bytes, transforms):
        """
        Transform image bytes into tensor with preprocess.
        Input:
        - image_bytes, list of strings, input images encoded in base64.
        - transforms, torchvision transforms for preprocessing.
        Output:
        - 
        """
        images = []
        for ib in image_bytes:
            image = Image.open(io.BytesIO(ib))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            # convert PIL image to tensor
            image_tensor = v2.functional.to_image(image)
            image_tensor = v2.functional.resize(image_tensor, [248, 248], 
                                                interpolation=InterpolationMode.BICUBIC, 
                                                max_size=None, antialias=True)
            images.append(image_tensor)
            # print(image_tensor.shape)
        images = torch.stack(images, dim=0)
        # print(images.shape) # shape order: (N, C, H, W)
        return transforms(images)

    def postprocess(self, model_outputs):
        """
        Postprocess the output of model. 
        """

        id2label = json.load(open(self.label_path, "r", encoding='utf-8'))
        id2label = {int(k):v for k,v in id2label.items()}

        topk_prob, topk_class = torch.topk(model_outputs.softmax(dim=1) * 100, k=self.top_k)
        # print(topk_prob.tolist(), topk_class.tolist())
        # print(id2label[topk_class.cpu().numpy()[0][0]])

        probs = topk_prob.tolist()
        classes = topk_class.tolist()
        # torch.cuda.empty_cache()
        res_probs = []
        res_classes = []
        for i in range(len(classes)):
            # print(classes[i])
            name = [id2label[id] for id in classes[i]]
            res_probs.append(probs[i])
            res_classes.append(name)
        # print("inside get_prediction --> ", res_classes, res_probs)

        return res_classes, res_probs

    def run(self):
        pass


app = Flask(__name__)
# load model
vit = ViT_Flask()
model, transforms_trimm = vit.load_model()

# enable muti-gpu inference
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        try:
            keyword = 'inputs'
            input_json = request.json
            # if 'file' not in request.files:
            #     return jsonify({'error': 'Keyword "file" is required in the request.'})
            if keyword not in input_json:
                return jsonify({'error': 'Keyword {keyword} is required in the request.'})
            
            # process inputs in batch
            filenames = []
            image_bytes = []
            for input in input_json.get(keyword):
                # print(input.keys())

                _, ext = os.path.splitext(input['filename'])
                if ext.lower() not in {'.jpg', '.jpeg', '.png', '.tif'}:
                    return jsonify({'error': 'The input file should be image.'})
                
                filenames.append(input['filename'])
                image_bytes.append(base64.urlsafe_b64decode(input['image']))
            
            classes_res, probs_res = [], []
            # split by batch size
            for i in range(0, len(image_bytes), vit.batch_size):
                print(len(image_bytes[i:i + vit.batch_size]))
                bytes_batch = image_bytes[i:i + vit.batch_size]

                # model inference
                with torch.no_grad():
                    image_inputs = vit.preprocess(bytes_batch, transforms_trimm)
                    outputs = model(image_inputs.to(vit.device))
                    classes, probs = vit.postprocess(outputs)
                    # concatenate results
                    classes_res += classes
                    probs_res += probs

                # gc.collect()
                # torch.cuda.empty_cache()

            # print(type(classes), type(probs))
            # print(classes_res)
            torch.cuda.empty_cache()

            if len(filenames) != len(classes_res):
                return jsonify({'class_name': classes_res, 
                                'probs': probs_res, 
                                'warning': f"The input length ({len(filenames)}) and output length ({len(classes_res)}) are not match!"})

            return jsonify({'class_name': classes_res, 'probs': probs_res})
        
        except Exception as e:
            torch.cuda.empty_cache()
            return jsonify({'error': str(e)})

@app.route('/release', methods=['POST'])    
def release():
    if request.method == 'POST':
        torch.cuda.empty_cache()
    
    return jsonify({'state': "finish memory release."})

@app.route('/restart', methods=['POST'])
def restart():
    if request.method == 'POST':
        # Perform any necessary cleanup or shutdown tasks
        # torch.cuda.empty_cache()

        # response = {'message': 'Server restart initiated'}
        # # Send the response to the client
        # response = jsonify(response)
        # response.status_code = 200
        # response.autocorrect_location_header = False  # Disable autocorrection of the Location header
        # response.direct_passthrough = True
    
        # Restart the Flask app by reloading the Python process
        python = sys.executable
        os.execl(python, python, *sys.argv)
    
            

if __name__ == '__main__':

    app.run(host='0.0.0.0', 
            port=5000, 
            debug=True)
    
    # img_path = os.path.join(vit.root_path, 'lemon_pic.jpg')
    # with open(img_path, 'rb') as img_file:
    #     image_bytes = img_file.read()
    # image_inputs = vit.preprocess([image_bytes] * 2, transforms_trimm)

    # outputs = model(image_inputs.to(vit.device))

    # vit.postprocess(outputs)


    # # image preprocess
    # img_path = os.path.join(root_path, 'lemon_pic.jpg')
    # # img = read_image(img_path)
    # img = Image.open(img_path)
    # if img.mode == "RGBA":
    #     img = img.convert("RGB")
    # img_input = transforms_timm(img)

    # # model inference
    # device = torch.device("cuda")
    # model.to(device)
    # output = model(img_input.unsqueeze(dim=0).to(device))
    # topk_prob, topk_class = torch.topk(output.softmax(dim=1) * 100, k=1)
    # print(topk_prob, topk_class)

    # repo_id = "huggingface/label-files"
    # filename = "imagenet-22k-id2label.json"
    # id2label = json.load(open(os.path.join(root_path, filename), "r"))
    # id2label = {int(k):v for k,v in id2label.items()}

    # print(topk_class.cpu().numpy()[0][0])
    # print(id2label[topk_class.cpu().numpy()[0][0]])
