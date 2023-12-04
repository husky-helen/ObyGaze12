from transformers import XCLIPProcessor, XCLIPModel
from transformers import VivitImageProcessor, VivitModel

def load_model(model_name):

    if 'xclip' in model_name:
        processor = XCLIPProcessor.from_pretrained(model_name)
        model = XCLIPModel.from_pretrained(model_name)
        
    elif 'vivit' in model_name:
        processor = VivitImageProcessor.from_pretrained(model_name)
        model = VivitModel.from_pretrained(model_name)

    else:
        processor, model = None, None
    
    return processor, model