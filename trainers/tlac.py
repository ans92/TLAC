import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


from .dataset import DataManager


from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai


_tokenizer = _Tokenizer()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class TLAC(TrainerX):
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TLAC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames        
        
        self.classes = classnames

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        model_id = cfg.GEMINI_MODEL_ID_2 
             
        API_KEY = cfg.GEMINI_API_KEY
        genai.configure(api_key=API_KEY)
        
        self.mm_model = genai.GenerativeModel(model_name=model_id)

    def get_clip_class_features(self, classes):
        
        pairs = [name.replace("_", " ") for name in classes]
        pair_inputs = self.clip_processor(text=[f"a photo of a {c}" for c in pairs], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**pair_inputs)
        text_feat = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_feat
    
    
    def build_data_loader(self):
        dm = DataManager(self.cfg)
        
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        
        self.dm = dm
    
    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        
        if split is None:
            split = self.cfg.TEST.SPLIT
        
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        self.text_feat = self.get_clip_class_features(self.classes)

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            
            input, label, mm_pred, image_name = self.parse_batch_test(batch)
            ground_truth = [self.classes[ind] for ind in label.tolist()]
            
            ground_truth = [str(item).lower() if isinstance(item, str) else item for item in ground_truth]
            ground_truth = [item.replace("_", " ") if isinstance(item, str) else item for item in ground_truth]
            
            if self.cfg.MODEL_NAME == 'TLAC':
                second_mm_pred = []
                for ind,(a,b) in enumerate(zip(mm_pred, ground_truth)):
                    prompt = f"I have a list of classes: {self.classes} and a predicted class: {a}. Identify the exact matching class from the list? If exact match does not exists then give the sementically closest class without give any extra description and words such as 'There are not exact match and the closed match is'. Just give either exact match or closest match."
                    
                    try:
                        response = self.mm_model.generate_content([prompt])
                        ans = response.text
                        ans = ans.replace("\n", "")
                        ans = ans.replace(".", "")
                        ans = ans.replace("_", " ")                    
                        second_mm_pred.append(ans)     
                    except Exception as e:
                        ans = 'class'
                        second_mm_pred.append(ans)
                                               
            

                mm_pred_inputs = self.clip_processor(text=[f"a photo of a {c}" for c in second_mm_pred], return_tensors="pt", padding=True, max_length=77, truncation=True)
            else:
                mm_pred_inputs = self.clip_processor(text=[f"a photo of a {c}" for c in mm_pred], return_tensors="pt", padding=True, max_length=77, truncation=True)
            
            with torch.no_grad():
                mm_pred_features = self.clip_model.get_text_features(**mm_pred_inputs)
            mm_pred_features = mm_pred_features / mm_pred_features.norm(dim=-1, keepdim=True)
            output = mm_pred_features @ self.text_feat.t()
            #output = output.to('cuda')
            
            self.evaluator.process(output, label);break

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        
        input = input.to(self.device)
        label = label.to(self.device)        
        
        mm_pred = batch["mm_pred"]
        image_name = batch["impath"]
        
        return input, label, mm_pred, image_name

        