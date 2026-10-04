"""Export seven embedded-weight ONNX models from trained PyTorch checkpoints."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from common.models import Generator, build_classifier, build_moe, build_restorer
from common.paths import add_common_args, ensure_output_dirs, resolve_paths
from common.utils import get_device


class ClassifierProbs(nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, x): return torch.softmax(self.model(x), dim=1)


class MoEOutputs(nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, x): return self.model(x, True)


def main():
    args=add_common_args(argparse.ArgumentParser(),"task4.yaml").parse_args(); paths=resolve_paths(args); ensure_output_dirs(paths)
    device=get_device(); ckpt=Path(paths["ckpt_dir"]); out=Path(paths["out_dir"])/"onnx"; out.mkdir(parents=True,exist_ok=True)
    def load(name): return torch.load(ckpt/name,map_location=device)
    t1=load("task1_spatial_best.pth"); model=build_restorer(t1["cfg"]).to(device); model.load_state_dict(t1["state_dict"]); model.eval()
    torch.onnx.export(model,torch.rand(1,3,128,128,device=device),out/"task1_universal.onnx",input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"B"},"output":{0:"B"}},opset_version=17,dynamo=False)
    cc=load("task2_classifier.pth"); classifier=build_classifier(cc["cfg"]).to(device); classifier.load_state_dict(cc["state_dict"]); classifier.eval()
    torch.onnx.export(ClassifierProbs(classifier),torch.rand(1,3,128,128,device=device),out/"task2_classifier.onnx",input_names=["input"],output_names=["probs"],dynamic_axes={"input":{0:"B"},"probs":{0:"B"}},opset_version=17,dynamo=False)
    specialists={}
    for label,name in ((1,"salt"),(2,"blur"),(3,"occlusion")):
        c=load(f"task2_spec_{name}.pth"); expert=build_restorer(c["cfg"]).to(device); expert.load_state_dict(c["state_dict"]); expert.eval(); specialists[label]=expert
        torch.onnx.export(expert,torch.rand(1,3,128,128,device=device),out/f"task2_specialist_{name}.onnx",input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"B"},"output":{0:"B"}},opset_version=17,dynamo=False)
    mc=load("task3_soft_moe.pth"); gate=build_classifier(mc["gate_cfg"]).to(device)
    gate.load_state_dict({k.removeprefix("gate."):v for k,v in mc["state_dict"].items() if k.startswith("gate.")})
    moe_hp=mc.get("cfg",mc.get("hp")); moe=build_moe(gate,specialists,moe_hp["tau"]).to(device); moe.load_state_dict(mc["state_dict"]); moe.eval()
    torch.onnx.export(MoEOutputs(moe),torch.rand(1,3,128,128,device=device),out/"task3_soft_moe.onnx",input_names=["input"],output_names=["restored","weights"],dynamic_axes={"input":{0:"B"},"restored":{0:"B"},"weights":{0:"B"}},opset_version=17,dynamo=False)
    (out/"moe_tau.json").write_text(json.dumps({"tau":moe_hp["tau"]},indent=2),encoding="utf-8")
    face_ckpt=ckpt/"face2sketch_generator.pth"
    if not face_ckpt.exists(): face_ckpt=ckpt/"G_best.pt"
    fc=torch.load(face_ckpt,map_location=device); hp=paths["config"].get("final",{})
    sketch=Generator(hp.get("c",64),hp.get("emb",16),hp.get("drop",0.3)).to(device); sketch.load_state_dict(fc); sketch.eval()
    photo=torch.rand(1,3,128,128,device=device)*2-1; style=torch.zeros(1,dtype=torch.int64,device=device)
    torch.onnx.export(sketch,(photo,style),out/"face2sketch.onnx",input_names=["photo","style"],output_names=["sketch"],dynamic_axes={"photo":{0:"B"},"style":{0:"B"},"sketch":{0:"B"}},opset_version=17,dynamo=False)
    print("ONNX exports written to",out)


if __name__=="__main__": main()
