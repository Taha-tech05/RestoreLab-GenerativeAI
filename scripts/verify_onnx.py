"""Compare exported ONNX outputs with the corresponding PyTorch models (tolerance 1e-4)."""
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from common.models import Generator, build_classifier, build_moe, build_restorer
from common.paths import add_common_args, ensure_output_dirs, resolve_paths


class ClassifierProbs(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x): return torch.softmax(self.model(x),1)
class MoEOutputs(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x): return self.model(x,True)


RESULTS = []


def compare(name, model, feeds, inputs, device):
    session=ort.InferenceSession(str(name),providers=["CPUExecutionProvider"])
    with torch.no_grad(): expected=model(*[torch.from_numpy(feeds[k]).to(device) for k in inputs])
    expected=expected if isinstance(expected,tuple) else (expected,)
    actual=session.run(None,feeds)
    diffs=[float(np.max(np.abs(a-b.detach().cpu().numpy()))) for a,b in zip(actual,expected)]
    print(name.name,"max_abs_diff=",max(diffs))
    RESULTS.append({"model": name.name, "max_abs_diff": max(diffs), "tolerance": 1e-4})


def main():
    parser=add_common_args(argparse.ArgumentParser(),"task4.yaml")
    parser.add_argument("--onnx-dir",type=Path,default=None,help="Directory containing the seven exported ONNX files")
    args=parser.parse_args(); paths=resolve_paths(args); ensure_output_dirs(paths); device="cpu"
    ckpt=Path(paths["ckpt_dir"]); onnx=args.onnx_dir or (Path(paths["out_dir"])/"onnx"); rng=np.random.default_rng(42)
    x=rng.random((1,3,128,128),dtype=np.float32); tx=torch.from_numpy(x)
    c=torch.load(ckpt/"task1_spatial_best.pth",map_location=device); m=build_restorer(c["cfg"]); m.load_state_dict(c["state_dict"]); m.eval(); compare(onnx/"task1_universal.onnx",m,{"input":x},["input"],device)
    cc=torch.load(ckpt/"task2_classifier.pth",map_location=device); gate=build_classifier(cc["cfg"]); gate.load_state_dict(cc["state_dict"]); gate.eval(); compare(onnx/"task2_classifier.onnx",ClassifierProbs(gate),{"input":x},["input"],device)
    experts={}
    for i,name in enumerate(("salt","blur","occlusion"),1):
        ec=torch.load(ckpt/f"task2_spec_{name}.pth",map_location=device); e=build_restorer(ec["cfg"]); e.load_state_dict(ec["state_dict"]); e.eval(); experts[i]=e; compare(onnx/f"task2_specialist_{name}.onnx",e,{"input":x},["input"],device)
    mc=torch.load(ckpt/"task3_soft_moe.pth",map_location=device); moe_hp=mc.get("cfg",mc.get("hp")); g=build_classifier(mc["gate_cfg"]); g.load_state_dict({k.removeprefix("gate."):v for k,v in mc["state_dict"].items() if k.startswith("gate.")})
    moe=build_moe(g,experts,moe_hp["tau"]); moe.load_state_dict(mc["state_dict"]); moe.eval(); compare(onnx/"task3_soft_moe.onnx",MoEOutputs(moe),{"input":x},["input"],device)
    hp=paths["config"].get("final",{}); sk=Generator(hp.get("c",64),hp.get("emb",16),hp.get("drop",.3))
    face_ckpt=ckpt/"face2sketch_generator.pth"
    if not face_ckpt.exists(): face_ckpt=ckpt/"G_best.pt"
    sk.load_state_dict(torch.load(face_ckpt,map_location=device)); sk.eval()
    photo=rng.random((1,3,128,128),dtype=np.float32)*2-1; style=np.zeros((1,),dtype=np.int64)
    compare(onnx/"face2sketch.onnx",sk,{"photo":photo,"style":style},["photo","style"],device)
    (Path(paths["out_dir"])/"onnx_consistency.json").write_text(json.dumps(RESULTS,indent=2),encoding="utf-8")


if __name__=="__main__": main()
