#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    den=area(a)+area(b)-inter
    return inter/den if den > 0 else 0.0

def px(b, W, H):
    return [b[0]*W, b[1]*H, b[2]*W, b[3]*H]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--change-report", required=True)
    ap.add_argument("--diagnostics", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args=ap.parse_args()

    base=load_json(args.base)
    pilot=load_json(args.pilot)
    chg=load_json(args.change_report).get("changes",{})
    diag=load_json(args.diagnostics)

    out=Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary=[]

    for qid, c in chg.items():
        b0=base[qid]["bbox"]
        b1=pilot[qid]["bbox"]
        ent=base[qid]

        visible=ent.get("visible")
        if not visible:
            print("NO visible path:", qid)
            continue

        img_path=Path(args.data_root)/visible
        if not img_path.exists():
            print("MISSING:", img_path)
            continue

        img=Image.open(img_path).convert("RGB")
        W,H=img.size
        draw=ImageDraw.Draw(img)

        # RED = base, GREEN = R5A
        draw.rectangle(px(b0,W,H), outline="red", width=5)
        draw.rectangle(px(b1,W,H), outline="lime", width=5)

        title=f"{qid} | RED=BASE | GREEN=R5A"
        draw.rectangle([0,0,min(W,900),72], fill="white")
        draw.text((10,8), title, fill="black")
        draw.text((10,38), ent.get("query","")[:120], fill="black")

        d=diag.get(qid,{})
        program=d.get("semantic_program",{})
        ptype=program.get("program_type")
        subtype=d.get("failure_subtype")

        ratio=area(b1)/max(area(b0),1e-12)
        ov=iou(b0,b1)

        save_path=out/f"{qid}_base_red_r5a_green.jpg"
        img.save(save_path, quality=92)

        summary.append({
            "query_id": qid,
            "query": ent.get("query"),
            "program_type": ptype,
            "diagnostic_subtype": subtype,
            "base_bbox": b0,
            "r5a_bbox": b1,
            "base_area": area(b0),
            "r5a_area": area(b1),
            "area_ratio_r5a_over_base": ratio,
            "iou_base_vs_r5a": ov,
            "overlay": str(save_path),
        })

    summary.sort(key=lambda x: x["iou_base_vs_r5a"])
    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)

    print("\nR5-A changed-query inspection")
    print("="*90)
    for x in summary:
        print(
            f'{x["query_id"]}  '
            f'IoU(base,R5A)={x["iou_base_vs_r5a"]:.3f}  '
            f'area_ratio={x["area_ratio_r5a_over_base"]:.2f}  '
            f'program={x["program_type"]}'
        )
    print("\nOverlays:", out)
    print("Summary :", out/"summary.json")

if __name__=="__main__":
    main()
