import json
import math
import itertools
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ============================================================
# Stage 2.2 Hybrid
# - Keep Qwen as the final semantic judge.
# - Conservative loose-box suppression only for very obvious duplicates.
# - Query-conditioned RGB / Depth / IR routing.
# - Geometry is a SOFT HINT, not a hard filter.
# - Depth relations are stronger numerical evidence, but Qwen still makes final choice.
# - Reference candidates are shown visually to Qwen for joint target-reference reasoning.
# ============================================================

PROJECT_ROOT = Path('/root/autodl-tmp/aic_v1_project')
DATA_ROOT = PROJECT_ROOT / 'data/official/初赛数据集-基于大模型的多模态视觉理解与推理'
RGB_PATH = DATA_ROOT / 'Images/visible/000002.png'
IR_PATH = DATA_ROOT / 'Images/infrared/000002.png'
DEPTH_PATH = DATA_ROOT / 'Images/depth/000002.png'
INPUT_JSON = PROJECT_ROOT / 'outputs/stage1b_000002/stage1b_candidates.json'
OUTPUT_DIR = PROJECT_ROOT / 'outputs/stage2_2_000002'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / 'stage2_2_predictions.json'
HF_HUB = PROJECT_ROOT / 'models/hf_cache/hub'

MAX_SHORTLIST = 8
MAX_REFS = 5
MAX_GROUP_BASE = 6


def find_snapshot(repo):
    root = HF_HUB / repo / 'snapshots'
    for p in root.iterdir():
        if p.is_dir() and (p / 'config.json').exists():
            return str(p)
    raise RuntimeError(f'No snapshot for {repo}')


QWEN_PATH = find_snapshot('models--Qwen--Qwen3-VL-8B-Instruct')

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

rgb_pil = Image.open(RGB_PATH).convert('RGB')
ir_cv = cv2.imread(str(IR_PATH), cv2.IMREAD_UNCHANGED)
depth = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_UNCHANGED)
if ir_cv is None or depth is None:
    raise RuntimeError('Failed to load IR/depth image')
H, W = depth.shape[:2]

# Depth visualization. Only sent to Qwen when query explicitly needs depth.
valid = depth > 0
depth_vis = np.zeros((H, W), dtype=np.uint8)
if valid.any():
    vals = depth[valid].astype(np.float32)
    lo, hi = np.percentile(vals, [2, 98])
    norm = (depth.astype(np.float32) - lo) / max(float(hi - lo), 1.0)
    norm = np.clip(norm, 0, 1)
    depth_vis = ((1.0 - norm) * 255).astype(np.uint8)
    depth_vis[~valid] = 0
depth_vis_path = OUTPUT_DIR / 'depth_visualization.png'
Image.fromarray(depth_vis).convert('RGB').save(depth_vis_path)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def horizontal_overlap(a, b):
    ov = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    return ov / denom


def vertical_overlap(a, b):
    ov = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return ov / denom


def union_boxes(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def inner_box(box, ratio=0.65):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


def depth_stats(box):
    x1, y1, x2, y2 = inner_box(box)
    roi = depth[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return {'valid_ratio': 0.0, 'median_mm': None}
    mask = (roi > 300) & (roi <= 19999)
    vals = roi[mask]
    vr = float(vals.size / roi.size)
    return {
        'valid_ratio': round(vr, 4),
        'median_mm': int(np.median(vals)) if vals.size else None,
    }


DEPTH_RELATIONS = {
    'in front of', 'behind', 'nearest', 'closest',
    'farthest', 'farther', 'closer'
}
IR_WORDS = {
    'hot', 'hottest', 'warm', 'warmer', 'warmest',
    'heat', 'thermal', 'cold', 'colder'
}


def get_routing(parsed, query):
    relations = [str(x.get('relation', '')).lower() for x in parsed.get('constraints', [])]
    q = query.lower()
    return {
        'RGB': True,
        'geometry': bool(parsed.get('constraints')) or parsed.get('ordinal', 'none') != 'none',
        'depth': any(r in DEPTH_RELATIONS for r in relations),
        'infrared': any(w in q for w in IR_WORDS),
    }


# ============================================================
# Conservative loose-box suppression
# Intentionally stricter than Stage 2.1.
# 003 T6 -> T1 still triggers, but marginal cases should not.
# ============================================================

def tightness_filter(candidates, count):
    if count != 1:
        return [dict(x) for x in candidates], []

    candidates = [dict(x) for x in candidates]
    suppressed = set()
    reasons = []

    for i in range(len(candidates)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue

            a, b = candidates[i], candidates[j]
            ov = iou(a['bbox'], b['bbox'])
            if ov < 0.40:
                continue

            aa, ab = area(a['bbox']), area(b['bbox'])
            if aa <= ab:
                small_idx, large_idx = i, j
            else:
                small_idx, large_idx = j, i

            small, large = candidates[small_idx], candidates[large_idx]
            sa, la = area(small['bbox']), area(large['bbox'])
            if sa <= 0:
                continue

            area_ratio = la / sa
            small_score = float(small['score'])
            large_score = float(large['score'])
            score_ratio = small_score / max(large_score, 1e-6)
            score_gap = small_score - large_score

            # VERY conservative trigger:
            # - same-ish region (IoU >= .40)
            # - larger box at least 20% larger
            # - tight box DINO confidence is at least 1.5x larger
            # - and absolute confidence gap is at least .15
            if (
                area_ratio >= 1.20
                and score_ratio >= 1.50
                and score_gap >= 0.15
            ):
                suppressed.add(large_idx)
                reasons.append({
                    'suppressed': large['candidate_id'],
                    'kept': small['candidate_id'],
                    'iou': round(ov, 3),
                    'area_ratio': round(area_ratio, 3),
                    'small_score': round(small_score, 6),
                    'large_score': round(large_score, 6),
                    'score_ratio': round(score_ratio, 3),
                    'score_gap': round(score_gap, 3),
                })

    kept = [x for idx, x in enumerate(candidates) if idx not in suppressed]
    return kept, reasons


# ============================================================
# Relation evidence
# Geometry scores are hints. Depth relation is stronger evidence.
# ============================================================

def relation_score(target, ref, relation, allow_depth=False):
    tb, rb = target['bbox'], ref['bbox']
    tcx, tcy = center(tb)
    rcx, rcy = center(rb)
    hov, vov = horizontal_overlap(tb, rb), vertical_overlap(tb, rb)
    dx, dy = abs(tcx - rcx) / W, abs(tcy - rcy) / H
    near_y = max(0.0, 1.0 - dy / 0.35)
    near_center = max(0.0, 1.0 - math.hypot(dx, dy) / 0.45)
    inside_ref = rb[0] <= tcx <= rb[2] and rb[1] <= tcy <= rb[3]
    relation = relation.lower()

    details = {
        'horizontal_overlap': round(hov, 3),
        'vertical_overlap': round(vov, 3),
    }

    if relation in {'above', 'over'}:
        score = 0.50 * float(tcy < rcy) + 0.30 * hov + 0.20 * near_y
    elif relation in {'below', 'under'}:
        score = 0.50 * float(tcy > rcy) + 0.30 * hov + 0.20 * near_y
    elif relation == 'beside':
        score = 0.55 * vov + 0.45 * near_center
    elif relation == 'on':
        score = 0.60 * float(inside_ref) + 0.20 * hov + 0.20 * vov
        details['target_center_inside_reference'] = bool(inside_ref)
    elif relation == 'left of':
        score = 0.65 * float(tcx < rcx) + 0.35 * near_center
    elif relation == 'right of':
        score = 0.65 * float(tcx > rcx) + 0.35 * near_center
    elif relation in {'in front of', 'behind'} and allow_depth:
        td, rd = depth_stats(tb), depth_stats(rb)
        tdepth, rdepth = td['median_mm'], rd['median_mm']
        depth_ok = False
        depth_delta = None
        if tdepth is not None and rdepth is not None:
            depth_delta = rdepth - tdepth
            depth_ok = (tdepth < rdepth) if relation == 'in front of' else (tdepth > rdepth)
        score = 0.65 * float(depth_ok) + 0.20 * max(hov, vov) + 0.15 * near_center
        details.update({
            'target_depth_mm': tdepth,
            'reference_depth_mm': rdepth,
            'depth_delta_mm': depth_delta,
            'depth_relation_ok': bool(depth_ok),
        })
    else:
        score = near_center

    return float(score), details


def score_constraints(target, references, routing):
    if not references:
        return 1.0, []

    out = []
    for ref_info in references:
        relation = ref_info['relation']
        best = None
        for ref in ref_info['candidates'][:MAX_REFS]:
            rs, details = relation_score(target, ref, relation, routing['depth'])
            pair_score = 0.85 * rs + 0.15 * float(ref['score'])
            item = {
                'reference_id': ref['candidate_id'],
                'reference_detector_score': ref['score'],
                'relation_score': round(rs, 4),
                'pair_score': round(pair_score, 4),
                'details': details,
            }
            if best is None or item['pair_score'] > best['pair_score']:
                best = item
        out.append({
            'relation': relation,
            'reference_object': ref_info['reference_object'],
            'best_pair': best,
        })

    vals = [x['best_pair']['pair_score'] for x in out if x['best_pair'] is not None]
    return (min(vals) if vals else 0.0), out


def prepare_targets(item, routing):
    count = int(item['parsed'].get('count', 1) or 1)
    targets, suppressed = tightness_filter(item['target']['candidates'], count)
    scored = []

    for c in targets:
        c = dict(c)
        rel, info = score_constraints(c, item.get('references', []), routing)
        c['relation_score'] = round(rel, 4)
        c['relation_info'] = info

        # Hybrid priority is only for shortlist ordering; it is NOT a final rule.
        if not item.get('references'):
            priority = float(c['score'])
        elif routing['depth']:
            # Depth relations are more objective, so relation gets stronger weight.
            priority = 0.40 * float(c['score']) + 0.60 * rel
        else:
            # Ordinary geometry is only a hint. Preserve detector/semantic candidates.
            priority = 0.75 * float(c['score']) + 0.25 * rel

        c['hybrid_priority'] = round(priority, 4)
        if routing['depth']:
            c['depth'] = depth_stats(c['bbox'])
        scored.append(c)

    return scored, suppressed


def annotate_ordinal(candidates, ordinal):
    ordinal = str(ordinal).lower()
    for c in candidates:
        c['ordinal_rank'] = None

    if ordinal == 'none' or not candidates:
        return candidates

    if ordinal in {'leftmost', 'first from left'}:
        ordered = sorted(candidates, key=lambda c: center(c['bbox'])[0])
    elif ordinal in {'rightmost', 'first from right'}:
        ordered = sorted(candidates, key=lambda c: center(c['bbox'])[0], reverse=True)
    elif ordinal == 'topmost':
        ordered = sorted(candidates, key=lambda c: center(c['bbox'])[1])
    elif ordinal == 'bottommost':
        ordered = sorted(candidates, key=lambda c: center(c['bbox'])[1], reverse=True)
    else:
        ordered = candidates

    for i, c in enumerate(ordered):
        c['ordinal_rank'] = i
    return candidates


def make_shortlist(candidates, routing, ordinal):
    # IMPORTANT: no geometry hard filtering here.
    # First keep detector/semantic diversity; depth-routed queries may use hybrid priority.
    candidates = annotate_ordinal(candidates, ordinal)

    if routing['depth']:
        ordered = sorted(candidates, key=lambda c: c['hybrid_priority'], reverse=True)
    else:
        # Mostly preserve DINO confidence. Geometry remains visible to Qwen as a hint.
        ordered = sorted(candidates, key=lambda c: float(c['score']), reverse=True)

    shortlist = ordered[:MAX_SHORTLIST]

    # If ordinal exists, ensure the extreme candidate is not accidentally dropped.
    ord_candidates = [c for c in candidates if c.get('ordinal_rank') == 0]
    if ord_candidates and all(c['candidate_id'] != ord_candidates[0]['candidate_id'] for c in shortlist):
        shortlist[-1] = ord_candidates[0]

    return shortlist


def build_groups(targets, count, item, routing):
    if count <= 1 or count > 3:
        return []

    base = sorted(targets, key=lambda x: float(x['score']), reverse=True)[:MAX_GROUP_BASE]
    groups = []

    for combo in itertools.combinations(base, count):
        if any(iou(a['bbox'], b['bbox']) > 0.35 for a, b in itertools.combinations(combo, 2)):
            continue

        box = union_boxes([x['bbox'] for x in combo])
        det_score = sum(float(x['score']) for x in combo) / count
        fake = {'bbox': box, 'score': det_score}
        rel, info = score_constraints(fake, item.get('references', []), routing)

        if not item.get('references'):
            priority = det_score
        elif routing['depth']:
            priority = 0.40 * det_score + 0.60 * rel
        else:
            priority = 0.75 * det_score + 0.25 * rel

        g = {
            'candidate_id': '',
            'members': [x['candidate_id'] for x in combo],
            'bbox': [round(v, 2) for v in box],
            'score': round(det_score, 4),
            'relation_score': round(rel, 4),
            'hybrid_priority': round(priority, 4),
            'relation_info': info,
            'ordinal_rank': None,
        }
        if routing['depth']:
            g['depth'] = depth_stats(box)
        groups.append(g)

    # Keep detector confidence as the default ordering for non-depth group queries.
    groups.sort(key=lambda x: x['hybrid_priority'] if routing['depth'] else x['score'], reverse=True)
    groups = groups[:MAX_SHORTLIST]
    for i, g in enumerate(groups):
        g['candidate_id'] = f'G{i}'
    return groups


# ============================================================
# Visualization
# ============================================================

def draw_overlay(candidates, path, color='red'):
    img = rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for c in candidates:
        x1, y1, x2, y2 = c['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.rectangle([x1, max(0, y1 - 24), x1 + 70, y1], fill='white')
        draw.text((x1 + 2, max(0, y1 - 21)), c['candidate_id'], fill='black')
    img.save(path, quality=95)


def draw_reference_overlay(references, path):
    img = rgb_pil.copy()
    draw = ImageDraw.Draw(img)
    for ref_info in references:
        for ref in ref_info.get('candidates', [])[:MAX_REFS]:
            x1, y1, x2, y2 = ref['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='green', width=4)
            draw.rectangle([x1, max(0, y1 - 22), x1 + 85, y1], fill='white')
            draw.text((x1 + 2, max(0, y1 - 19)), ref['candidate_id'], fill='green')
    img.save(path, quality=95)


def make_crop_sheet(candidates, path):
    if not candidates:
        return None

    tw, th = 240, 180
    sheet = Image.new('RGB', (tw * len(candidates), th), 'white')

    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = map(int, c['bbox'])
        crop = rgb_pil.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
        crop.thumbnail((tw - 16, th - 35))
        tile = Image.new('RGB', (tw, th), 'white')
        tile.paste(crop, ((tw - crop.width) // 2, 30))
        ImageDraw.Draw(tile).text((8, 8), c['candidate_id'], fill='black')
        sheet.paste(tile, (i * tw, 0))

    sheet.save(path, quality=95)
    return path


print('Loading Qwen3-VL:', QWEN_PATH)
processor = AutoProcessor.from_pretrained(QWEN_PATH, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    local_files_only=True,
)
model.eval()


def parse_json(text):
    s, e = text.find('{'), text.rfind('}')
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            pass
    return None


def qwen_judge(qid, item, routing, candidates):
    overlay = OUTPUT_DIR / f'{qid}_shortlist.jpg'
    crops = OUTPUT_DIR / f'{qid}_crops.jpg'
    refs_overlay = OUTPUT_DIR / f'{qid}_references.jpg'

    draw_overlay(candidates, overlay)
    make_crop_sheet(candidates, crops)

    content = [
        {'type': 'image', 'image': str(RGB_PATH)},
        {'type': 'image', 'image': str(overlay)},
        {'type': 'image', 'image': str(crops)},
    ]

    if item.get('references'):
        draw_reference_overlay(item['references'], refs_overlay)
        content.append({'type': 'image', 'image': str(refs_overlay)})

    if routing['depth']:
        content.append({'type': 'image', 'image': str(depth_vis_path)})

    if routing['infrared']:
        content.append({'type': 'image', 'image': str(IR_PATH)})

    evidence = []
    for c in candidates:
        e = {
            'candidate_id': c['candidate_id'],
            'bbox': c['bbox'],
            'detector_score': c.get('score'),
            'relation_score_hint': c.get('relation_score'),
            'hybrid_priority_for_shortlist_only': c.get('hybrid_priority'),
            'ordinal_rank_hint': c.get('ordinal_rank'),
            'members': c.get('members'),
            'relation_info': c.get('relation_info'),
        }
        if routing['depth']:
            e['depth'] = c.get('depth')
        evidence.append(e)

    allowed = [k for k, v in routing.items() if v]
    prompt = f'''You are the FINAL visual grounding candidate judge.

QUERY: {item['query']}
Structured parse: {json.dumps(item['parsed'], ensure_ascii=False)}
ALLOWED EVIDENCE: {allowed}
Candidates: {json.dumps(evidence, ensure_ascii=False)}

Image order:
1) original RGB image
2) target candidates (red boxes)
3) target candidate crops
4) reference candidates (green boxes), if the query has reference objects
5) depth visualization, only for depth-routed queries
6) infrared image, only for infrared-routed queries

Rules:
- Qwen remains the final semantic judge. Do NOT blindly follow a numeric score.
- RGB is primary for identity, color, clothing, texture, material and semantic appearance.
- Geometry scores are SOFT HINTS only because reference detection may be noisy.
- For relation queries, jointly inspect the target and the correct reference object in the images.
- Depth may be used ONLY if depth is allowed. Smaller mm means closer to the camera.
- For in-front-of / behind / nearest / farthest queries, depth evidence is important.
- Infrared may be used ONLY if infrared is allowed.
- Never invent an irrelevant reason. If depth is not allowed, do not use depth.
- ordinal_rank is only a hint among semantically valid / relation-valid targets; do not choose a wrong object just because it is geometrically extreme.
- For count=1 prefer a tight box around one requested instance, but do not reject a valid candidate merely because another box is smaller.
- For count>1 choose a G* group candidate covering all requested objects.
- Select ONLY an existing candidate ID. Do not generate coordinates.

Return ONLY JSON:
{{"selected_candidate_id":"T0 or G0 or NONE","used_evidence":["RGB"],"reason":"brief grounded reason"}}'''

    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt'
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=180, do_sample=False)

    n = inputs.input_ids.shape[1]
    raw = processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
    return parse_json(raw), raw


results = {}
for qid, item in data.items():
    print('\n' + '=' * 90)
    print(qid, item['query'])

    routing = get_routing(item['parsed'], item['query'])
    print('ROUTING:', routing)

    targets, suppressed = prepare_targets(item, routing)
    print('SUPPRESSED:', suppressed)

    count = int(item['parsed'].get('count', 1) or 1)
    if count == 1:
        shortlist = make_shortlist(
            targets,
            routing,
            item['parsed'].get('ordinal', 'none'),
        )
    else:
        shortlist = build_groups(targets, count, item, routing)

    print('SHORTLIST:')
    for c in shortlist:
        print(
            c['candidate_id'],
            'det=', c.get('score'),
            'rel_hint=', c.get('relation_score'),
            'priority=', c.get('hybrid_priority'),
            'ordinal=', c.get('ordinal_rank'),
            'bbox=', c['bbox'],
        )

    decision, raw = qwen_judge(qid, item, routing, shortlist)
    if decision is None:
        decision = {
            'selected_candidate_id': 'NONE',
            'used_evidence': [],
            'reason': 'Qwen JSON parse failure',
        }

    selected_id = decision.get('selected_candidate_id', 'NONE')
    cmap = {c['candidate_id']: c for c in shortlist}
    chosen = cmap.get(selected_id)
    bbox = chosen['bbox'] if chosen else None
    norm = [
        round(bbox[0] / W, 6), round(bbox[1] / H, 6),
        round(bbox[2] / W, 6), round(bbox[3] / H, 6),
    ] if bbox else None

    print('DECISION:', decision)
    print('BBOX:', bbox)

    results[qid] = {
        'query': item['query'],
        'parsed': item['parsed'],
        'routing': routing,
        'suppressed': suppressed,
        'shortlist': shortlist,
        'decision': decision,
        'bbox': bbox,
        'bbox_normalized': norm,
        'qwen_raw': raw,
    }

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('\n' + '=' * 90)
print('STAGE 2.2 COMPLETE')
print(OUTPUT_JSON)
print('Peak CUDA allocated:', round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), 'GB')
