from src.utils.face_detection.compute_iou import compute_iou

def evaluate_image(pred_boxes, gt_boxes, iou_threshold=0.5):

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:

        match_found = False

        for i, gt in enumerate(gt_boxes):

            if i in matched_gt:
                continue

            iou = compute_iou(pred, gt)

            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break

        if not match_found:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn
