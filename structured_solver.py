import cv2
import numpy as np
import argparse
import math
import os

def extract_and_classify_pieces(scattered_path):
    print(f"Loading scattered image: {scattered_path}")
    img = cv2.imread(scattered_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: Could not load scattered image.")
        return None, [], [], []
        
    if img.shape[2] == 4:
        # Use alpha channel if available
        alpha = img[:, :, 3]
        _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        # Fallback to background color thresholding
        img_bgr = img[:, :, :3]
        bg_color = img_bgr[0, 0]
        lower_bound = np.clip(bg_color - 5, 0, 255)
        upper_bound = np.clip(bg_color + 5, 0, 255)
        bg_mask = cv2.inRange(img_bgr, lower_bound, upper_bound)
        thresh = cv2.bitwise_not(bg_mask)
        
        # Add alpha channel for later use
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = 255

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corners = []
    edges = []
    interiors = []
    
    piece_id = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        
        piece_mask_full = np.zeros_like(thresh)
        cv2.drawContours(piece_mask_full, [cnt], -1, 255, -1)
        
        piece_img = img[y1:y2, x1:x2].copy()
        piece_mask = piece_mask_full[y1:y2, x1:x2].copy()
        piece_img[:, :, 3] = piece_mask
        
        # --- Classification Logic using Polygon Approximation ---
        shifted_cnt = cnt - [x1, y1]
        
        # Calculate bounding box of the actual contour to get a more accurate width/height
        # rather than using the padded bounding box
        cx, cy, cw, ch = cv2.boundingRect(shifted_cnt)
        
        # INCREASE EPSILON to smooth out the new pronounced curves
        # so they don't get broken down into many small straight lines
        epsilon = 0.01 * cv2.arcLength(shifted_cnt, True)
        approx = cv2.approxPolyDP(shifted_cnt, epsilon, True)
        
        straight_edges = []
        
        for j in range(len(approx)):
            p1 = approx[j][0]
            p2 = approx[(j+1)%len(approx)][0]
            
            length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            # Use the contour's bounding box for threshold
            if length > min(cw, ch) * 0.50:
                angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
                
                is_new = True
                for a in straight_edges:
                    diff = min(abs(angle - a), 180 - abs(angle - a))
                    if diff < 20:
                        is_new = False
                        break
                if is_new:
                    straight_edges.append(angle)
                    
        # Debug: Save classification info
        piece_data = {
            'id': piece_id,
            'image': piece_img,
            'mask': piece_mask,
            'straight_edges': straight_edges
        }
        
        # Save the extracted piece image to disk
        os.makedirs("extracted_pieces", exist_ok=True)
        cv2.imwrite(f"extracted_pieces/piece_{piece_id}.png", piece_img)
        
        # 按照数脖子分类 (Classify by counting necks)
        # 4条边减去直线边的数量，就是脖子的数量
        num_necks = 4 - len(straight_edges)
        
        if num_necks <= 2: # 2个脖子是corner (如果因为误差识别出0或1个脖子，也归为corner)
            corners.append(piece_data)
            print(f"Piece {piece_id} -> CORNER ({num_necks} necks)")
        elif num_necks == 3: # 3个脖子是edge
            edges.append(piece_data)
            print(f"Piece {piece_id} -> EDGE ({num_necks} necks)")
        else: # 4个脖子是INTERIOR
            interiors.append(piece_data)
            print(f"Piece {piece_id} -> INTERIOR ({num_necks} necks)")
            
        piece_id += 1
        
    print(f"Extracted {piece_id} pieces.")
    print(f"Classified: {len(corners)} Corners, {len(edges)} Edges, {len(interiors)} Interiors.")
    return img, corners, edges, interiors

def get_geometric_rotation(piece, piece_type, target_pos):
    """
    piece: piece dictionary
    piece_type: 'corner' or 'edge'
    target_pos: 
      for corner: 0 (TL), 1 (TR), 2 (BL), 3 (BR)
      for edge: 0 (Top), 1 (Bottom), 2 (Left), 3 (Right)
    Returns the base angle to rotate the piece so it fits the target position.
    """
    img = piece['image']
    mask = piece['mask']
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    
    M = cv2.moments(cnt)
    if M['m00'] == 0: return 0
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    x, y, cw, ch = cv2.boundingRect(cnt)
    
    lines = []
    max_dim = max(cw, ch)
    threshold = max_dim * 0.46
    
    for j in range(len(approx)):
        p1 = approx[j][0]
        p2 = approx[(j+1)%len(approx)][0]
        length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if length > threshold:
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
            
            is_new = True
            for l in lines:
                a = l['angle']
                diff = min(abs(angle - a), 180 - abs(angle - a))
                if diff < 20:
                    is_new = False
                    break
            if is_new:
                mx = (p1[0] + p2[0]) / 2.0
                my = (p1[1] + p2[1]) / 2.0
                lines.append({'angle': angle, 'mx': mx, 'my': my, 'p1': p1, 'p2': p2})
                
    if piece_type == 'corner' and len(lines) >= 2:
        l1 = lines[0]
        l2 = lines[1]
        
        p1_1 = np.array([l1['p1'][0], l1['p1'][1], 1])
        p1_2 = np.array([l1['p2'][0], l1['p2'][1], 1])
        L1 = np.cross(p1_1, p1_2)
        
        p2_1 = np.array([l2['p1'][0], l2['p1'][1], 1])
        p2_2 = np.array([l2['p2'][0], l2['p2'][1], 1])
        L2 = np.cross(p2_1, p2_2)
        
        pt = np.cross(L1, L2)
        if pt[2] != 0:
            ix = pt[0] / pt[2]
            iy = pt[1] / pt[2]
            
            dx = cx - ix
            dy = cy - iy
            angle_V = math.degrees(math.atan2(dy, dx))
            
            target_angles = {
                0: 45,   # TL
                1: 135,  # TR
                2: -45,  # BL
                3: -135  # BR
            }
            target_angle = target_angles[target_pos]
            base_edge_angle = lines[0]['angle']
            
            best_rot = 0
            best_diff = float('inf')
            
            for rot_offset in [0, 90, 180, 270]:
                rot = base_edge_angle - rot_offset
                new_angle_V = angle_V - rot
                diff = (new_angle_V - target_angle + 180) % 360 - 180
                if abs(diff) < best_diff:
                    best_diff = abs(diff)
                    best_rot = rot
                    
            return best_rot
            
    elif piece_type == 'edge' and len(lines) >= 1:
        l1 = lines[0]
        mx = l1['mx']
        my = l1['my']
        
        dx = cx - mx
        dy = cy - my
        angle_V = math.degrees(math.atan2(dy, dx))
        
        target_angles = {
            0: 90,   # Top edge, piece extends down
            1: -90,  # Bottom edge, piece extends up
            2: 0,    # Left edge, piece extends right
            3: 180   # Right edge, piece extends left
        }
        target_angle = target_angles[target_pos]
        base_edge_angle = lines[0]['angle']
        
        best_rot = 0
        best_diff = float('inf')
        
        for rot_offset in [0, 90, 180, 270]:
            rot = base_edge_angle - rot_offset
            new_angle_V = angle_V - rot
            diff = (new_angle_V - target_angle + 180) % 360 - 180
            if abs(diff) < best_diff:
                best_diff = abs(diff)
                best_rot = rot
                
        return best_rot
        
    return 0

def align_and_match(piece, orig_bgr, piece_type='interior', used_positions=None):
    """
    Aligns the piece based on its straight edges (if any) and matches it to the original image.
    piece_type: 'corner', 'edge', or 'interior' to restrict the search area structurally.
    """
    if used_positions is None:
        used_positions = []
        
    piece_img = piece['image']
    piece_mask = piece['mask']
    straight_edges = piece['straight_edges']
    
    best_val = float('inf')
    best_M = None
    best_pos_idx = -1
    
    # Determine angles to test based on classification
    angles_to_test = []
    if len(straight_edges) >= 1:
        # It's a corner or edge. We align the first straight edge to 0, 90, 180, 270.
        base_angle = straight_edges[0]
        for rot in [0, 90, 180, 270]:
            test_angle = base_angle - rot
            angles_to_test.append(test_angle)
    else:
        angles_to_test = range(0, 360, 5)
        
    scale = 0.25
    orig_bgr_small = cv2.resize(orig_bgr, (0,0), fx=scale, fy=scale)
    piece_img_small = cv2.resize(piece_img, (0,0), fx=scale, fy=scale)
    piece_mask_small = cv2.resize(piece_mask, (0,0), fx=scale, fy=scale)
    
    # Coarse search
    best_coarse_val = float('inf')
    best_coarse_angle = 0
    best_coarse_pos = -1
    
    orig_h_s, orig_w_s = orig_bgr_small.shape[:2]
    
    for angle in angles_to_test:
        h, w = piece_img_small.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M_rot[0, 0])
        sin = np.abs(M_rot[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M_rot[0, 2] += (new_w / 2.0) - center[0]
        M_rot[1, 2] += (new_h / 2.0) - center[1]
        
        rotated_piece = cv2.warpAffine(piece_img_small, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        rotated_mask = cv2.warpAffine(piece_mask_small, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
        if w_crop == 0 or h_crop == 0: continue
            
        cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
        cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
        
        template = cropped_piece[:, :, :3]
        mask_8u = cropped_mask.astype(np.uint8)
        
        # Erode the mask to ignore the black jigsaw borders during matching
        kernel = np.ones((5,5), np.uint8)
        mask_8u_eroded = cv2.erode(mask_8u, kernel, iterations=2)
        
        if cv2.countNonZero(mask_8u_eroded) < 10: continue
        
        if piece_type == 'corner':
            # Test 4 corners with margin
            margin = 20
            positions = [
                (0, 0),
                (orig_w_s - w_crop - margin, 0),
                (0, orig_h_s - h_crop - margin),
                (orig_w_s - w_crop - margin, orig_h_s - h_crop - margin)
            ]
            for i, (px, py) in enumerate(positions):
                if i in used_positions: continue
                px = max(0, px)
                py = max(0, py)
                target_roi = orig_bgr_small[py:min(orig_h_s, py+h_crop+margin), px:min(orig_w_s, px+w_crop+margin)]
                if target_roi.shape[0] < template.shape[0] or target_roi.shape[1] < template.shape[1]: continue
                
                res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
                min_val, _, _, _ = cv2.minMaxLoc(res)
                if min_val < best_coarse_val:
                    best_coarse_val = min_val
                    best_coarse_angle = angle
                    best_coarse_pos = i
                    
        elif piece_type == 'edge':
            # Top: slide horizontally along y=0 strip; Bottom: along bottom strip;
            # Left: slide vertically along x=0 strip; Right: along right strip.
            # ROI must be wider/taller than the template so matchTemplate can slide.
            edge_rois = [
                (0,                      0,                       orig_w_s, h_crop),  # Top
                (0,                      orig_h_s - h_crop,       orig_w_s, h_crop),  # Bottom
                (0,                      0,                       w_crop,   orig_h_s), # Left
                (max(0, orig_w_s-w_crop),0,                       w_crop,   orig_h_s), # Right
            ]
            for i, (rx, ry, rw, rh) in enumerate(edge_rois):
                if rx < 0 or ry < 0 or rx+rw > orig_w_s or ry+rh > orig_h_s: continue
                target_roi = orig_bgr_small[ry:ry+rh, rx:rx+rw]
                if target_roi.shape[0] < template.shape[0] or target_roi.shape[1] < template.shape[1]: continue

                res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
                min_val, _, _, _ = cv2.minMaxLoc(res)
                if min_val < best_coarse_val:
                    best_coarse_val = min_val
                    best_coarse_angle = angle
                    best_coarse_pos = i
        else:
            # Interior: global search
            res = cv2.matchTemplate(orig_bgr_small, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < best_coarse_val:
                best_coarse_val = min_val
                best_coarse_angle = angle
            
    # Fine search
    fine_angles = range(int(best_coarse_angle) - 3, int(best_coarse_angle) + 4)
    orig_h, orig_w = orig_bgr.shape[:2]
    
    for angle in fine_angles:
        h, w = piece_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M_rot[0, 0])
        sin = np.abs(M_rot[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M_rot[0, 2] += (new_w / 2.0) - center[0]
        M_rot[1, 2] += (new_h / 2.0) - center[1]
        
        rotated_piece = cv2.warpAffine(piece_img, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        rotated_mask = cv2.warpAffine(piece_mask, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
        if w_crop == 0 or h_crop == 0: continue
            
        cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
        cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
        
        template = cropped_piece[:, :, :3]
        mask_8u = cropped_mask.astype(np.uint8)
        
        # Erode the mask to ignore the black jigsaw borders during matching
        kernel = np.ones((5,5), np.uint8)
        mask_8u_eroded = cv2.erode(mask_8u, kernel, iterations=2)
        
        if cv2.countNonZero(mask_8u_eroded) < 100: continue
            
        if piece_type == 'corner':
            pos_idx = best_coarse_pos
            
            # For corners, we don't need fine angle search because geometric rotation is exact
            if angle != int(best_coarse_angle):
                continue
                
            # Force exact alignment to the corners!
            if pos_idx == 0: # TL
                tx, ty = 0, 0
                px, py = 0, 0
            elif pos_idx == 1: # TR
                tx, ty = 0, 0
                px = orig_w - w_crop
                py = 0
            elif pos_idx == 2: # BL
                tx, ty = 0, 0
                px = 0
                py = orig_h - h_crop
            elif pos_idx == 3: # BR
                tx, ty = 0, 0
                px = orig_w - w_crop
                py = orig_h - h_crop
                
            best_val = 0 # Perfect match by definition
            M_full = M_rot.copy()
            M_full[0, 2] += px - x
            M_full[1, 2] += py - y
            best_M = M_full
            best_pos_idx = pos_idx
                    
        elif piece_type == 'edge':
            pos_idx = best_coarse_pos

            if pos_idx == 0: # Top
                roi_y1, roi_y2 = 0, h_crop
                roi_x1, roi_x2 = 0, orig_w
                py_fixed = 0
            elif pos_idx == 1: # Bottom
                roi_y1, roi_y2 = max(0, orig_h - h_crop), orig_h
                roi_x1, roi_x2 = 0, orig_w
                py_fixed = orig_h - h_crop
            elif pos_idx == 2: # Left
                roi_y1, roi_y2 = 0, orig_h
                roi_x1, roi_x2 = 0, w_crop
                px_fixed = 0
            elif pos_idx == 3: # Right
                roi_y1, roi_y2 = 0, orig_h
                roi_x1, roi_x2 = max(0, orig_w - w_crop), orig_w
                px_fixed = orig_w - w_crop

            target_roi = orig_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            if target_roi.shape[0] >= template.shape[0] and target_roi.shape[1] >= template.shape[1]:
                res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                if min_val < best_val:
                    best_val = min_val
                    tx, ty = min_loc
                    M_full = M_rot.copy()
                    if pos_idx == 0: # Top
                        M_full[0, 2] += (roi_x1 + tx) - x
                        M_full[1, 2] += py_fixed - y
                    elif pos_idx == 1: # Bottom
                        M_full[0, 2] += (roi_x1 + tx) - x
                        M_full[1, 2] += py_fixed - y
                    elif pos_idx == 2: # Left
                        M_full[0, 2] += px_fixed - x
                        M_full[1, 2] += (roi_y1 + ty) - y
                    elif pos_idx == 3: # Right
                        M_full[0, 2] += px_fixed - x
                        M_full[1, 2] += (roi_y1 + ty) - y
                    best_M = M_full
                    best_pos_idx = pos_idx
        else:
            res = cv2.matchTemplate(orig_bgr, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            if min_val < best_val:
                best_val = min_val
                tx, ty = min_loc
                M_full = M_rot.copy()
                M_full[0, 2] += tx - x
                M_full[1, 2] += ty - y
                best_M = M_full
                
    return best_M, best_val, best_pos_idx

def solve_structured(scattered_path, original_path, output_path):
    orig_img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    if orig_img is None:
        print("Error: Could not load original image.")
        return
    if orig_img.shape[2] == 3:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
        
    orig_bgr = orig_img[:, :, :3]
    orig_h, orig_w = orig_img.shape[:2]
    
    _, corners, edges, interiors = extract_and_classify_pieces(scattered_path)
    
    canvas = np.zeros_like(orig_img)
    output_prefix = output_path.replace('.png', '')
    step = 1

    # --- 1. Place Corners Sequentially ---
    print("\n--- Matching Corners Sequentially (TL -> TR -> BL -> BR) ---")
    placed_corners = set()
    corner_positions = [(0, 'TL'), (1, 'TR'), (2, 'BL'), (3, 'BR')]
    
    frontiers = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for pos_idx, pos_name in corner_positions:
        print(f"Scanning original image for {pos_name} corner...")
        best_piece_idx = -1
        best_val = float('inf')
        best_M = None
        best_px = 0
        best_py = 0
        best_w_crop = 0
        best_h_crop = 0
        
        for idx, piece in enumerate(corners):
            if idx in placed_corners: continue
            
            rot = get_geometric_rotation(piece, 'corner', pos_idx)
            
            piece_img = piece['image']
            piece_mask = piece['mask']
            h, w = piece_img.shape[:2]
            center = (w / 2.0, h / 2.0)
            
            M_rot = cv2.getRotationMatrix2D(center, rot, 1.0)
            cos = np.abs(M_rot[0, 0])
            sin = np.abs(M_rot[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M_rot[0, 2] += (new_w / 2.0) - center[0]
            M_rot[1, 2] += (new_h / 2.0) - center[1]
            
            rotated_piece = cv2.warpAffine(piece_img, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            rotated_mask = cv2.warpAffine(piece_mask, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
            if w_crop == 0 or h_crop == 0: continue
            
            cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
            cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
            template = cropped_piece[:, :, :3]
            mask_8u = cropped_mask.astype(np.uint8)
            kernel = np.ones((5,5), np.uint8)
            mask_8u_eroded = cv2.erode(mask_8u, kernel, iterations=2)
            
            if cv2.countNonZero(mask_8u_eroded) < 100: continue
            
            if pos_idx == 0: # TL
                px, py = 0, 0
            elif pos_idx == 1: # TR
                px = orig_w - w_crop
                py = 0
            elif pos_idx == 2: # BL
                px = 0
                py = orig_h - h_crop
            elif pos_idx == 3: # BR
                px = orig_w - w_crop
                py = orig_h - h_crop
                
            target_roi = orig_bgr[py:py+h_crop, px:px+w_crop]
            if target_roi.shape[0] < template.shape[0] or target_roi.shape[1] < template.shape[1]: continue
            
            res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            
            if min_val < best_val:
                best_val = min_val
                M_full = M_rot.copy()
                M_full[0, 2] += px - x
                M_full[1, 2] += py - y
                best_M = M_full
                best_piece_idx = idx
                best_px = px
                best_py = py
                best_w_crop = w_crop
                best_h_crop = h_crop
                
        if best_piece_idx != -1:
            placed_corners.add(best_piece_idx)
            print(f"  -> Rotated and compared pieces. Piece {corners[best_piece_idx]['id']} matches best! (diff {best_val:.4f})")
            
            piece = corners[best_piece_idx]
            warped = cv2.warpAffine(piece['image'], best_M, (canvas.shape[1], canvas.shape[0]))
            warped_mask = cv2.warpAffine(piece['mask'], best_M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)
            mask = warped_mask > 128
            np.copyto(canvas, warped, where=mask[:,:,None])
            
            cv2.imwrite(f"{output_prefix}_step_{step:02d}_corner_{pos_name}.png", canvas)
            step += 1
            
            # Initialize frontiers
            if pos_idx == 0: # TL
                frontiers[0] = best_px + best_w_crop - 30
                frontiers[2] = best_py + best_h_crop - 30
            elif pos_idx == 1: # TR
                frontiers[3] = best_py + best_h_crop - 30
            elif pos_idx == 2: # BL
                frontiers[1] = best_px + best_w_crop - 30

    # --- 2. Place Edges Sequentially ---
    print("\n--- Placing Edge pieces sequentially ---")

    edge_names = {0: 'Top', 1: 'Bottom', 2: 'Left', 3: 'Right'}

    def best_edge_match_for_piece(piece, pos_idx):
        best_val = float('inf')
        best_M = None
        best_coord = -1
        
        base_angle = get_geometric_rotation(piece, 'edge', pos_idx)
        
        piece_img = piece['image']
        piece_mask = piece['mask']
        
        for angle in [base_angle]:
            h, w = piece_img.shape[:2]
            center = (w / 2.0, h / 2.0)
            
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos_v = np.abs(M_rot[0, 0])
            sin_v = np.abs(M_rot[0, 1])
            new_w = int((h * sin_v) + (w * cos_v))
            new_h = int((h * cos_v) + (w * sin_v))
            M_rot[0, 2] += (new_w / 2.0) - center[0]
            M_rot[1, 2] += (new_h / 2.0) - center[1]
            
            rotated_piece = cv2.warpAffine(piece_img, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            rotated_mask = cv2.warpAffine(piece_mask, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
            if w_crop == 0 or h_crop == 0: continue
            
            cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
            cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
            template = cropped_piece[:, :, :3]
            mask_8u = cropped_mask.astype(np.uint8)
            kernel = np.ones((5,5), np.uint8)
            mask_8u_eroded = cv2.erode(mask_8u, kernel, iterations=2)
            
            if cv2.countNonZero(mask_8u_eroded) < 100: continue
            
            if pos_idx == 0: # Top
                target_roi = orig_bgr[0:h_crop, 0:orig_w]
                py = 0
                px_offset = 0
            elif pos_idx == 1: # Bottom
                target_roi = orig_bgr[orig_h - h_crop:orig_h, 0:orig_w]
                py = orig_h - h_crop
                px_offset = 0
            elif pos_idx == 2: # Left
                target_roi = orig_bgr[0:orig_h, 0:w_crop]
                px_offset = 0
                py = 0
            elif pos_idx == 3: # Right
                target_roi = orig_bgr[0:orig_h, orig_w - w_crop:orig_w]
                px_offset = orig_w - w_crop
                py = 0
                
            if target_roi.shape[0] < template.shape[0] or target_roi.shape[1] < template.shape[1]: continue
            
            res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            
            if min_val < best_val:
                best_val = min_val
                tx, ty = min_loc
                M_full = M_rot.copy()
                
                global_tx = px_offset + tx
                global_ty = py + ty
                M_full[0, 2] += global_tx - x
                M_full[1, 2] += global_ty - y
                
                if pos_idx in [0, 1]:
                    best_coord = global_tx
                else:
                    best_coord = global_ty
                    
                best_M = M_full
                
        return best_val, best_M, best_coord

    # Pre-assign pieces to edges based on minimum diff
    edge_assignments = {0: [], 1: [], 2: [], 3: []}
    for idx, piece in enumerate(edges):
        best_val = float('inf')
        best_pos = -1
        best_coord = -1
        best_M = None
        
        for pos_idx in [0, 1, 2, 3]:
            val, M_full, coord = best_edge_match_for_piece(piece, pos_idx)
            if val < best_val:
                best_val = val
                best_pos = pos_idx
                best_coord = coord
                best_M = M_full
                
        if best_pos != -1:
            edge_assignments[best_pos].append({
                'idx': idx,
                'piece': piece,
                'val': best_val,
                'coord': best_coord,
                'M': best_M
            })

    for pos_idx in [0, 1, 2, 3]:
        print(f"\n--- Placing {edge_names[pos_idx]} Edge ---")
        
        # Sort assigned pieces by coordinate
        assigned_pieces = edge_assignments[pos_idx]
        assigned_pieces.sort(key=lambda x: x['coord'])
        
        for match in assigned_pieces:
            piece = match['piece']
            best_M = match['M']
            
            # Check overlap with canvas
            warped_mask_test = cv2.warpAffine(piece['mask'], best_M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)
            canvas_alpha = canvas[:,:,3] if canvas.shape[2] == 4 else cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            overlap = np.logical_and(warped_mask_test > 128, canvas_alpha > 10)
            if np.sum(overlap) > 2000: 
                print(f"  -> Skipped Piece {piece['id']} due to overlap")
                continue
                
            print(f"  -> Placed Piece {piece['id']} on {edge_names[pos_idx]} at coord {match['coord']} (diff {match['val']:.4f})")
            
            warped = cv2.warpAffine(piece['image'], best_M, (canvas.shape[1], canvas.shape[0]))
            warped_mask = cv2.warpAffine(piece['mask'], best_M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)
            mask = warped_mask > 128
            np.copyto(canvas, warped, where=mask[:,:,None])
            
            cv2.imwrite(f"{output_prefix}_step_{step:02d}_edge_{edge_names[pos_idx]}.png", canvas)
            step += 1

    # --- 3. Place Interiors Globally ---
    print("\n--- Placing Interiors Globally ---")
    
    num_pieces = len(corners) + len(edges) + len(interiors)
    best_r, best_c = 1, num_pieces
    best_diff = float('inf')
    target_ratio = orig_h / orig_w
    for r in range(1, num_pieces + 1):
        if num_pieces % r == 0:
            c = num_pieces // r
            ratio = r / c
            if abs(ratio - target_ratio) < best_diff:
                best_diff = abs(ratio - target_ratio)
                best_r = r
                best_c = c
    R, C = best_r, best_c
    
    w_est = orig_w / C
    h_est = orig_h / R
    
    interior_matches = []
    
    for row in range(1, R - 1):
        for col in range(1, C - 1):
            print(f"Evaluating slot ({row}, {col})...")
            
            margin_x = int(w_est * 0.6)
            margin_y = int(h_est * 0.6)
            
            roi_x1 = max(0, int(col * w_est) - margin_x)
            roi_y1 = max(0, int(row * h_est) - margin_y)
            roi_x2 = min(orig_w, int((col+1) * w_est) + margin_x)
            roi_y2 = min(orig_h, int((row+1) * h_est) + margin_y)
            
            target_roi = orig_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            
            for idx, piece in enumerate(interiors):
                piece_img = piece['image']
                piece_mask = piece['mask']
                
                # Coarse search with scaling for speed
                best_coarse_val = float('inf')
                best_coarse_angle = 0
                
                scale = 0.25
                target_roi_small = cv2.resize(target_roi, (0,0), fx=scale, fy=scale)
                piece_img_small = cv2.resize(piece_img, (0,0), fx=scale, fy=scale)
                piece_mask_small = cv2.resize(piece_mask, (0,0), fx=scale, fy=scale)
                
                for angle in range(0, 360, 5):
                    h, w = piece_img_small.shape[:2]
                    center = (w / 2.0, h / 2.0)
                    
                    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    cos = np.abs(M_rot[0, 0])
                    sin = np.abs(M_rot[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    M_rot[0, 2] += (new_w / 2.0) - center[0]
                    M_rot[1, 2] += (new_h / 2.0) - center[1]
                    
                    rotated_mask = cv2.warpAffine(piece_mask_small, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
                    if w_crop == 0 or h_crop == 0: continue
                    
                    if target_roi_small.shape[0] < h_crop or target_roi_small.shape[1] < w_crop: continue
                    
                    rotated_piece = cv2.warpAffine(piece_img_small, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                    cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
                    cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
                    
                    template = cropped_piece[:, :, :3]
                    mask_8u = cropped_mask.astype(np.uint8)
                    kernel = np.ones((3,3), np.uint8)
                    mask_8u_eroded = cv2.erode(mask_8u, kernel, iterations=1)
                    
                    if cv2.countNonZero(mask_8u_eroded) < 10: continue
                    
                    res = cv2.matchTemplate(target_roi_small, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u_eroded)
                    min_val, _, _, _ = cv2.minMaxLoc(res)
                    
                    if min_val < best_coarse_val:
                        best_coarse_val = min_val
                        best_coarse_angle = angle
                        
                # Fine search
                best_val = float('inf')
                best_M = None
                
                for angle in np.arange(best_coarse_angle - 5.0, best_coarse_angle + 5.2, 0.2):
                    h, w = piece_img.shape[:2]
                    center = (w / 2.0, h / 2.0)
                    
                    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    cos = np.abs(M_rot[0, 0])
                    sin = np.abs(M_rot[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    M_rot[0, 2] += (new_w / 2.0) - center[0]
                    M_rot[1, 2] += (new_h / 2.0) - center[1]
                    
                    rotated_mask = cv2.warpAffine(piece_mask, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    x, y, w_crop, h_crop = cv2.boundingRect(rotated_mask)
                    if w_crop == 0 or h_crop == 0: continue
                    
                    if target_roi.shape[0] < h_crop or target_roi.shape[1] < w_crop: continue
                    
                    rotated_piece = cv2.warpAffine(piece_img, M_rot, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                    cropped_piece = rotated_piece[y:y+h_crop, x:x+w_crop]
                    cropped_mask = rotated_mask[y:y+h_crop, x:x+w_crop]
                    
                    template = cropped_piece[:, :, :3]
                    mask_8u = cropped_mask.astype(np.uint8)
                    
                    if cv2.countNonZero(mask_8u) < 100: continue
                    
                    res = cv2.matchTemplate(target_roi, template, cv2.TM_SQDIFF_NORMED, mask=mask_8u)
                    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                    
                    if min_val < best_val:
                        best_val = min_val
                        tx, ty = min_loc
                        M_full = M_rot.copy()
                        M_full[0, 2] += roi_x1 + tx - x
                        M_full[1, 2] += roi_y1 + ty - y
                        best_M = M_full
                        
                if best_M is not None and best_val < 0.3: # Keep reasonable matches
                    interior_matches.append({
                        'row': row,
                        'col': col,
                        'idx': idx,
                        'piece': piece,
                        'val': best_val,
                        'M': best_M
                    })
                    
    # Sort all matches by diff
    interior_matches.sort(key=lambda x: x['val'])
    
    placed_interiors = set()
    filled_slots = set()
    
    for match in interior_matches:
        if match['idx'] in placed_interiors: continue
        if (match['row'], match['col']) in filled_slots: continue
        
        piece = match['piece']
        best_M = match['M']
        row = match['row']
        col = match['col']
        
        # Check overlap with canvas
        warped_mask_test = cv2.warpAffine(piece['mask'], best_M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST)
        canvas_alpha = canvas[:,:,3] if canvas.shape[2] == 4 else cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        canvas_mask_bin = canvas_alpha > 10
        overlap = np.logical_and(warped_mask_test > 128, canvas_mask_bin)
        piece_area = cv2.countNonZero((warped_mask_test > 128).astype(np.uint8))
        
        if np.sum(overlap) > piece_area * 0.25:
            continue  # Skip this match if it overlaps too much with existing pieces
            
        placed_interiors.add(match['idx'])
        filled_slots.add((row, col))
        
        print(f"  -> Placed Piece {piece['id']} at ({row}, {col}) with diff {match['val']:.4f}")
        
        warped = cv2.warpAffine(piece['image'], best_M, (canvas.shape[1], canvas.shape[0]))
        mask = warped_mask_test > 128
        np.copyto(canvas, warped, where=mask[:,:,None])
        
        cv2.imwrite(f"{output_prefix}_step_{step:02d}_interior.png", canvas)
        step += 1
        
    # Report any unfilled slots
    for row in range(1, R - 1):
        for col in range(1, C - 1):
            if (row, col) not in filled_slots:
                print(f"  -> Failed to find piece for ({row}, {col})")

    cv2.imwrite(output_path, canvas)
    print(f"\nSaved complete puzzle to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scattered", help="Scattered image")
    parser.add_argument("original", help="Original image")
    parser.add_argument("-o", "--output", default="structured_assembled.png")
    args = parser.parse_args()
    solve_structured(args.scattered, args.original, args.output)
