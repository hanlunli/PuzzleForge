import math
import random
import argparse
import os
from PIL import Image, ImageDraw

def cubic_bezier(p0, p1, p2, p3, num_points=20):
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points

def get_jigsaw_edge(tab_direction=1):
    # Randomize the shape slightly for each edge to increase difficulty
    # Shoulder waves (increased amplitude for more pronounced curves)
    w1 = random.uniform(-0.15, 0.15)
    w2 = random.uniform(-0.15, 0.15)
    w3 = random.uniform(-0.15, 0.15)
    w4 = random.uniform(-0.15, 0.15)
    
    # Neck and head positions
    nx1 = random.uniform(0.38, 0.43)
    nx2 = random.uniform(0.57, 0.62)
    
    hx1 = random.uniform(0.25, 0.35)
    hx2 = random.uniform(0.65, 0.75)
    
    hy1 = random.uniform(-0.12, -0.20)
    hy2 = random.uniform(-0.22, -0.32)
    
    raw_segments = [
        # Left shoulder
        [(0.15, w1), (0.30, w2), (nx1, 0.0)],
        # Left neck and head
        [(nx1 - 0.05, -0.05), (hx1, -0.05), (hx1, hy1)],
        # Top left of head
        [(hx1, hy2), (0.45, hy2 - 0.05), (0.5, hy2 - 0.05)],
        # Top right of head
        [(0.55, hy2 - 0.05), (hx2, hy2), (hx2, hy1)],
        # Right neck and head
        [(hx2, -0.05), (nx2 + 0.05, -0.05), (nx2, 0.0)],
        # Right shoulder
        [(nx2 + 0.1, w3), (0.85, w4), (1.0, 0.0)]
    ]
    
    points = []
    p0 = (0.0, 0.0)
    for seg in raw_segments:
        p1 = (seg[0][0], seg[0][1] * tab_direction)
        p2 = (seg[1][0], seg[1][1] * tab_direction)
        p3 = (seg[2][0], seg[2][1] * tab_direction)
        
        curve = cubic_bezier(p0, p1, p2, p3, num_points=20)
        if len(points) == 0:
            points.extend(curve)
        else:
            points.extend(curve[1:])
        p0 = p3
        
    return points

def map_points(points, start_pos, end_pos):
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    
    mapped = []
    for px, py in points:
        sx = px * length
        sy = py * length
        rx = sx * math.cos(angle) - sy * math.sin(angle)
        ry = sx * math.sin(angle) + sy * math.cos(angle)
        mapped.append((start_pos[0] + rx, start_pos[1] + ry))
    return mapped

def extract_piece(image, polygon):
    # Use supersampling for anti-aliasing to get smooth edges
    scale = 4
    mask_size = (image.width * scale, image.height * scale)
    mask_hq = Image.new("L", mask_size, 0)
    draw = ImageDraw.Draw(mask_hq)
    
    polygon_hq = [(p[0] * scale, p[1] * scale) for p in polygon]
    draw.polygon(polygon_hq, fill=255)
    
    # Downscale with LANCZOS for high-quality anti-aliasing
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
        
    mask = mask_hq.resize(image.size, resample=resample_filter)
    
    piece = Image.new("RGBA", image.size, (0, 0, 0, 0))
    piece.paste(image, (0, 0), mask)
    
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    
    padding = 2
    min_x = max(0, math.floor(min_x) - padding)
    min_y = max(0, math.floor(min_y) - padding)
    max_x = min(image.width, math.ceil(max_x) + padding)
    max_y = min(image.height, math.ceil(max_y) + padding)
    
    piece_cropped = piece.crop((min_x, min_y, max_x, max_y))
    return piece_cropped, (min_x, min_y)

def generate_puzzle_pieces(image_path, rows, cols):
    print(f"Loading image {image_path}...")
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None

    width, height = img.size
    w = width / cols
    h = height / rows
    tab_size = min(w, h) * 0.2

    print(f"Generating puzzle grid ({rows}x{cols})...")
    
    # Generate boundaries
    h_boundaries = [[[] for _ in range(cols)] for _ in range(rows + 1)]
    v_boundaries = [[[] for _ in range(cols + 1)] for _ in range(rows)]

    # Horizontal boundaries
    for r in range(rows + 1):
        for c in range(cols):
            start = (c * w, r * h)
            end = ((c + 1) * w, r * h)
            if r == 0 or r == rows:
                h_boundaries[r][c] = [start, end]
            else:
                tab_dir = random.choice([1, -1])
                h_boundaries[r][c] = map_points(get_jigsaw_edge(tab_dir), start, end)

    # Vertical boundaries
    for r in range(rows):
        for c in range(cols + 1):
            start = (c * w, r * h)
            end = (c * w, (r + 1) * h)
            if c == 0 or c == cols:
                v_boundaries[r][c] = [start, end]
            else:
                tab_dir = random.choice([1, -1])
                v_boundaries[r][c] = map_points(get_jigsaw_edge(tab_dir), start, end)

    pieces = []
    print("Extracting pieces...")
    for r in range(rows):
        for c in range(cols):
            # Clockwise perimeter
            top = h_boundaries[r][c]
            right = v_boundaries[r][c + 1]
            bottom = h_boundaries[r + 1][c][::-1]
            left = v_boundaries[r][c][::-1]
            
            # Combine paths, avoiding duplicate corners
            polygon = top[:-1] + right[:-1] + bottom[:-1] + left[:-1]
            
            piece_img, pos = extract_piece(img, polygon)
            pieces.append((piece_img, pos))
            
    return pieces, width, height

def scatter_pieces(pieces, width, height, output_path):
    print(f"Scattering {len(pieces)} pieces onto canvas...")
    canvas_w = width * 3
    canvas_h = height * 3
    # Use a transparent background.
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    from PIL import ImageChops, ImageFilter
    canvas_mask = Image.new("1", (canvas_w, canvas_h), 0)
    
    for i, (piece_img, (ox, oy)) in enumerate(pieces):
        angle = random.uniform(0, 360)
        rotated = piece_img.rotate(angle, resample=Image.BICUBIC, expand=True)
        piece_mask = rotated.split()[3].point(lambda p: 1 if p > 10 else 0, mode="1")
        # Create a buffered mask for collision detection to ensure spacing
        buffered_mask = piece_mask.filter(ImageFilter.MaxFilter(15))
        
        max_x = canvas_w - rotated.width
        max_y = canvas_h - rotated.height
        
        placed = False
        if max_x > 0 and max_y > 0:
            for attempt in range(1000):
                rx = random.randint(0, max_x)
                ry = random.randint(0, max_y)
                
                region = canvas_mask.crop((rx, ry, rx + rotated.width, ry + rotated.height))
                collision = ImageChops.logical_and(region, buffered_mask).getbbox()
                
                if collision is None:
                    # No collision, place it here
                    # Paste the buffered mask to the canvas mask to keep pieces apart
                    canvas_mask.paste(1, (rx, ry), buffered_mask)
                    canvas.paste(rotated, (rx, ry), rotated)
                    placed = True
                    break
                    
        if not placed:
            print(f"Warning: Could not find a non-overlapping spot for piece {i}. Placing randomly.")
            rx = random.randint(0, max(0, max_x))
            ry = random.randint(0, max(0, max_y))
            canvas.paste(rotated, (rx, ry), rotated)
        
    print(f"Saving to {output_path}...")
    canvas.save(output_path)
    print("Done!")

def generate_puzzle(image_path, rows, cols, output_path):
    pieces, width, height = generate_puzzle_pieces(image_path, rows, cols)
    if pieces:
        scatter_pieces(pieces, width, height, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scattered jigsaw puzzle from an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-r", "--rows", type=int, default=4, help="Number of rows (default: 4)")
    parser.add_argument("-c", "--cols", type=int, default=4, help="Number of columns (default: 4)")
    parser.add_argument("-o", "--output", default="puzzle_scattered.png", help="Path to save the output image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.")
    else:
        generate_puzzle(args.image, args.rows, args.cols, args.output)
