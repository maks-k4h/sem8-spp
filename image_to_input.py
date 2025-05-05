import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert image to PARCS input format')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--clusters', type=int, required=True, help='Number of clusters')
    parser.add_argument('--output', required=True, help='Output text file path')
    args = parser.parse_args()

    # Read image and convert to RGB
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Could not read image file")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb_img.shape

    # Write to output file
    with open(args.output, 'w') as f:
        f.write(f"{args.clusters}\n")
        f.write(f"{width} {height}\n")
        for row in rgb_img:
            pixel_line = ' '.join(f"{p[0]} {p[1]} {p[2]}" for p in row)
            f.write(pixel_line + '\n')

if __name__ == "__main__":
    main()