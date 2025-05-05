import cv2
import numpy as np
import argparse

def read_ppm(filename):
    with open(filename, 'r') as f:
        # Read header
        assert f.readline().strip() == 'P3'
        width, height = map(int, f.readline().split())
        assert f.readline().strip() == '255'
        
        # Read pixel data
        data = []
        for line in f:
            data.extend(map(int, line.strip().split()))
        
        # Convert to numpy array
        pixels = np.array(data, dtype=np.uint8).reshape((height, width, 3))
        return pixels

def main():
    parser = argparse.ArgumentParser(description='Convert PARCS output to image')
    parser.add_argument('--input', required=True, help='Input PPM file path')
    parser.add_argument('--output', required=True, help='Output image path')
    args = parser.parse_args()

    # Read and convert PPM to BGR format
    ppm_image = read_ppm(args.input)
    bgr_image = cv2.cvtColor(ppm_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, bgr_image)

if __name__ == "__main__":
    main()
