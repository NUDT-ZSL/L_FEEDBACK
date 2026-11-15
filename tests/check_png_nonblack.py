#!/usr/bin/env python3
import json
import struct
import sys
import zlib


def decode_png(path):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    position = 8
    compressed = bytearray()
    width = height = color_type = None
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        chunk_type = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        position += length + 12
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if bit_depth != 8 or compression != 0 or filtering != 0 or interlace != 0:
                raise ValueError("unsupported PNG encoding")
        elif chunk_type == b"IDAT":
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            break

    channels = {2: 3, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    previous = bytearray(stride)
    rows = []
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        row = bytearray(raw[offset:offset + stride])
        offset += stride
        for index in range(stride):
            left = row[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + above) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                estimate = left + above - upper_left
                left_distance = abs(estimate - left)
                above_distance = abs(estimate - above)
                upper_left_distance = abs(estimate - upper_left)
                predictor = (
                    left
                    if left_distance <= above_distance and left_distance <= upper_left_distance
                    else (above if above_distance <= upper_left_distance else upper_left)
                )
                row[index] = (row[index] + predictor) & 0xFF
            elif filter_type != 0:
                raise ValueError("unsupported PNG filter")
        rows.append(row)
        previous = row
    return width, height, channels, rows


def main():
    path = sys.argv[1]
    minimum_fraction = float(sys.argv[2])
    minimum_channel = int(sys.argv[3])
    width, height, channels, rows = decode_png(path)
    total = width * height
    nonblack = 0
    maximum = 0
    for row in rows:
        for index in range(0, len(row), channels):
            rgb = row[index:index + 3]
            channel_maximum = max(rgb)
            maximum = max(maximum, channel_maximum)
            if channel_maximum >= minimum_channel:
                nonblack += 1
    fraction = nonblack / total
    print(json.dumps({
        "path": path,
        "width": width,
        "height": height,
        "nonblack_pixels": nonblack,
        "nonblack_fraction": fraction,
        "maximum_channel": maximum,
    }))
    if fraction < minimum_fraction:
        raise SystemExit("PNG is too dark: ambient/background check failed")


if __name__ == "__main__":
    main()

