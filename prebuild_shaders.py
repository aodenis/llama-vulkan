#!/usr/bin/env python3

import os
import struct
import sys
import subprocess as sp

# Structure : [size] [count n] [n file struct] [data]
# File struct : [name offset, content offset, size]
# Shader are consecutive in data


def main():
    if not os.path.exists("prebuilt_shaders"):
        os.mkdir("prebuilt_shaders", mode=0o755)

    else:
        for x in os.listdir("prebuilt_shaders"):
            if x.endswith(".spv"):
                os.unlink("prebuilt_shaders/" + x)

    shader_sources = [x.removesuffix(".comp") for x in os.listdir("shaders") if x.endswith(".comp")]

    for shader_source in shader_sources:
        ret = sp.call(["glslangValidator", "--target-env", "vulkan1.2", "-DUSE_SPEVAR=1", "-e", "main", "--quiet",
                       f"shaders/{shader_source}.comp", "-o", f"prebuilt_shaders/{shader_source}.spv"])
        if ret != 0:
            return ret

    count = len(shader_sources)
    full_header_size = (1 + 3 * count) * 4
    data = b"\0" * full_header_size
    shader_name_offsets: list[int] = []
    spv_offsets: list[tuple[int, int]] = []

    for shader in shader_sources:
        shader_name = shader.encode('ascii') + b'\0'
        shader_name_offsets.append(len(data))
        data += shader_name

    data += b"\0" * ((4 - len(data)) % 4)  # padding

    for shader in shader_sources:
        spv = open(f"prebuilt_shaders/{shader}.spv", "rb").read()
        spv_offsets.append((len(data), len(spv)))
        data += spv

    header = [count]
    for a, (b, c) in zip(shader_name_offsets, spv_offsets):
        header.append(a)
        header.append(b)
        header.append(c)
    header = struct.pack("I" * (count * 3 + 1), *header)
    assert len(header) == full_header_size
    data = header + data[full_header_size:]
    encoded_data = "const unsigned char raw_packed_shaders[] = {" + ",".join(map(str, data)) + "};\n"
    open("generated/packed_spv.c", "w").write(encoded_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
