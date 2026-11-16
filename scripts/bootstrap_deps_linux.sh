#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_dir="$root/.deps/download"
prefix="$root/.deps/usr"
mkdir -p "$download_dir" "$prefix"

# These packages are extracted locally, so the script does not need sudo or a
# system-wide install. Runtime GL/X11 packages are included because the
# development-package linker scripts point at those shared objects.
packages=(
    libglm-dev
    libstb-dev
    nlohmann-json3-dev
    libglfw3-dev
    libglfw3
    libgl-dev
    libglx-dev
    libopengl-dev
    libglvnd-dev
    libglvnd0
    libglx0
    libopengl0
    libgl1
    libx11-dev
    libxrandr-dev
    libxinerama-dev
    libxcursor-dev
    libxi-dev
    libxext-dev
    libxxf86vm-dev
    libx11-6
    libxrandr2
    libxinerama1
    libxcursor1
    libxi6
    libxext6
    libxxf86vm1
)

cd "$download_dir"
for package in "${packages[@]}"; do
    echo "fetching $package"
    apt-get download "$package"
done

for archive in *.deb; do
    dpkg-deb -x "$archive" "$prefix"
done

echo
echo "local dependency prefix: $prefix/usr"
echo "configure with:"
echo "  cmake -S \"$root\" -B \"$root/build/linux-release\" \\"
echo "    -DCMAKE_BUILD_TYPE=Release \\"
echo "    -DCMAKE_PREFIX_PATH=\"$prefix/usr\""

