#include "image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tilegen {

namespace {

constexpr std::uint32_t kCrcPolynomial = 0xEDB88320U;

std::array<std::uint32_t, 256> makeCrcTable() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t value = i;
        for (int bit = 0; bit < 8; ++bit) {
            value = (value & 1U) ? (value >> 1U) ^ kCrcPolynomial : value >> 1U;
        }
        table[i] = value;
    }
    return table;
}

const std::array<std::uint32_t, 256>& crcTable() {
    static const auto table = makeCrcTable();
    return table;
}

std::uint32_t adler32(const std::vector<std::uint8_t>& data) {
    constexpr std::uint32_t mod = 65521U;
    std::uint32_t a = 1;
    std::uint32_t b = 0;
    for (std::uint8_t byte : data) {
        a = (a + byte) % mod;
        b = (b + a) % mod;
    }
    return (b << 16U) | a;
}

void appendU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>(value & 0xFFU));
}

void appendChunk(std::vector<std::uint8_t>& out, const char type[5],
                 const std::vector<std::uint8_t>& data) {
    appendU32(out, static_cast<std::uint32_t>(data.size()));
    const std::size_t typeStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t checksum = crc32(out.data() + typeStart, 4 + data.size());
    appendU32(out, checksum);
}

std::uint32_t readU32(const std::vector<std::uint8_t>& data, std::size_t offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("truncated PNG integer");
    }
    return (static_cast<std::uint32_t>(data[offset]) << 24U) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(data[offset + 3]);
}

std::vector<std::uint8_t> zlibStore(const std::vector<std::uint8_t>& input) {
    std::vector<std::uint8_t> out;
    out.reserve(input.size() + input.size() / 65535U * 5U + 16U);
    out.push_back(0x78);
    out.push_back(0x01);
    std::size_t offset = 0;
    do {
        const std::size_t remaining = input.size() - offset;
        const std::uint16_t length =
            static_cast<std::uint16_t>(std::min<std::size_t>(remaining, 65535U));
        const bool finalBlock = offset + length >= input.size();
        out.push_back(finalBlock ? 1U : 0U);
        out.push_back(static_cast<std::uint8_t>(length & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((length >> 8U) & 0xFFU));
        const std::uint16_t inverse = static_cast<std::uint16_t>(~length);
        out.push_back(static_cast<std::uint8_t>(inverse & 0xFFU));
        out.push_back(static_cast<std::uint8_t>((inverse >> 8U) & 0xFFU));
        out.insert(out.end(), input.begin() + static_cast<std::ptrdiff_t>(offset),
                   input.begin() + static_cast<std::ptrdiff_t>(offset + length));
        offset += length;
    } while (offset < input.size());
    appendU32(out, adler32(input));
    return out;
}

class BitReader {
public:
    explicit BitReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::uint32_t bits(int count) {
        if (count == 0) {
            return 0;
        }
        if (count < 0 || count > 24) {
            throw std::runtime_error("invalid deflate bit count");
        }
        while (bitCount_ < count) {
            if (position_ >= bytes_.size()) {
                throw std::runtime_error("truncated deflate stream");
            }
            bitBuffer_ |= static_cast<std::uint64_t>(bytes_[position_++]) << bitCount_;
            bitCount_ += 8;
        }
        const std::uint32_t value =
            static_cast<std::uint32_t>(bitBuffer_ & ((1ULL << count) - 1ULL));
        bitBuffer_ >>= count;
        bitCount_ -= count;
        return value;
    }

    void alignToByte() {
        const int discard = bitCount_ % 8;
        if (discard != 0) {
            bits(discard);
        }
    }

    std::uint32_t readUint32LittleEndian() {
        alignToByte();
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            value |= bits(8) << (8 * i);
        }
        return value;
    }

    void skipBytes(std::size_t count) {
        alignToByte();
        for (std::size_t i = 0; i < count; ++i) {
            bits(8);
        }
    }

    std::size_t bytesConsumed() const { return position_; }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t position_ = 0;
    std::uint64_t bitBuffer_ = 0;
    int bitCount_ = 0;
};

struct Huffman {
    static constexpr int kMaxBits = 15;
    std::array<int, kMaxBits + 1> count{};
    std::vector<int> symbol;
};

Huffman buildHuffman(const std::vector<int>& lengths) {
    Huffman huffman;
    for (int length : lengths) {
        if (length < 0 || length > Huffman::kMaxBits) {
            throw std::runtime_error("invalid Huffman code length");
        }
        if (length > 0) {
            ++huffman.count[length];
        }
    }
    if (huffman.count[0] == static_cast<int>(lengths.size())) {
        return huffman;
    }
    int left = 1;
    for (int length = 1; length <= Huffman::kMaxBits; ++length) {
        left <<= 1;
        left -= huffman.count[length];
        if (left < 0) {
            throw std::runtime_error("oversubscribed Huffman tree");
        }
    }
    std::array<int, Huffman::kMaxBits + 1> offsets{};
    for (int length = 1; length < Huffman::kMaxBits; ++length) {
        offsets[length + 1] = offsets[length] + huffman.count[length];
    }
    huffman.symbol.assign(lengths.size(), 0);
    for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
        if (lengths[symbol] != 0) {
            huffman.symbol[static_cast<std::size_t>(offsets[lengths[symbol]]++)] =
                static_cast<int>(symbol);
        }
    }
    return huffman;
}

int decodeSymbol(BitReader& reader, const Huffman& huffman) {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int length = 1; length <= Huffman::kMaxBits; ++length) {
        code |= static_cast<int>(reader.bits(1));
        const int count = huffman.count[length];
        if (code - count < first) {
            const std::size_t symbolIndex =
                static_cast<std::size_t>(index + code - first);
            if (symbolIndex >= huffman.symbol.size()) {
                throw std::runtime_error("invalid Huffman symbol");
            }
            return huffman.symbol[symbolIndex];
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    throw std::runtime_error("invalid Huffman code");
}

const std::array<int, 29> kLengthBase = {
    3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

const std::array<int, 29> kLengthExtra = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

const std::array<int, 30> kDistanceBase = {
    1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

const std::array<int, 30> kDistanceExtra = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
    6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

std::vector<std::uint8_t> inflateZlib(const std::vector<std::uint8_t>& input,
                                     std::size_t expectedSize) {
    if (input.size() < 6) {
        throw std::runtime_error("truncated zlib stream");
    }
    const std::uint8_t cmf = input[0];
    const std::uint8_t flg = input[1];
    if ((cmf & 0x0FU) != 8) {
        throw std::runtime_error("unsupported zlib compression method");
    }
    if (((static_cast<unsigned int>(cmf) << 8U) + flg) % 31U != 0U) {
        throw std::runtime_error("invalid zlib header checksum");
    }
    if ((flg & 0x20U) != 0U) {
        throw std::runtime_error("zlib preset dictionaries are not supported");
    }

    const std::vector<std::uint8_t> payload(input.begin() + 2, input.end() - 4);
    BitReader reader(payload);
    std::vector<std::uint8_t> output;
    output.reserve(expectedSize);

    bool finalBlock = false;
    while (!finalBlock) {
        finalBlock = reader.bits(1) != 0;
        const std::uint32_t blockType = reader.bits(2);
        if (blockType == 0) {
            reader.alignToByte();
            const std::uint32_t length = reader.bits(16);
            const std::uint32_t inverse = reader.bits(16);
            if ((length ^ 0xFFFFU) != inverse) {
                throw std::runtime_error("invalid stored deflate block length");
            }
            for (std::uint32_t i = 0; i < length; ++i) {
                output.push_back(static_cast<std::uint8_t>(reader.bits(8)));
            }
            continue;
        }

        Huffman literalTree;
        Huffman distanceTree;
        if (blockType == 1) {
            std::vector<int> literalLengths(288, 0);
            for (int i = 0; i <= 143; ++i) {
                literalLengths[static_cast<std::size_t>(i)] = 8;
            }
            for (int i = 144; i <= 255; ++i) {
                literalLengths[static_cast<std::size_t>(i)] = 9;
            }
            for (int i = 256; i <= 279; ++i) {
                literalLengths[static_cast<std::size_t>(i)] = 7;
            }
            for (int i = 280; i <= 287; ++i) {
                literalLengths[static_cast<std::size_t>(i)] = 8;
            }
            literalTree = buildHuffman(literalLengths);
            distanceTree = buildHuffman(std::vector<int>(32, 5));
        } else if (blockType == 2) {
            const int hlit = static_cast<int>(reader.bits(5)) + 257;
            const int hdist = static_cast<int>(reader.bits(5)) + 1;
            const int hclen = static_cast<int>(reader.bits(4)) + 4;
            static constexpr std::array<int, 19> order = {
                16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            std::vector<int> codeLengths(19, 0);
            for (int i = 0; i < hclen; ++i) {
                codeLengths[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] =
                    static_cast<int>(reader.bits(3));
            }
            const Huffman codeTree = buildHuffman(codeLengths);
            std::vector<int> lengths;
            lengths.reserve(static_cast<std::size_t>(hlit + hdist));
            while (static_cast<int>(lengths.size()) < hlit + hdist) {
                const int symbol = decodeSymbol(reader, codeTree);
                if (symbol <= 15) {
                    lengths.push_back(symbol);
                } else if (symbol == 16) {
                    if (lengths.empty()) {
                        throw std::runtime_error("repeat code without previous length");
                    }
                    const int repeat = 3 + static_cast<int>(reader.bits(2));
                    const int previous = lengths.back();
                    for (int i = 0; i < repeat; ++i) {
                        lengths.push_back(previous);
                    }
                } else if (symbol == 17) {
                    const int repeat = 3 + static_cast<int>(reader.bits(3));
                    lengths.insert(lengths.end(), static_cast<std::size_t>(repeat), 0);
                } else if (symbol == 18) {
                    const int repeat = 11 + static_cast<int>(reader.bits(7));
                    lengths.insert(lengths.end(), static_cast<std::size_t>(repeat), 0);
                } else {
                    throw std::runtime_error("invalid code-length symbol");
                }
                if (lengths.size() > static_cast<std::size_t>(hlit + hdist)) {
                    throw std::runtime_error("too many Huffman code lengths");
                }
            }
            std::vector<int> literalLengths(lengths.begin(), lengths.begin() + hlit);
            std::vector<int> distanceLengths(lengths.begin() + hlit, lengths.end());
            literalTree = buildHuffman(literalLengths);
            distanceTree = buildHuffman(distanceLengths);
        } else {
            throw std::runtime_error("invalid deflate block type");
        }

        while (true) {
            const int symbol = decodeSymbol(reader, literalTree);
            if (symbol < 256) {
                output.push_back(static_cast<std::uint8_t>(symbol));
            } else if (symbol == 256) {
                break;
            } else if (symbol >= 257 && symbol <= 285) {
                const std::size_t lengthIndex = static_cast<std::size_t>(symbol - 257);
                const int length = kLengthBase[lengthIndex] +
                                   static_cast<int>(reader.bits(kLengthExtra[lengthIndex]));
                const int distanceSymbol = decodeSymbol(reader, distanceTree);
                if (distanceSymbol < 0 || distanceSymbol >= 30) {
                    throw std::runtime_error("invalid deflate distance symbol");
                }
                const std::size_t distanceIndex =
                    static_cast<std::size_t>(distanceSymbol);
                const int distance =
                    kDistanceBase[distanceIndex] +
                    static_cast<int>(reader.bits(kDistanceExtra[distanceIndex]));
                if (distance <= 0 || static_cast<std::size_t>(distance) > output.size()) {
                    throw std::runtime_error("invalid deflate back-reference distance");
                }
                for (int i = 0; i < length; ++i) {
                    output.push_back(output[output.size() -
                                            static_cast<std::size_t>(distance)]);
                }
            } else {
                throw std::runtime_error("invalid literal/length symbol");
            }
            if (output.size() > expectedSize) {
                throw std::runtime_error("inflated data exceeds expected PNG size");
            }
        }
        if (output.size() > expectedSize) {
            throw std::runtime_error("inflated data exceeds expected PNG size");
        }
    }

    if (output.size() != expectedSize) {
        std::ostringstream message;
        message << "inflated PNG size mismatch: got " << output.size()
                << ", expected " << expectedSize;
        throw std::runtime_error(message.str());
    }
    const std::uint32_t expectedAdler = readU32(input, input.size() - 4);
    if (adler32(output) != expectedAdler) {
        throw std::runtime_error("zlib Adler-32 checksum mismatch");
    }
    return output;
}

std::uint8_t paethPredictor(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) {
        return static_cast<std::uint8_t>(a);
    }
    if (pb <= pc) {
        return static_cast<std::uint8_t>(b);
    }
    return static_cast<std::uint8_t>(c);
}

std::vector<Rgba8> decodePngBytes(const std::vector<std::uint8_t>& bytes,
                                  int& width, int& height) {
    static constexpr std::array<std::uint8_t, 8> signature = {
        0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (bytes.size() < signature.size() ||
        !std::equal(signature.begin(), signature.end(), bytes.begin())) {
        throw std::runtime_error("invalid PNG signature");
    }

    bool sawHeader = false;
    std::uint32_t pngWidth = 0;
    std::uint32_t pngHeight = 0;
    int bitDepth = 0;
    int colorType = -1;
    int interlace = 0;
    std::vector<std::uint8_t> compressed;
    std::vector<Rgba8> palette(256);
    std::size_t paletteSize = 0;
    std::vector<std::uint8_t> transparency;

    std::size_t position = signature.size();
    bool sawEnd = false;
    while (position + 12 <= bytes.size()) {
        const std::uint32_t length = readU32(bytes, position);
        if (length > std::numeric_limits<std::uint32_t>::max() / 2U ||
            position + 12U + length > bytes.size()) {
            throw std::runtime_error("truncated PNG chunk");
        }
        const char* type =
            reinterpret_cast<const char*>(bytes.data() + position + 4);
        const std::uint32_t storedCrc = readU32(bytes, position + 8 + length);
        const std::uint32_t actualCrc = crc32(bytes.data() + position + 4, 4 + length);
        if (storedCrc != actualCrc) {
            throw std::runtime_error("PNG chunk CRC mismatch");
        }
        const std::vector<std::uint8_t> data(
            bytes.begin() + static_cast<std::ptrdiff_t>(position + 8),
            bytes.begin() + static_cast<std::ptrdiff_t>(position + 8 + length));
        if (std::strncmp(type, "IHDR", 4) == 0) {
            if (sawHeader || length != 13) {
                throw std::runtime_error("invalid PNG IHDR chunk");
            }
            sawHeader = true;
            pngWidth = readU32(data, 0);
            pngHeight = readU32(data, 4);
            bitDepth = data[8];
            colorType = data[9];
            if (data[10] != 0 || data[11] != 0) {
                throw std::runtime_error("unsupported PNG compression/filter method");
            }
            interlace = data[12];
        } else if (std::strncmp(type, "PLTE", 4) == 0) {
            if (length % 3 != 0 || length > 768) {
                throw std::runtime_error("invalid PNG palette");
            }
            paletteSize = length / 3;
            for (std::size_t i = 0; i < paletteSize; ++i) {
                palette[i] = {data[i * 3], data[i * 3 + 1], data[i * 3 + 2], 255};
            }
        } else if (std::strncmp(type, "tRNS", 4) == 0) {
            transparency = data;
        } else if (std::strncmp(type, "IDAT", 4) == 0) {
            compressed.insert(compressed.end(), data.begin(), data.end());
        } else if (std::strncmp(type, "IEND", 4) == 0) {
            sawEnd = true;
            break;
        }
        position += 12 + length;
    }
    if (!sawHeader || !sawEnd) {
        throw std::runtime_error("PNG is missing IHDR or IEND");
    }
    if (pngWidth == 0 || pngHeight == 0 || pngWidth > 32768 || pngHeight > 32768) {
        throw std::runtime_error("invalid PNG dimensions");
    }
    if (interlace != 0) {
        throw std::runtime_error("interlaced PNG is not supported");
    }
    if (bitDepth != 8) {
        throw std::runtime_error("only 8-bit PNG textures are supported");
    }
    if (colorType != 0 && colorType != 2 && colorType != 3 && colorType != 4 &&
        colorType != 6) {
        throw std::runtime_error("unsupported PNG color type");
    }
    if (colorType == 3 && paletteSize == 0) {
        throw std::runtime_error("palette PNG has no PLTE chunk");
    }
    if (compressed.empty()) {
        throw std::runtime_error("PNG has no IDAT data");
    }

    int channels = 0;
    switch (colorType) {
        case 0:
            channels = 1;
            break;
        case 2:
            channels = 3;
            break;
        case 3:
            channels = 1;
            break;
        case 4:
            channels = 2;
            break;
        case 6:
            channels = 4;
            break;
    }
    const std::size_t rowBytes = static_cast<std::size_t>(pngWidth) *
                                 static_cast<std::size_t>(channels);
    const std::size_t expectedSize =
        (rowBytes + 1U) * static_cast<std::size_t>(pngHeight);
    std::vector<std::uint8_t> raw = inflateZlib(compressed, expectedSize);
    std::vector<std::uint8_t> reconstructed(rowBytes * pngHeight);

    for (std::uint32_t y = 0; y < pngHeight; ++y) {
        const std::size_t rowStart = static_cast<std::size_t>(y) * (rowBytes + 1U);
        const int filter = raw[rowStart];
        const std::uint8_t* source = raw.data() + rowStart + 1;
        std::uint8_t* destination =
            reconstructed.data() + static_cast<std::size_t>(y) * rowBytes;
        const std::uint8_t* previous =
            y == 0 ? nullptr
                   : reconstructed.data() + static_cast<std::size_t>(y - 1) * rowBytes;
        for (std::size_t x = 0; x < rowBytes; ++x) {
            const int left = x >= static_cast<std::size_t>(channels)
                                 ? destination[x - static_cast<std::size_t>(channels)]
                                 : 0;
            const int up = previous ? previous[x] : 0;
            const int upperLeft =
                previous && x >= static_cast<std::size_t>(channels)
                    ? previous[x - static_cast<std::size_t>(channels)]
                    : 0;
            int predictor = 0;
            switch (filter) {
                case 0:
                    predictor = 0;
                    break;
                case 1:
                    predictor = left;
                    break;
                case 2:
                    predictor = up;
                    break;
                case 3:
                    predictor = (left + up) / 2;
                    break;
                case 4:
                    predictor = paethPredictor(left, up, upperLeft);
                    break;
                default:
                    throw std::runtime_error("invalid PNG row filter");
            }
            destination[x] = static_cast<std::uint8_t>(source[x] + predictor);
        }
    }

    std::vector<Rgba8> rgba(static_cast<std::size_t>(pngWidth) * pngHeight);
    for (std::uint32_t y = 0; y < pngHeight; ++y) {
        const std::uint8_t* row =
            reconstructed.data() + static_cast<std::size_t>(y) * rowBytes;
        for (std::uint32_t x = 0; x < pngWidth; ++x) {
            Rgba8 pixel{};
            if (colorType == 0) {
                pixel.r = pixel.g = pixel.b = row[x];
                pixel.a = 255;
            } else if (colorType == 2) {
                pixel.r = row[x * 3];
                pixel.g = row[x * 3 + 1];
                pixel.b = row[x * 3 + 2];
                pixel.a = 255;
            } else if (colorType == 3) {
                const std::size_t index = row[x];
                if (index >= paletteSize) {
                    throw std::runtime_error("PNG palette index out of range");
                }
                pixel = palette[index];
            } else if (colorType == 4) {
                pixel.r = pixel.g = pixel.b = row[x * 2];
                pixel.a = row[x * 2 + 1];
            } else {
                pixel.r = row[x * 4];
                pixel.g = row[x * 4 + 1];
                pixel.b = row[x * 4 + 2];
                pixel.a = row[x * 4 + 3];
            }
            rgba[static_cast<std::size_t>(y) * pngWidth + x] = pixel;
        }
    }

    if (!transparency.empty()) {
        for (std::size_t i = 0; i < rgba.size(); ++i) {
            if (colorType == 0 && transparency.size() >= 2) {
                const int key =
                    (static_cast<int>(transparency[0]) << 8) | transparency[1];
                if (rgba[i].r == static_cast<std::uint8_t>(key & 0xFF)) {
                    rgba[i].a = 0;
                }
            } else if (colorType == 2 && transparency.size() >= 6) {
                const int red =
                    (static_cast<int>(transparency[0]) << 8) | transparency[1];
                const int green =
                    (static_cast<int>(transparency[2]) << 8) | transparency[3];
                const int blue =
                    (static_cast<int>(transparency[4]) << 8) | transparency[5];
                if (rgba[i].r == static_cast<std::uint8_t>(red & 0xFF) &&
                    rgba[i].g == static_cast<std::uint8_t>(green & 0xFF) &&
                    rgba[i].b == static_cast<std::uint8_t>(blue & 0xFF)) {
                    rgba[i].a = 0;
                }
            }
        }
        if (colorType == 3) {
            for (std::size_t i = 0; i < rgba.size(); ++i) {
                const std::size_t paletteIndex =
                    reconstructed[((i / pngWidth) * rowBytes) + (i % pngWidth)];
                if (paletteIndex < transparency.size()) {
                    rgba[i].a = transparency[paletteIndex];
                }
            }
        }
    }

    width = static_cast<int>(pngWidth);
    height = static_cast<int>(pngHeight);
    return rgba;
}

}  // namespace

Image::Image(int widthValue, int heightValue, PremulPixel fill)
    : width(widthValue),
      height(heightValue),
      pixels(static_cast<std::size_t>(std::max(0, widthValue)) *
             static_cast<std::size_t>(std::max(0, heightValue)),
             fill) {}

PremulPixel& Image::at(int x, int y) {
    return pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)];
}

const PremulPixel& Image::at(int x, int y) const {
    return pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)];
}

Rgba8 Texture::sampleBilinear(double u, double v, bool repeat) const {
    if (empty()) {
        return {};
    }
    if (!std::isfinite(u) || !std::isfinite(v)) {
        throw std::runtime_error("texture coordinate is not finite");
    }
    double x = u * static_cast<double>(width) - 0.5;
    double y = v * static_cast<double>(height) - 0.5;
    auto coordinate = [repeat](double value, int size) {
        const double limit = static_cast<double>(size);
        if (repeat) {
            value = std::fmod(value, limit);
            if (value < 0.0) {
                value += limit;
            }
        } else {
            value = std::clamp(value, 0.0, limit - 1.0);
        }
        return value;
    };
    auto pixelAt = [this](int x, int y) -> const Rgba8& {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                      static_cast<std::size_t>(x)];
    };

    const double x0 = std::floor(x);
    const double y0 = std::floor(y);
    const double fx = x - x0;
    const double fy = y - y0;
    const int ix0 = static_cast<int>(x0);
    const int iy0 = static_cast<int>(y0);
    const int ix1 = repeat ? static_cast<int>(coordinate(x0 + 1.0, width))
                           : std::clamp(ix0 + 1, 0, width - 1);
    const int iy1 = repeat ? static_cast<int>(coordinate(y0 + 1.0, height))
                           : std::clamp(iy0 + 1, 0, height - 1);
    const int cx0 = repeat ? static_cast<int>(coordinate(x0, width))
                           : std::clamp(ix0, 0, width - 1);
    const int cy0 = repeat ? static_cast<int>(coordinate(y0, height))
                           : std::clamp(iy0, 0, height - 1);

    const Rgba8& p00 = pixelAt(cx0, cy0);
    const Rgba8& p10 = pixelAt(ix1, cy0);
    const Rgba8& p01 = pixelAt(cx0, iy1);
    const Rgba8& p11 = pixelAt(ix1, iy1);
    auto blend = [fx, fy](std::uint8_t a, std::uint8_t b, std::uint8_t c,
                          std::uint8_t d) {
        const double top = a * (1.0 - fx) + b * fx;
        const double bottom = c * (1.0 - fx) + d * fx;
        return static_cast<std::uint8_t>(
            std::clamp(std::lround(top * (1.0 - fy) + bottom * fy), 0L, 255L));
    };
    return {blend(p00.r, p10.r, p01.r, p11.r),
            blend(p00.g, p10.g, p01.g, p11.g),
            blend(p00.b, p10.b, p01.b, p11.b),
            blend(p00.a, p10.a, p01.a, p11.a)};
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t size,
                    std::uint32_t initial) {
    std::uint32_t value = initial ^ 0xFFFFFFFFU;
    for (std::size_t i = 0; i < size; ++i) {
        value = crcTable()[(value ^ data[i]) & 0xFFU] ^ (value >> 8U);
    }
    return value ^ 0xFFFFFFFFU;
}

std::uint32_t crc32(const std::vector<std::uint8_t>& data) {
    return crc32(data.data(), data.size());
}

std::string crc32Hex(std::uint32_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(8, '0');
    for (int i = 7; i >= 0; --i) {
        result[static_cast<std::size_t>(i)] = digits[value & 0xFU];
        value >>= 4U;
    }
    return result;
}

std::vector<std::uint8_t> encodePngRgba(int width, int height,
                                        const std::vector<Rgba8>& pixels) {
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<std::size_t>(width) *
                              static_cast<std::size_t>(height)) {
        throw std::runtime_error("invalid RGBA image passed to PNG encoder");
    }
    std::vector<std::uint8_t> raw;
    raw.reserve((static_cast<std::size_t>(width) * 4U + 1U) *
                static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        for (int x = 0; x < width; ++x) {
            const Rgba8& pixel =
                pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                       static_cast<std::size_t>(x)];
            raw.push_back(pixel.r);
            raw.push_back(pixel.g);
            raw.push_back(pixel.b);
            raw.push_back(pixel.a);
        }
    }

    std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<std::uint8_t> header;
    appendU32(header, static_cast<std::uint32_t>(width));
    appendU32(header, static_cast<std::uint32_t>(height));
    header.push_back(8);
    header.push_back(6);
    header.push_back(0);
    header.push_back(0);
    header.push_back(0);
    appendChunk(png, "IHDR", header);
    appendChunk(png, "IDAT", zlibStore(raw));
    appendChunk(png, "IEND", {});
    return png;
}

std::vector<Rgba8> decodePngRgba(const std::vector<std::uint8_t>& bytes,
                                 int& width, int& height) {
    return decodePngBytes(bytes, width, height);
}

std::vector<Rgba8> decodePngRgbaFile(const std::string& path, int& width,
                                     int& height) {
    return decodePngBytes(readBinaryFile(path), width, height);
}

std::vector<std::uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot determine file size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty() &&
        !input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("cannot read file: " + path);
    }
    return bytes;
}

void writeBinaryFile(const std::string& path,
                     const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open file for writing: " + path);
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("cannot write file: " + path);
    }
}

}  // namespace tilegen
