#include "scene.hpp"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>

namespace tilegen {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw SceneError(path, message);
}

const JsonValue* requireField(const JsonValue& object, const std::string& key,
                              const std::string& base) {
    const JsonValue* value = object.find(key);
    if (value == nullptr) {
        fail(jsonPath(base, key), "required field is missing");
    }
    return value;
}

const JsonValue* findAny(const JsonValue& object,
                         std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        const JsonValue* value = object.find(key);
        if (value != nullptr) {
            return value;
        }
    }
    return nullptr;
}

double requireNumber(const JsonValue& value, const std::string& path) {
    if (!value.isNumber()) {
        fail(path, "must be a finite number");
    }
    const double number = value.asNumber();
    if (!std::isfinite(number)) {
        fail(path, "must be finite (NaN and infinity are not allowed)");
    }
    return number;
}

bool optionalBool(const JsonValue& object, const std::string& key,
                  const std::string& base, bool fallback) {
    const JsonValue* value = object.find(key);
    if (value == nullptr) {
        return fallback;
    }
    if (!value->isBool()) {
        fail(jsonPath(base, key), "must be a boolean");
    }
    return value->asBool();
}

Vec2 parseVec2(const JsonValue& value, const std::string& path) {
    if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.size() != 2) {
            fail(path, "must contain exactly two numbers [x, y]");
        }
        return {requireNumber(array[0], jsonPath(path, 0)),
                requireNumber(array[1], jsonPath(path, 1))};
    }
    if (value.isObject()) {
        return {
            requireNumber(*requireField(value, "x", path), jsonPath(path, "x")),
            requireNumber(*requireField(value, "y", path), jsonPath(path, "y"))};
    }
    fail(path, "must be [x, y] or an object with x/y");
}

std::vector<Vec2> parsePoints(const JsonValue& value, const std::string& path,
                              std::size_t minimum) {
    if (!value.isArray()) {
        fail(path, "must be an array of points");
    }
    std::vector<Vec2> points;
    points.reserve(value.asArray().size());
    for (std::size_t i = 0; i < value.asArray().size(); ++i) {
        points.push_back(parseVec2(value.asArray()[i], jsonPath(path, i)));
    }
    if (points.size() < minimum) {
        fail(path, "must contain at least " + std::to_string(minimum) + " points");
    }
    return points;
}

double parseColorComponent(const JsonValue& value, const std::string& path) {
    const double component = requireNumber(value, path);
    if (component < 0.0) {
        fail(path, "color components cannot be negative");
    }
    return component;
}

int parseHexDigit(char ch, const std::string& path) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    fail(path, "invalid hexadecimal color");
}

Color parseHexColor(const std::string& text, const std::string& path) {
    if (text.empty() || text[0] != '#') {
        fail(path, "invalid color string");
    }
    const std::string digits = text.substr(1);
    auto expandShort = [&path](char ch) {
        const int value = parseHexDigit(ch, path);
        return static_cast<double>(value * 17) / 255.0;
    };
    if (digits.size() == 3 || digits.size() == 4) {
        Color color;
        color.r = expandShort(digits[0]);
        color.g = expandShort(digits[1]);
        color.b = expandShort(digits[2]);
        color.a = digits.size() == 4 ? expandShort(digits[3]) : 1.0;
        return color;
    }
    if (digits.size() == 6 || digits.size() == 8) {
        auto byteAt = [&path, &digits](std::size_t offset) {
            const int high = parseHexDigit(digits[offset], path);
            const int low = parseHexDigit(digits[offset + 1], path);
            return static_cast<double>(high * 16 + low) / 255.0;
        };
        Color color;
        color.r = byteAt(0);
        color.g = byteAt(2);
        color.b = byteAt(4);
        color.a = digits.size() == 8 ? byteAt(6) : 1.0;
        return color;
    }
    fail(path, "hex color must be #RGB, #RGBA, #RRGGBB, or #RRGGBBAA");
}

Color parseColor(const JsonValue& value, const std::string& path) {
    if (value.isString()) {
        const std::string& text = value.asString();
        if (!text.empty() && text[0] == '#') {
            return parseHexColor(text, path);
        }
        if (text == "transparent") {
            return {0.0, 0.0, 0.0, 0.0};
        }
        if (text == "black") {
            return {0.0, 0.0, 0.0, 1.0};
        }
        if (text == "white") {
            return {1.0, 1.0, 1.0, 1.0};
        }
        if (text == "red") {
            return {1.0, 0.0, 0.0, 1.0};
        }
        if (text == "green") {
            return {0.0, 1.0, 0.0, 1.0};
        }
        if (text == "blue") {
            return {0.0, 0.0, 1.0, 1.0};
        }
        fail(path, "unknown color name");
    }
    if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.size() != 3 && array.size() != 4) {
            fail(path, "color array must contain 3 or 4 components");
        }
        std::array<double, 4> components = {0.0, 0.0, 0.0, 1.0};
        bool byteScale = false;
        for (std::size_t i = 0; i < array.size(); ++i) {
            components[i] = parseColorComponent(array[i], jsonPath(path, i));
            byteScale = byteScale || components[i] > 1.0;
        }
        if (byteScale) {
            for (std::size_t i = 0; i < array.size(); ++i) {
                if (components[i] > 255.0) {
                    fail(jsonPath(path, i), "8-bit color component exceeds 255");
                }
                components[i] /= 255.0;
            }
        }
        return {components[0], components[1], components[2], components[3]};
    }
    if (value.isObject()) {
        Color color;
        color.r = parseColorComponent(*requireField(value, "r", path),
                                      jsonPath(path, "r"));
        color.g = parseColorComponent(*requireField(value, "g", path),
                                      jsonPath(path, "g"));
        color.b = parseColorComponent(*requireField(value, "b", path),
                                      jsonPath(path, "b"));
        if (value.contains("a")) {
            color.a = parseColorComponent(*value.find("a"), jsonPath(path, "a"));
        }
        const double maximum =
            std::max({color.r, color.g, color.b, color.a});
        if (maximum > 1.0) {
            if (maximum > 255.0) {
                fail(path, "color component exceeds 255");
            }
            color.r /= 255.0;
            color.g /= 255.0;
            color.b /= 255.0;
            color.a /= 255.0;
        }
        return color;
    }
    fail(path, "must be a color string, array, or object");
}

double parseOpacity(const JsonValue& object, const std::string& base,
                    double fallback) {
    const JsonValue* value = object.find("opacity");
    if (value == nullptr) {
        return fallback;
    }
    const double opacity = requireNumber(*value, jsonPath(base, "opacity"));
    if (opacity < 0.0 || opacity > 1.0) {
        fail(jsonPath(base, "opacity"), "must be in the range [0, 1]");
    }
    return opacity;
}

FillRule parseFillRule(const JsonValue& object, const std::string& base,
                       FillRule fallback) {
    const JsonValue* value = findAny(object, {"fill_rule", "fillRule", "winding"});
    if (value == nullptr) {
        return fallback;
    }
    if (!value->isString()) {
        fail(jsonPath(base, "fill_rule"), "must be \"nonzero\" or \"evenodd\"");
    }
    std::string rule = value->asString();
    std::transform(rule.begin(), rule.end(), rule.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (rule == "nonzero" || rule == "non-zero" || rule == "winding") {
        return FillRule::NonZero;
    }
    if (rule == "evenodd" || rule == "even-odd") {
        return FillRule::EvenOdd;
    }
    fail(jsonPath(base, "fill_rule"), "unknown fill rule '" + rule + "'");
}

BlendMode parseBlendMode(const JsonValue& object, const std::string& base) {
    const JsonValue* value = object.find("blend");
    if (value == nullptr) {
        value = object.find("blend_mode");
    }
    if (value == nullptr) {
        return BlendMode::Normal;
    }
    if (!value->isString()) {
        fail(jsonPath(base, "blend"), "must be \"normal\", \"add\", or \"multiply\"");
    }
    std::string mode = value->asString();
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (mode == "normal" || mode == "source-over" || mode == "source_over") {
        return BlendMode::Normal;
    }
    if (mode == "add" || mode == "plus" || mode == "linear-dodge") {
        return BlendMode::Add;
    }
    if (mode == "multiply") {
        return BlendMode::Multiply;
    }
    fail(jsonPath(base, "blend"), "unknown blend mode '" + mode + "'");
}

CapStyle parseCap(const JsonValue& object, const std::string& base) {
    const JsonValue* value = findAny(object, {"cap", "line_cap", "linecap"});
    if (value == nullptr) {
        return CapStyle::Butt;
    }
    if (!value->isString()) {
        fail(jsonPath(base, "cap"), "must be \"butt\", \"round\", or \"square\"");
    }
    std::string cap = value->asString();
    std::transform(cap.begin(), cap.end(), cap.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (cap == "butt") {
        return CapStyle::Butt;
    }
    if (cap == "round") {
        return CapStyle::Round;
    }
    if (cap == "square") {
        return CapStyle::Square;
    }
    fail(jsonPath(base, "cap"), "unknown line cap '" + cap + "'");
}

JoinStyle parseJoin(const JsonValue& object, const std::string& base) {
    const JsonValue* value = findAny(object, {"join", "line_join", "linejoin"});
    if (value == nullptr) {
        return JoinStyle::Miter;
    }
    if (!value->isString()) {
        fail(jsonPath(base, "join"), "must be \"miter\", \"round\", or \"bevel\"");
    }
    std::string join = value->asString();
    std::transform(join.begin(), join.end(), join.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (join == "miter") {
        return JoinStyle::Miter;
    }
    if (join == "round") {
        return JoinStyle::Round;
    }
    if (join == "bevel") {
        return JoinStyle::Bevel;
    }
    fail(jsonPath(base, "join"), "unknown line join '" + join + "'");
}

UvRect parseUvRect(const JsonValue& value, const std::string& path) {
    if (!value.isArray() || value.asArray().size() != 4) {
        fail(path, "must contain four numbers [u0, v0, u1, v1]");
    }
    return {requireNumber(value.asArray()[0], jsonPath(path, 0)),
            requireNumber(value.asArray()[1], jsonPath(path, 1)),
            requireNumber(value.asArray()[2], jsonPath(path, 2)),
            requireNumber(value.asArray()[3], jsonPath(path, 3))};
}

FillStyle parseFillStyle(const JsonValue& value, const std::string& path,
                         const Scene& scene) {
    FillStyle style;
    if (value.isArray() || value.isString()) {
        style.color = parseColor(value, path);
        return style;
    }
    if (!value.isObject()) {
        fail(path, "fill must be a color or an object");
    }
    const JsonValue* color = findAny(value, {"color", "tint"});
    if (color != nullptr) {
        style.color = parseColor(*color, jsonPath(path, "color"));
    }
    style.opacity = parseOpacity(value, path, 1.0);
    if (const JsonValue* uv = findAny(value, {"uv_rect", "uvRect", "uv"})) {
        style.uv = parseUvRect(*uv, jsonPath(path, "uv_rect"));
    }
    style.repeat = optionalBool(value, "repeat", path, false);

    const JsonValue* texture =
        findAny(value, {"texture", "texture_path", "texture_file", "image"});
    if (texture != nullptr) {
        std::string texturePath;
        if (texture->isString()) {
            texturePath = texture->asString();
        } else if (texture->isObject()) {
            const JsonValue& file =
                *requireField(*texture, "path", jsonPath(path, "texture"));
            if (!file.isString()) {
                fail(jsonPath(jsonPath(path, "texture"), "path"), "must be a string");
            }
            texturePath = file.asString();
            if (const JsonValue* nestedUv =
                    findAny(*texture, {"uv_rect", "uvRect", "uv"})) {
                style.uv = parseUvRect(*nestedUv,
                                       jsonPath(jsonPath(path, "texture"), "uv_rect"));
            }
            style.repeat =
                optionalBool(*texture, "repeat", jsonPath(path, "texture"), style.repeat);
        } else {
            fail(jsonPath(path, "texture"), "must be a path string or an object");
        }
        if (texturePath.empty()) {
            fail(jsonPath(path, "texture"), "texture path cannot be empty");
        }
        std::filesystem::path resolved(texturePath);
        if (resolved.is_relative()) {
            resolved = std::filesystem::path(scene.sourceDirectory) / resolved;
        }
        style.textureKey = std::filesystem::absolute(resolved).lexically_normal().string();
    }
    return style;
}

StrokeStyle parseStrokeStyle(const JsonValue& value, const std::string& path,
                             double fallbackWidth) {
    StrokeStyle style;
    style.width = fallbackWidth;
    if (value.isArray() || value.isString()) {
        style.color = parseColor(value, path);
        return style;
    }
    if (!value.isObject()) {
        fail(path, "stroke must be a color or an object");
    }
    if (const JsonValue* width = findAny(value, {"width", "stroke_width", "line_width"})) {
        style.width = requireNumber(*width, jsonPath(path, "width"));
    }
    const JsonValue* color = findAny(value, {"color", "tint"});
    if (color != nullptr) {
        style.color = parseColor(*color, jsonPath(path, "color"));
    }
    style.opacity = parseOpacity(value, path, 1.0);
    style.cap = parseCap(value, path);
    style.join = parseJoin(value, path);
    if (const JsonValue* limit = findAny(value, {"miter_limit", "miterLimit"})) {
        style.miterLimit = requireNumber(*limit, jsonPath(path, "miter_limit"));
    }
    return style;
}

Affine parseAffine(const JsonValue& value, const std::string& path) {
    if (value.isArray()) {
        const auto& array = value.asArray();
        if (array.size() == 6) {
            Affine transform;
            transform.a = requireNumber(array[0], jsonPath(path, 0));
            transform.b = requireNumber(array[1], jsonPath(path, 1));
            transform.c = requireNumber(array[2], jsonPath(path, 2));
            transform.d = requireNumber(array[3], jsonPath(path, 3));
            transform.e = requireNumber(array[4], jsonPath(path, 4));
            transform.f = requireNumber(array[5], jsonPath(path, 5));
            return transform;
        }
        fail(path, "matrix must contain six numbers");
    }
    if (!value.isObject()) {
        fail(path, "must be a transform object or six-number matrix");
    }

    Vec2 origin{0.0, 0.0};
    if (const JsonValue* pivot =
            findAny(value, {"origin", "pivot", "anchor", "base_point"})) {
        origin = parseVec2(*pivot, jsonPath(path, "origin"));
    }
    Vec2 translation{0.0, 0.0};
    if (const JsonValue* translate =
            findAny(value, {"translate", "translation", "offset"})) {
        translation = parseVec2(*translate, jsonPath(path, "translate"));
    }
    double rotation = 0.0;
    if (const JsonValue* rotate =
            findAny(value, {"rotate", "rotation", "rotation_degrees", "angle"})) {
        rotation = requireNumber(*rotate, jsonPath(path, "rotate"));
    }
    double scaleX = 1.0;
    double scaleY = 1.0;
    if (const JsonValue* scale = findAny(value, {"scale", "scaling"})) {
        if (scale->isNumber()) {
            scaleX = scaleY = requireNumber(*scale, jsonPath(path, "scale"));
        } else {
            const Vec2 parsed = parseVec2(*scale, jsonPath(path, "scale"));
            scaleX = parsed.x;
            scaleY = parsed.y;
        }
    }
    if (scaleX == 0.0 || scaleY == 0.0) {
        fail(jsonPath(path, "scale"), "scale components must be non-zero");
    }
    const double radians = rotation * 3.141592653589793238462643383279502884 / 180.0;
    const double cosine = std::cos(radians);
    const double sine = std::sin(radians);

    Affine translateToOrigin{1.0, 0.0, 0.0, 1.0, -origin.x, -origin.y};
    Affine scaling{scaleX, 0.0, 0.0, scaleY, 0.0, 0.0};
    Affine rotating{cosine, sine, -sine, cosine, 0.0, 0.0};
    Affine translateBack{1.0, 0.0, 0.0, 1.0, origin.x + translation.x,
                         origin.y + translation.y};
    return translateBack * rotating * scaling * translateToOrigin;
}

std::vector<std::vector<Vec2>> parsePolygonRings(const JsonValue& object,
                                                const std::string& base) {
    std::vector<std::vector<Vec2>> rings;
    const JsonValue* ringsValue = findAny(object, {"rings", "polygons"});
    if (ringsValue != nullptr) {
        if (!ringsValue->isArray()) {
            fail(jsonPath(base, "rings"), "must be an array of point rings");
        }
        for (std::size_t i = 0; i < ringsValue->asArray().size(); ++i) {
            rings.push_back(parsePoints(ringsValue->asArray()[i],
                                        jsonPath(jsonPath(base, "rings"), i), 3));
        }
    } else {
        const JsonValue* points = findAny(object, {"points", "coordinates", "path"});
        if (points == nullptr) {
            fail(jsonPath(base, "points"), "required point array is missing");
        }
        if (points->isArray() && !points->asArray().empty() &&
            points->asArray().front().isArray()) {
            const auto& array = points->asArray();
            const bool nested =
                !array.front().asArray().empty() && array.front().asArray().front().isArray();
            if (nested) {
                for (std::size_t i = 0; i < array.size(); ++i) {
                    rings.push_back(parsePoints(array[i],
                                                jsonPath(jsonPath(base, "points"), i), 3));
                }
            } else {
                rings.push_back(parsePoints(*points, jsonPath(base, "points"), 3));
            }
        } else {
            rings.push_back(parsePoints(*points, jsonPath(base, "points"), 3));
        }
    }
    const JsonValue* holes = object.find("holes");
    if (holes != nullptr) {
        if (!holes->isArray()) {
            fail(jsonPath(base, "holes"), "must be an array of point rings");
        }
        for (std::size_t i = 0; i < holes->asArray().size(); ++i) {
            rings.push_back(parsePoints(holes->asArray()[i],
                                        jsonPath(jsonPath(base, "holes"), i), 3));
        }
    }
    if (rings.empty()) {
        fail(jsonPath(base, "rings"), "polygon must contain at least one ring");
    }
    return rings;
}

Primitive parsePrimitive(const JsonValue& value, const std::string& path,
                         FillRule layerFillRule, const Scene& scene) {
    if (!value.isObject()) {
        fail(path, "primitive must be an object");
    }
    const JsonValue& typeValue = *requireField(value, "type", path);
    if (!typeValue.isString()) {
        fail(jsonPath(path, "type"), "must be a string");
    }
    std::string type = typeValue.asString();
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    Primitive primitive;
    primitive.fillRule = parseFillRule(value, path, layerFillRule);
    if (type == "polygon" || type == "fill" || type == "shape" || type == "ring") {
        primitive.kind = PrimitiveKind::Polygon;
        primitive.rings = parsePolygonRings(value, path);
        const JsonValue* fill = findAny(value, {"fill", "material", "paint"});
        if (fill != nullptr && !(fill->isBool() && !fill->asBool())) {
            if (fill->isBool() && fill->asBool()) {
                primitive.fill = FillStyle{};
            } else {
                primitive.fill =
                    parseFillStyle(*fill, jsonPath(path, "fill"), scene);
            }
        } else if (fill == nullptr) {
            const JsonValue* color = findAny(value, {"color", "fill_color"});
            if (color != nullptr) {
                FillStyle style;
                style.color = parseColor(*color, jsonPath(path, "color"));
                style.opacity = parseOpacity(value, path, 1.0);
                if (const JsonValue* texture =
                        findAny(value, {"texture", "texture_path", "texture_file"})) {
                    std::map<std::string, JsonValue> textureFields;
                    if (texture->isString()) {
                        textureFields.emplace("texture", *texture);
                    } else if (texture->isObject()) {
                        textureFields.emplace("texture", *texture);
                    } else {
                        fail(jsonPath(path, "texture"),
                             "must be a path string or an object");
                    }
                    FillStyle textured =
                        parseFillStyle(JsonValue(std::move(textureFields)),
                                       jsonPath(path, "texture"), scene);
                    style.textureKey = textured.textureKey;
                    style.uv = textured.uv;
                    style.repeat = textured.repeat;
                }
                primitive.fill = style;
            }
        }
        if (const JsonValue* stroke = findAny(value, {"stroke", "outline"})) {
            if (!(stroke->isBool() && !stroke->asBool())) {
                double width = 1.0;
                if (const JsonValue* strokeWidth =
                        findAny(value, {"stroke_width", "line_width"})) {
                    width = requireNumber(*strokeWidth, jsonPath(path, "stroke_width"));
                }
                if (stroke->isBool() && stroke->asBool()) {
                    primitive.stroke = StrokeStyle{};
                    primitive.stroke->width = width;
                } else {
                    primitive.stroke =
                        parseStrokeStyle(*stroke, jsonPath(path, "stroke"), width);
                }
            }
        }
        if (!primitive.fill && !primitive.stroke) {
            fail(path, "polygon requires a fill or stroke style");
        }
    } else if (type == "polyline" || type == "line" || type == "stroke" ||
               type == "path") {
        primitive.kind = PrimitiveKind::Polyline;
        const JsonValue* points = findAny(value, {"points", "coordinates", "path"});
        if (points == nullptr) {
            fail(jsonPath(path, "points"), "required point array is missing");
        }
        primitive.points = parsePoints(*points, jsonPath(path, "points"), 2);
        const JsonValue* stroke = findAny(value, {"stroke", "style"});
        double width = 1.0;
        if (const JsonValue* strokeWidth =
                findAny(value, {"stroke_width", "line_width", "width"})) {
            width = requireNumber(*strokeWidth, jsonPath(path, "stroke_width"));
        }
        if (stroke == nullptr) {
            const JsonValue* color = findAny(value, {"color", "stroke_color"});
            if (color == nullptr) {
                fail(path, "polyline requires a stroke style");
            }
            primitive.stroke = StrokeStyle{};
            primitive.stroke->width = width;
            primitive.stroke->color = parseColor(*color, jsonPath(path, "color"));
            primitive.stroke->opacity = parseOpacity(value, path, 1.0);
        } else {
            if (stroke->isBool() && stroke->asBool()) {
                primitive.stroke = StrokeStyle{};
                primitive.stroke->width = width;
            } else {
                primitive.stroke =
                    parseStrokeStyle(*stroke, jsonPath(path, "stroke"), width);
            }
        }
    } else if (type == "circle" || type == "ellipse") {
        primitive.kind = PrimitiveKind::Circle;
        const JsonValue* center = findAny(value, {"center", "origin"});
        if (center == nullptr) {
            fail(jsonPath(path, "center"), "required field is missing");
        }
        primitive.center = parseVec2(*center, jsonPath(path, "center"));
        const JsonValue& radius = *requireField(value, "radius", path);
        primitive.radius = requireNumber(radius, jsonPath(path, "radius"));
        if (primitive.radius <= 0.0) {
            fail(jsonPath(path, "radius"), "must be greater than zero");
        }
        const JsonValue* fill = findAny(value, {"fill", "material"});
        if (fill != nullptr && !(fill->isBool() && !fill->asBool())) {
            if (fill->isBool() && fill->asBool()) {
                primitive.fill = FillStyle{};
            } else {
                primitive.fill =
                    parseFillStyle(*fill, jsonPath(path, "fill"), scene);
            }
        }
        if (const JsonValue* stroke = findAny(value, {"stroke", "outline"})) {
            if (!(stroke->isBool() && !stroke->asBool())) {
                if (stroke->isBool() && stroke->asBool()) {
                    primitive.stroke = StrokeStyle{};
                } else {
                    primitive.stroke =
                        parseStrokeStyle(*stroke, jsonPath(path, "stroke"), 1.0);
                }
            }
        }
        if (!primitive.fill && !primitive.stroke) {
            fail(path, "circle requires a fill or stroke style");
        }
    } else {
        fail(jsonPath(path, "type"), "unsupported primitive type '" + type + "'");
    }

    auto validateStyle = [&path](const std::optional<FillStyle>& fill,
                                 const std::optional<StrokeStyle>& stroke,
                                 const std::string& stylePath) {
        if (fill && fill->opacity < 0.0) {
            fail(jsonPath(stylePath, "fill.opacity"), "opacity cannot be negative");
        }
        if (stroke) {
            if (!std::isfinite(stroke->width) || stroke->width < 0.0) {
                fail(jsonPath(stylePath, "stroke.width"),
                     "stroke width must be finite and non-negative");
            }
            if (!std::isfinite(stroke->miterLimit) || stroke->miterLimit < 1.0) {
                fail(jsonPath(stylePath, "stroke.miter_limit"),
                     "miter limit must be finite and at least 1");
            }
            if (!std::isfinite(stroke->opacity) || stroke->opacity < 0.0 ||
                stroke->opacity > 1.0) {
                fail(jsonPath(stylePath, "stroke.opacity"),
                     "stroke opacity must be in [0, 1]");
            }
        }
    };
    validateStyle(primitive.fill, primitive.stroke, path);
    return primitive;
}

std::string parseLayerId(const JsonValue& layer, const std::string& path) {
    const JsonValue* id = findAny(layer, {"id", "layer_id", "name"});
    if (id == nullptr) {
        fail(jsonPath(path, "id"), "required layer ID is missing");
    }
    if (!id->isString() || id->asString().empty()) {
        fail(jsonPath(path, "id"), "must be a non-empty string");
    }
    return id->asString();
}

void loadTextures(Scene& scene, std::map<std::string, std::string>& errorPaths) {
    for (const auto& layer : scene.layers) {
        for (const auto& primitive : layer.primitives) {
            if (!primitive.fill || primitive.fill->textureKey.empty()) {
                continue;
            }
            const std::string& key = primitive.fill->textureKey;
            if (scene.textures.find(key) != scene.textures.end()) {
                continue;
            }
            try {
                int width = 0;
                int height = 0;
                std::vector<Rgba8> pixels = decodePngRgbaFile(key, width, height);
                Texture texture;
                texture.width = width;
                texture.height = height;
                texture.pixels = std::move(pixels);
                scene.textures.emplace(key, std::move(texture));
            } catch (const std::exception& error) {
                std::string path = errorPaths.count(key) != 0
                                       ? errorPaths[key]
                                       : "$.layers";
                fail(path, "cannot load texture '" + key + "': " + error.what());
            }
        }
    }
}

}  // namespace

Vec2 Affine::apply(Vec2 point) const {
    return {a * point.x + c * point.y + e, b * point.x + d * point.y + f};
}

Affine operator*(const Affine& left, const Affine& right) {
    Affine result;
    result.a = left.a * right.a + left.c * right.b;
    result.b = left.b * right.a + left.d * right.b;
    result.c = left.a * right.c + left.c * right.d;
    result.d = left.b * right.c + left.d * right.d;
    result.e = left.a * right.e + left.c * right.f + left.e;
    result.f = left.b * right.e + left.d * right.f + left.f;
    return result;
}

SceneError::SceneError(std::string path, const std::string& message)
    : std::runtime_error(path + ": " + message), path_(std::move(path)) {}

Color premultiplyColor(const Color& color, double opacity) {
    const double alpha = std::clamp(color.a * opacity, 0.0, 1.0);
    return {color.r * alpha, color.g * alpha, color.b * alpha, alpha};
}

Scene loadScene(const std::string& path) {
    const std::string text = readTextFile(path);
    JsonParser parser;
    JsonValue root;
    try {
        root = parser.parse(text);
    } catch (const JsonError& error) {
        throw SceneError("$", error.what());
    }
    if (!root.isObject()) {
        fail("$", "scene root must be a JSON object");
    }

    Scene scene;
    scene.sourcePath = std::filesystem::absolute(path).lexically_normal().string();
    scene.sourceDirectory =
        std::filesystem::path(scene.sourcePath).parent_path().string();
    scene.sourceCrc32 = crc32(reinterpret_cast<const std::uint8_t*>(text.data()),
                              text.size());
    scene.sourceBytes = text.size();

    const JsonValue* widthValue = root.find("width");
    const JsonValue* heightValue = root.find("height");
    const JsonValue* canvas = findAny(root, {"canvas", "resolution", "size"});
    if (canvas != nullptr) {
        if (canvas->isArray()) {
            if (canvas->asArray().size() != 2) {
                fail(jsonPath("$", "canvas"), "must contain [width, height]");
            }
            widthValue = &canvas->asArray()[0];
            heightValue = &canvas->asArray()[1];
        } else if (canvas->isObject()) {
            widthValue = canvas->find("width");
            heightValue = canvas->find("height");
        } else {
            fail(jsonPath("$", "canvas"), "must be an object or [width, height]");
        }
    }
    if (widthValue == nullptr) {
        fail("$.width", "required scene width is missing");
    }
    if (heightValue == nullptr) {
        fail("$.height", "required scene height is missing");
    }
    const double width = requireNumber(*widthValue, "$.width");
    const double height = requireNumber(*heightValue, "$.height");
    if (width <= 0.0 || height <= 0.0 || width > 65536.0 || height > 65536.0 ||
        std::floor(width) != width || std::floor(height) != height) {
        fail("$.width", "scene width and height must be positive integers <= 65536");
    }
    scene.width = static_cast<int>(width);
    scene.height = static_cast<int>(height);

    if (const JsonValue* background =
            findAny(root, {"background", "background_color"})) {
        scene.background = parseColor(*background, "$.background");
    }

    const JsonValue* layersValue = root.find("layers");
    if (layersValue == nullptr) {
        fail("$.layers", "required layers array is missing");
    }
    if (!layersValue->isArray()) {
        fail("$.layers", "must be an array");
    }
    if (layersValue->asArray().empty()) {
        fail("$.layers", "at least one layer is required");
    }

    std::map<std::string, std::string> texturePaths;
    std::map<std::string, std::string> seenIds;
    for (std::size_t layerIndex = 0; layerIndex < layersValue->asArray().size();
         ++layerIndex) {
        const std::string layerPath = jsonPath("$.layers", layerIndex);
        const JsonValue& layerValue = layersValue->asArray()[layerIndex];
        if (!layerValue.isObject()) {
            fail(layerPath, "layer must be an object");
        }
        Layer layer;
        layer.id = parseLayerId(layerValue, layerPath);
        if (seenIds.find(layer.id) != seenIds.end()) {
            fail(jsonPath(layerPath, "id"), "duplicate layer ID '" + layer.id + "'");
        }
        seenIds.emplace(layer.id, layerPath);
        layer.visible = optionalBool(layerValue, "visible", layerPath, true);
        layer.opacity = parseOpacity(layerValue, layerPath, 1.0);
        layer.blend = parseBlendMode(layerValue, layerPath);
        layer.fillRule = parseFillRule(layerValue, layerPath, FillRule::NonZero);
        if (const JsonValue* transform =
                findAny(layerValue, {"transform", "matrix"})) {
            layer.transform = parseAffine(*transform, jsonPath(layerPath, "transform"));
        } else if (findAny(layerValue, {"origin", "pivot", "anchor", "translate",
                                        "translation", "rotate", "rotation",
                                        "scale", "scaling"}) != nullptr) {
            std::map<std::string, JsonValue> fields;
            for (const char* key :
                 {"origin", "pivot", "anchor", "translate", "translation",
                  "rotate", "rotation", "scale", "scaling"}) {
                if (const JsonValue* value = layerValue.find(key)) {
                    fields.emplace(key, *value);
                }
            }
            layer.transform = parseAffine(
                JsonValue(std::move(fields)), jsonPath(layerPath, "transform"));
        }

        const JsonValue* primitives =
            findAny(layerValue, {"primitives", "geometry", "shapes", "elements", "paths"});
        if (primitives == nullptr) {
            fail(jsonPath(layerPath, "primitives"),
                 "required geometry array is missing");
        }
        if (!primitives->isArray()) {
            fail(jsonPath(layerPath, "primitives"), "must be an array");
        }
        if (primitives->asArray().empty()) {
            fail(jsonPath(layerPath, "primitives"), "must not be empty");
        }
        for (std::size_t primitiveIndex = 0;
             primitiveIndex < primitives->asArray().size(); ++primitiveIndex) {
            const std::string primitivePath =
                jsonPath(jsonPath(layerPath, "primitives"), primitiveIndex);
            Primitive primitive = parsePrimitive(
                primitives->asArray()[primitiveIndex], primitivePath, layer.fillRule,
                scene);
            if (primitive.fill && !primitive.fill->textureKey.empty()) {
                texturePaths.emplace(primitive.fill->textureKey,
                                     jsonPath(primitivePath, "fill.texture"));
            }
            layer.primitives.push_back(std::move(primitive));
        }
        scene.layers.push_back(std::move(layer));
    }

    loadTextures(scene, texturePaths);
    return scene;
}

}  // namespace tilegen
