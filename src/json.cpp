#include "skinbake/json.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace skinbake {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned>(ch)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

std::string json_string(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string json_u64(std::uint64_t value) {
    return std::to_string(value);
}

std::string json_i64(std::int64_t value) {
    return std::to_string(value);
}

std::string json_double(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}

std::string json_bool(bool value) {
    return value ? "true" : "false";
}

std::string json_object(const std::vector<JsonField>& fields) {
    std::ostringstream out;
    out << '{';
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << json_string(fields[i].first) << ':' << fields[i].second;
    }
    out << '}';
    return out.str();
}

}

