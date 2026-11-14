#include "audio_processor.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace lipsync {
namespace {

struct FormatContextGuard {
    AVFormatContext* ctx = nullptr;
    ~FormatContextGuard() {
        if (ctx) avformat_close_input(&ctx);
    }
};

struct CodecContextGuard {
    AVCodecContext* ctx = nullptr;
    ~CodecContextGuard() {
        if (ctx) avcodec_free_context(&ctx);
    }
};

struct SwrGuard {
    SwrContext* ctx = nullptr;
    ~SwrGuard() {
        if (ctx) swr_free(&ctx);
    }
};

void put_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void put_u16_le(std::vector<uint8_t>& buf, uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

// 手写 44 字节标准 PCM16 WAV，避免依赖外部 muxer 行为，结果确定。
bool write_wav_pcm16(const std::string& path,
                     int sample_rate,
                     int channels,
                     const std::vector<uint8_t>& pcm,
                     std::string& error_message) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error_message = "cannot open output wav for writing: " + path;
        return false;
    }
    std::vector<uint8_t> header;
    header.reserve(44);
    const uint32_t data_size = static_cast<uint32_t>(pcm.size());
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;

    header.insert(header.end(), {'R', 'I', 'F', 'F'});
    put_u32_le(header, 36u + data_size);
    header.insert(header.end(), {'W', 'A', 'V', 'E'});
    header.insert(header.end(), {'f', 'm', 't', ' '});
    put_u32_le(header, 16u);
    put_u16_le(header, 1u);                 // PCM
    put_u16_le(header, static_cast<uint16_t>(channels));
    put_u32_le(header, static_cast<uint32_t>(sample_rate));
    put_u32_le(header, byte_rate);
    put_u16_le(header, block_align);
    put_u16_le(header, 16u);                // bits per sample
    header.insert(header.end(), {'d', 'a', 't', 'a'});
    put_u32_le(header, data_size);

    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(pcm.data()),
              static_cast<std::streamsize>(pcm.size()));
    if (!out) {
        error_message = "failed writing wav data: " + path;
        return false;
    }
    return true;
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                                 static_cast<uint16_t>(p[1] << 8));
}

} // namespace

ProcessError process_audio(const std::string& input_path,
                           const std::string& output_wav_path,
                           int target_sample_rate,
                           int target_channels,
                           AudioResult& result) {
    FormatContextGuard fmt_guard;
    AVFormatContext* fmt = nullptr;
    int ret = avformat_open_input(&fmt, input_path.c_str(), nullptr, nullptr);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE,
                            "avformat_open_input failed for " + input_path);
    }
    fmt_guard.ctx = fmt;

    ret = avformat_find_stream_info(fmt, nullptr);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "avformat_find_stream_info failed");
    }

    int stream_index = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "no audio stream found in " + input_path);
    }
    AVStream* stream = fmt->streams[stream_index];
    AVCodecParameters* par = stream->codecpar;

    const AVCodec* decoder = avcodec_find_decoder(par->codec_id);
    if (!decoder) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "decoder not available for codec id " +
                                                          std::to_string(par->codec_id));
    }
    CodecContextGuard codec_guard;
    codec_guard.ctx = avcodec_alloc_context3(decoder);
    if (!codec_guard.ctx) {
        return ProcessError(ErrorCode::E_INTERNAL, "avcodec_alloc_context3 failed");
    }
    ret = avcodec_parameters_to_context(codec_guard.ctx, par);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "avcodec_parameters_to_context failed");
    }
    ret = avcodec_open2(codec_guard.ctx, decoder, nullptr);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "avcodec_open2 failed");
    }

    const int source_sample_rate = codec_guard.ctx->sample_rate;
    const int source_channels = codec_guard.ctx->ch_layout.nb_channels;
    if (source_sample_rate <= 0 || source_channels <= 0) {
        return ProcessError(ErrorCode::E_AUDIO_DECODE, "invalid source audio parameters");
    }

    AVChannelLayout out_layout;
    if (target_channels == 2) {
        out_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else {
        out_layout = AV_CHANNEL_LAYOUT_MONO;
    }

    SwrGuard swr_guard;
    SwrContext* swr = nullptr;
    ret = swr_alloc_set_opts2(&swr,
                              &out_layout, AV_SAMPLE_FMT_S16, target_sample_rate,
                              &codec_guard.ctx->ch_layout,
                              codec_guard.ctx->sample_fmt, source_sample_rate,
                              0, nullptr);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_RESAMPLE, "swr_alloc_set_opts2 failed");
    }
    swr_guard.ctx = swr;
    ret = swr_init(swr);
    if (ret < 0) {
        return ProcessError(ErrorCode::E_RESAMPLE, "swr_init failed");
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) {
        av_packet_free(&packet);
        av_frame_free(&frame);
        return ProcessError(ErrorCode::E_INTERNAL, "packet/frame allocation failed");
    }

    std::vector<uint8_t> pcm_out;
    int64_t total_source_samples = 0;
    bool decode_error = false;
    std::string decode_message;

    auto convert_frame = [&](const AVFrame* input_frame) -> bool {
        const int capacity = static_cast<int>(
            av_rescale_rnd(input_frame ? input_frame->nb_samples : 0,
                           target_sample_rate, source_sample_rate, AV_ROUND_UP)) +
                             32;
        uint8_t* out_ptr = nullptr;
        int converted = av_samples_alloc(&out_ptr, nullptr, target_channels,
                                         capacity, AV_SAMPLE_FMT_S16, 0);
        if (converted < 0) {
            decode_message = "av_samples_alloc failed";
            return false;
        }
        int got;
        if (input_frame) {
            got = swr_convert(swr, &out_ptr, capacity,
                              input_frame->extended_data, input_frame->nb_samples);
        } else {
            got = swr_convert(swr, &out_ptr, capacity, nullptr, 0); // flush
        }
        if (got < 0) {
            av_freep(&out_ptr);
            decode_message = "swr_convert failed";
            return false;
        }
        const size_t byte_count = static_cast<size_t>(got) *
                                  static_cast<size_t>(target_channels) * sizeof(int16_t);
        pcm_out.insert(pcm_out.end(), out_ptr, out_ptr + byte_count);
        av_freep(&out_ptr);
        return true;
    };

    while (av_read_frame(fmt, packet) >= 0) {
        if (packet->stream_index == stream_index) {
            ret = avcodec_send_packet(codec_guard.ctx, packet);
            if (ret < 0) {
                decode_error = true;
                decode_message = "avcodec_send_packet failed";
                av_packet_unref(packet);
                break;
            }
            while (true) {
                ret = avcodec_receive_frame(codec_guard.ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) {
                    decode_error = true;
                    decode_message = "avcodec_receive_frame failed";
                    break;
                }
                total_source_samples += frame->nb_samples;
                if (!convert_frame(frame)) {
                    decode_error = true;
                    break;
                }
                av_frame_unref(frame);
            }
        }
        av_packet_unref(packet);
        if (decode_error) break;
    }

    if (!decode_error) {
        avcodec_send_packet(codec_guard.ctx, nullptr); // flush decoder
        while (true) {
            ret = avcodec_receive_frame(codec_guard.ctx, frame);
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
            if (ret < 0) {
                decode_error = true;
                decode_message = "decoder flush failed";
                break;
            }
            total_source_samples += frame->nb_samples;
            if (!convert_frame(frame)) {
                decode_error = true;
                break;
            }
            av_frame_unref(frame);
        }
    }
    if (!decode_error) {
        if (!convert_frame(nullptr)) { // flush resampler
            decode_error = true;
        }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);

    if (decode_error) {
        return ProcessError(ErrorCode::E_RESAMPLE, decode_message);
    }

    std::string io_error;
    if (!write_wav_pcm16(output_wav_path, target_sample_rate,
                         target_channels, pcm_out, io_error)) {
        return ProcessError(ErrorCode::E_IO, io_error);
    }

    double duration = static_cast<double>(total_source_samples) /
                      static_cast<double>(source_sample_rate);
    duration = std::round(duration * 1000.0) / 1000.0; // 毫秒精度

    result.output_wav_path = output_wav_path;
    result.source_duration_seconds = duration;
    result.source_sample_rate = source_sample_rate;
    result.source_channels = source_channels;
    result.target_sample_rate = target_sample_rate;
    result.target_channels = target_channels;
    return ProcessError{};
}

ProcessError read_wav_info(const std::string& wav_path, WavInfo& info) {
    std::ifstream in(wav_path, std::ios::binary);
    if (!in) {
        return ProcessError(ErrorCode::E_IO, "cannot open wav: " + wav_path);
    }
    uint8_t header[44];
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(header)) ||
        std::memcmp(header, "RIFF", 4) != 0 ||
        std::memcmp(header + 8, "WAVE", 4) != 0 ||
        std::memcmp(header + 12, "fmt ", 4) != 0) {
        return ProcessError(ErrorCode::E_IO, "not a valid RIFF/WAVE file: " + wav_path);
    }
    info.channels = read_u16_le(header + 22);
    info.sample_rate = static_cast<int>(read_u32_le(header + 24));
    info.bits_per_sample = read_u16_le(header + 34);
    info.data_bytes = read_u32_le(header + 40);
    return ProcessError{};
}

} // namespace lipsync
