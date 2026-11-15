#pragma once

#include "automix/wav_io.h"

namespace automix {

class BandlimitedResampler {
public:
    AudioData convert(const AudioData& input, int targetSampleRate) const;

    static double passbandLimitForRates(int sourceRate, int targetRate);
};

}  // namespace automix
