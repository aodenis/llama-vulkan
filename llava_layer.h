#ifndef VULKAN_LLAMA__LLAVA_LAYER_H
#define VULKAN_LLAMA__LLAVA_LAYER_H

#include <map>
#include "types.h"

class llava_layer {
public:
    llava_layer(llava_context* context, u32 layer_id);

private:
    map<string, llava_buffer> named_buffers;
};


#endif //VULKAN_LLAMA__LLAVA_LAYER_H
