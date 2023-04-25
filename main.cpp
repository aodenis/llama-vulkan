#include "llava_context.h"

int main(int argc, char **argv) {
    shared_ptr<llava_context> ctx = make_shared<llava_context>();
    ctx->run(argc, argv);
}
