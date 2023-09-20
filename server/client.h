
#ifndef VULKAN_LLAMA_SERVER_CLIENT_H
#define VULKAN_LLAMA_SERVER_CLIENT_H

#include "types.h"
#include <set>
#include <string>
#include <thread>
#include <list>
#include <vector>
#include <map>
#include <sys/poll.h>
#include <condition_variable>

namespace lsrv {
class client {
public:
    client(llava_server* server, int fd, u32 pos);
    ~client();
    bool on_polled(short revents);
    const u32 client_id;
    llava_server* const server;
    const int fd;
    u32 position;
    set<u32> subscriptions;

    void add_out_packet(OutCommands packet_type, u32 reply_code, vector<u8> const &vector);
    void add_new_session_to_queue(u32 request_code, u32 session_id);

private:
    vector<u8> input_buffer;
    list<vector<u8>> outbound_buffer;
    bool pop_message();
    bool is_failed = false;
    void ack(u32 req_id, ReturnCode result);

private:
    static u32 ticker;
    void process_message(tlv_header *header, void *packet_ptr);
};
}

#endif
