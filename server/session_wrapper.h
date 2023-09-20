#ifndef VULKAN_LLAMA_SERVER_SESSION_WRAPPER_H
#define VULKAN_LLAMA_SERVER_SESSION_WRAPPER_H

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
class session_wrapper {
public:
    explicit session_wrapper(llava_server *server);
    ~session_wrapper();
    u32 const session_id;
    llava_server *const server;
    void start();
    void push_order(u32 uid, u32 reqId, u32 opcode, u32 arg1 = 0, std::string s = {});
    void join();

private:
    void run();
    mutex session_mutex;
    thread *loop_thread = nullptr;
    condition_variable cv;
    list<session_order> orders;
    void process_orders(llava_session &session, bool blocking);
    list<tuple<u32, OutCommands, u32, vector<u8>>> outbound_packets;
    list<tuple<u32, OutCommands, vector<u8>>> outbound_broadcasts;
    void flush_outbound_packets();

private:
    bool should_run = false;
    u32 should_tick = 0;
    u32 tick_client = 0;
    bool should_die = false;

private:
    static u32 ticker;
    void setup_exceptions();

    void add_outbound_ack(u32 client_id, u32 request_code, ReturnCode return_code);
};
}

#endif
