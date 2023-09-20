#ifndef VULKAN_LLAMA_LLAVA_SERVER_H
#define VULKAN_LLAMA_LLAVA_SERVER_H

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

class llava_server {
public:
    explicit llava_server(llava_context* context);
    void serve_forever();
    void create_session(client* calling_client, u32 request_code);
    session_wrapper *get_session_by_id(u32 i);
    [[nodiscard]] map<u32, session_wrapper*> const& get_sessions() const;
    llava_context* const ctx;

    void remove_client_at_position(u32 i);
    void clear_write_bit_at(u32 i);
    void set_write_bit_at(u32 i);
    [[nodiscard]] const vector<u8> &get_token_map_message() const;

    void receive_session_outbound_packets(list<tuple<u32, OutCommands, u32, vector<u8>>> &packets, list<tuple<u32, OutCommands, vector<u8>>> &broadcasts);
    void notify_session_death(u32 session_i) const;
    void add_subscription(u32 client_id, u32 session_id);
    void drop_subscription(u32 client_id, u32 session_id);

private:
    const pair<int, int> ping_pipes;
    vector<u8> tokenMapMessage;
    vector<pollfd> pollfdArray;
    map<u32, client*> clients;
    map<u32, session_wrapper*> sessions;
    map<u32, client*> client_by_client_id;

private:
    map<u32, set<u32>> clients_subscriptions;
    map<u32, set<u32>> sessions_subscribers;

private:
    // When messages from session running in threads are sent, they are put here
    mutex session_messages_mutex;
    list<tuple<u32, OutCommands, u32, vector<u8>>> session_messages;
    list<tuple<u32, OutCommands, vector<u8>>> broadcast_messages;
};
}

#endif
