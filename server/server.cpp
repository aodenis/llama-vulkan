#include "../llava_context.h"
#include "../llava_session.h"
#include "server.h"
#include "client.h"
#include "session_wrapper.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/poll.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <list>
#include <cassert>
#include <utility>
#include <csignal>

using namespace lsrv;

pair<int, int> mk_pipes() {
    int pipes[2];
    assert (pipe(pipes) >= 0);
    return {pipes[0], pipes[1]};
}

llava_server::llava_server(llava_context *context) : ctx(context), ping_pipes(mk_pipes()) {
    u32 finalSize = 0;
    for (auto& token : ctx->get_model()->get_tokens()) {
        finalSize += 4 + token.text.size();
    }
    tokenMapMessage.resize(finalSize);
    u32 cursor = 0;
    for (auto& token : ctx->get_model()->get_tokens()) {
        u32 sz = token.text.size();
        memcpy(tokenMapMessage.data() + cursor, &sz, 4);
        cursor += 4;
        memcpy(tokenMapMessage.data() + cursor, token.text.data(), token.text.size());
        cursor += token.text.size();
    }
}

void llava_server::serve_forever() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("socket");
        return;
    }

    const int enable = 1;

    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0)
    {
        perror("setsockopt(SO_REUSEADDR)");
        close(server_socket);
        return;
    }

    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int)) < 0)
    {
        perror("setsockopt(SO_REUSEPORT)");
        close(server_socket);
        return;
    }

    sockaddr_in addr {};
    addr.sin_port = htons(1337);
    addr.sin_family = AF_INET;
    inet_pton(AF_INET, "0.0.0.0", &addr.sin_addr);

    if(bind(server_socket, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_socket);
        return;
    }

    if (listen(server_socket, 5) < 0) {
        perror("listen");
        close(server_socket);
        return;
    }

    int sigfd = ctx->get_signal_fd();

    cout << "[*] Started server on 0.0.0.0:1337" << endl;

    pollfdArray.emplace_back() = {
            .fd = server_socket,
            .events = POLLIN,
            .revents = 0
    };

    if (sigfd != -1) {
        pollfdArray.emplace_back() = {
                .fd = sigfd,
                .events = POLLIN,
                .revents = 0
        };
    }
    pollfdArray.emplace_back() = {
            .fd = STDIN_FILENO,
            .events = POLLIN,
            .revents = 0
    };


    pollfdArray.emplace_back() = {
            .fd = ping_pipes.first,
            .events = POLLIN,
            .revents = 0
    };

    bool should_die = false;
    list<int> to_drop;
    while((not should_die) or (not sessions.empty())) {
        int pollret = poll(pollfdArray.data(), pollfdArray.size(), 1000);
        if(pollret < 0)
		{
			if(errno == EINTR)
			{
                continue;
			}
            cerr << "[?] Error while polling " << strerror(errno) << endl;
            return;
		}
        if (pollret == 0) {
            continue;
        }

        u32 dead_session_id = 0;
        u32 pollfdArraySize = pollfdArray.size();
        for(u32 i = 0; (i < pollfdArraySize) and pollret; ++i)
        {
            if(not pollfdArray.at(i).revents) {
                continue;
            }
            --pollret;
            short revents = pollfdArray.at(i).revents;
            pollfdArray.at(i).revents = 0;
            if (pollfdArray.at(i).fd == server_socket)
            {
                // New client
                sockaddr_in new_peer_addr {};
                socklen_t len = sizeof(sockaddr_in);

                int new_fd = accept(server_socket, (sockaddr*)&new_peer_addr, &len);
                char address[INET_ADDRSTRLEN + 1];
                cout << "[*] New client from " << inet_ntop(new_peer_addr.sin_family, &new_peer_addr.sin_addr, address, len) << ":" << new_peer_addr.sin_port << endl;

                if (new_fd >= 0) {
                    // Add client to the structure
                    pollfdArray.emplace_back() = {
                        .fd = new_fd,
                        .events = POLLIN,
                        .revents = 0
                    };
                    auto* new_obj = new client(this, new_fd, pollfdArray.size() - 1);
                    assert(clients.emplace(new_fd, new_obj).second);
                    assert(client_by_client_id.emplace(new_obj->client_id, new_obj).second);
                } else {
                    perror("accept");
                }
                continue;
            }

            if (pollfdArray.at(i).fd == sigfd) {
                // Signal handler
                u32 sig = ctx->pop_signal(false);
                // cout << "[*] Signal caught (" << sig << ")" << endl;
                should_die = true;
                for (auto& [_, session] : sessions) {
                    session->push_order(0, 0, cmdKillSession);
                }
                continue;
            }

            if (pollfdArray.at(i).fd == STDIN_FILENO) {
                should_die = true;
                for (auto& [_, session] : sessions) {
                    session->push_order(0, 0, cmdKillSession);
                }
                continue;
            }

            if (pollfdArray.at(i).fd == ping_pipes.first) {
                // This is meant to flush session messages
                assert(read(ping_pipes.first, &dead_session_id, 4) == 4); // Dismiss it
                lock_guard guard(session_messages_mutex);
                for ( ; not broadcast_messages.empty(); broadcast_messages.pop_front()) {
                    auto& [session_id, packet_type, packet_content] = broadcast_messages.front();
                    for (auto client_id : clients_subscriptions[session_id]) {
                        auto it = client_by_client_id.find(client_id);
                        if (it == client_by_client_id.end()) {
                            cerr << "Leftover subscriber???" << endl;
                        } else {
                            it->second->add_out_packet(packet_type, 0, packet_content);
                        }
                    }
                }
                for ( ; not session_messages.empty(); session_messages.pop_front()) {
                    auto& [client_id, packet_type, reply_code, packet_content] = session_messages.front();
                    auto it = client_by_client_id.find(client_id);
                    if (it == client_by_client_id.end()) {
                        cerr << "Message from session to an unknown client???" << endl;
                    } else {
                        it->second->add_out_packet(packet_type, reply_code, packet_content);
                    }
                }
                continue;
            }

            if (auto it = clients.find(pollfdArray.at(i).fd); it != clients.end()) {
                // Client event
                if (not it->second->on_polled(revents)) {
                    to_drop.emplace_back(it->second->client_id);
                }
                continue;
            }

            cerr << "[?] Unknown fd received an event " << pollfdArray.at(i).fd << endl;
        }

        if (dead_session_id) {
            auto it = sessions.find(dead_session_id);
            assert(it != sessions.end());
            it->second->join();
            for (auto subscriber_id : sessions_subscribers[dead_session_id]) {
                clients_subscriptions[subscriber_id].erase(dead_session_id);
            }
            sessions_subscribers.erase(dead_session_id);
            delete it->second;
            sessions.erase(it);
        }

        for (auto client_id : to_drop) {
            auto it = client_by_client_id.find(client_id);
            assert(it != client_by_client_id.end());
            for (auto session_id : sessions_subscribers[client_id]) {
                sessions_subscribers[session_id].erase(client_id);
            }
            clients_subscriptions.erase(client_id);
            clients.erase(it->second->fd);
            delete it->second;
            client_by_client_id.erase(it);
        }
        to_drop.clear();
    }

    while (not clients.empty()) {
        client_by_client_id.erase(clients.begin()->second->client_id);
        delete clients.begin()->second;
        clients.erase(clients.begin());
    }

    close(server_socket);
}

void llava_server::create_session(client* calling_client, u32 request_code) {
    auto* ns = new session_wrapper(this);
    sessions.emplace(ns->session_id, ns);
    for (auto& [_, client] : clients) {
        client->add_new_session_to_queue((client == calling_client) ? request_code : 0, ns->session_id);
    }
    ns->start();
}

session_wrapper *llava_server::get_session_by_id(u32 i) {
    auto jt = sessions.find(i);
    if (jt != sessions.end()) {
        return jt->second;
    }
    return nullptr;
}

void llava_server::remove_client_at_position(u32 position) {
    if (position + 1 == pollfdArray.size()) {
        pollfdArray.pop_back();
        return;
    }
    pollfdArray.at(position) = pollfdArray.back();
    pollfdArray.pop_back();
    clients.at(pollfdArray.at(position).fd)->position = position;
}

map<u32, session_wrapper *> const &llava_server::get_sessions() const {
    return sessions;
}

void llava_server::clear_write_bit_at(u32 position) {
    pollfdArray.at(position).events &= (~POLLOUT);
}

void llava_server::set_write_bit_at(u32 position) {
    pollfdArray.at(position).events |= POLLOUT;
}

const vector<u8> &llava_server::get_token_map_message() const {
    return tokenMapMessage;
}

void llava_server::notify_session_death(u32 session_id) const {
    if (write(ping_pipes.second, &session_id, 4) != 4) {
        int errsv = errno;
        cerr << "Ping (death) failed, errno=" << strerror(errsv) << endl;
    }
}

void llava_server::add_subscription(u32 client_id, u32 session_id) {
    clients_subscriptions[client_id].insert(session_id);
    sessions_subscribers[session_id].insert(client_id);
}

void llava_server::drop_subscription(u32 client_id, u32 session_id) {
    clients_subscriptions[client_id].erase(session_id);
    sessions_subscribers[session_id].erase(client_id);
}

void llava_server::receive_session_outbound_packets(list<tuple<u32, OutCommands, u32, vector<u8>>> &packets, list<tuple<u32, OutCommands, vector<u8>>> &broadcasts) {
    {
        lock_guard lock(session_messages_mutex);
        broadcast_messages.splice(broadcast_messages.end(), broadcasts);
        session_messages.splice(session_messages.end(), packets);
    }

    int zero = 0;
    if(write(ping_pipes.second, &zero, 4) != 4) {
        int errsv = errno;
        cerr << "Ping failed, errno=" << strerror(errsv) << endl;
    }
}
