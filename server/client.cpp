#include "client.h"
#include "server.h"
#include "session_wrapper.h"
#include "../llava_context.h"
#include "../llava_session.h"
#include "../utils.h"
#include <sys/socket.h>
#include <sys/poll.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <cassert>
#include <utility>
#include <csignal>

using namespace lsrv;

u32 client::ticker = 1U;

void client::ack(u32 request_code, ReturnCode res) {
    vector<u8> nb(4);
    memcpy(nb.data(), &res, 4);
    add_out_packet(outCmdAck, request_code, nb);
}

void client::add_new_session_to_queue(u32 request_code, u32 session_id) {
    // Main thread
    vector<u8> nb(4);
    memcpy(nb.data(), &session_id, 4);
    add_out_packet(outCmdNewSession, request_code, nb);
}

client::client(llava_server *_server, int _fd, u32 pos) : client_id(ticker++), server(_server), fd(_fd), position(pos) {

}

client::~client() {
    close(fd);
    server->remove_client_at_position(position);
}

bool client::on_polled(short revents) {
    // sync
    if (revents & (POLLERR | POLLHUP)) {
        return false;
    }

    if (revents & POLLOUT) {
        bool first_send = true;
        while (not outbound_buffer.empty()) {
            vector<u8> &to_send = outbound_buffer.front();
            long r = send(fd, to_send.data(), to_send.size(), MSG_DONTWAIT);
            if (r < 0) {
                if (errno == EAGAIN or errno == EWOULDBLOCK) {
                    break;
                } else {
                    perror("send");
                    return false;
                }
            }
            if (first_send and (r == 0)) {
                cerr << "Short write of client socket" << endl;
                return false;
            }
            first_send = false;
            if (r != to_send.size()) {
                memmove(to_send.data(), to_send.data() + r, to_send.size() - r);
                to_send.resize(to_send.size() - r);
                outbound_buffer.emplace_front(std::move(to_send));
                break;
            } else {
                outbound_buffer.pop_front();
            }
        }
        if (outbound_buffer.empty()) {
            server->clear_write_bit_at(position);
        }
    }

    if (revents & POLLIN) {
        char buf[1024];
        while (input_buffer.size() < 0x20000) {
            errno = 0;
            long r = recv(fd, buf, 1024, MSG_DONTWAIT);
            if (r < 0) {
                if (errno == EAGAIN or errno == EWOULDBLOCK) {
                    break;
                } else {
                    perror("recv");
                    return false;
                }
            } else if (r == 0) {
                if (errno != 0) {
                    int errsv = errno;
                    cerr << "Short recv of client socket " << strerror(errsv) << endl;
                }
                return false;
            } else {
                u32 current_size = input_buffer.size();
                input_buffer.resize(current_size + r);
                memcpy(input_buffer.data() + current_size, buf, r);
            }
            if (r != 1024) {
                break;
            }
        }

        while (not is_failed) {
            if (not pop_message()) {
                break;
            }
        }
    }

    return true;
}

bool client::pop_message() {
    // Main thread
    if (input_buffer.size() < sizeof(tlv_header)) {
        return false;
    }

    auto *header = (tlv_header *) (input_buffer.data());

    if (header->length > 0x10000) {
        cerr << "[!] Packet too large" << endl;
        is_failed = true;
        return false;
    }

    if (sizeof(tlv_header) + header->length > input_buffer.size()) {
        return false;
    }

    void *packet_ptr = input_buffer.data() + sizeof(tlv_header);

    process_message(header, packet_ptr);
    memmove(input_buffer.data(), input_buffer.data() + header->length + sizeof(tlv_header), input_buffer.size() - (header->length + sizeof(tlv_header)));
    input_buffer.resize(input_buffer.size() - header->length - sizeof(tlv_header));
    return true;
}

void client::process_message(tlv_header *header, void *packet_ptr) {
    if (header->command < 16) {
        if (header->command == cmdGetTokenMap) {
            vector<u8> new_buf(server->get_token_map_message());
            add_out_packet(outCmdTokenMap, header->request_id, new_buf);
        } else if (header->command == cmdNewSession) { // newSession
            server->create_session(this, header->request_id);
        } else if (header->command == cmdListSessions) { // listSessions
            vector<u8> nb(4 * server->get_sessions().size());
            u32 i = 0;
            for (auto &[k, v]: server->get_sessions()) {
                memcpy(nb.data() + (i++) * 4, &k, 4);
            }
            add_out_packet(outCmdSessionList, header->request_id, nb);
        } else {
            ack(header->request_id, ReturnCode::unknown_command);
        }
        return;
    }

    if (header->command < 32) {
        if (header->length != 4) {
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }
        u32 session = *((u32 *) packet_ptr);
        session_wrapper *sessionHandler = server->get_session_by_id(session);

        if (sessionHandler == nullptr) {
            ack(header->request_id, ReturnCode::no_such_session);
            return;
        }

        switch (header->command) {
            case cmdSubscribe:
                server->add_subscription(client_id, session);
                ack(header->request_id, ReturnCode::ok);
                break;
            case cmdUnsubscribe:
                server->drop_subscription(client_id, session);
                ack(header->request_id, ReturnCode::ok);
                break;
            case cmdGetSessionTokens:
            case cmdKillSession:
            case cmdStartSessionGeneration:
            case cmdTick:
            case cmdStopSessionGeneration:
            case cmdGetSessionStatus:
                sessionHandler->push_order(client_id, header->request_id, header->command);
                break;
            default:
                ack(header->request_id, ReturnCode::unknown_command);
        }
        return;
    }

    if (header->command == cmdRewind) {
        if (header->length != sizeof(cmdRewind_data)) {
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }
        auto *data = (cmdRewind_data *) packet_ptr;

        session_wrapper *sessionHandler = server->get_session_by_id(data->session);
        if (sessionHandler) {
            sessionHandler->push_order(client_id, header->request_id, cmdRewind, data->new_position);
        } else {
            ack(header->request_id, ReturnCode::no_such_session);
        }
        return;
    }

    if (header->command == cmdSetSessionOptions) {
        if (header->length != sizeof(cmdSetSessionOptions_data)) {
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }
        auto *data = (cmdSetSessionOptions_data *) packet_ptr;

        session_wrapper *sessionHandler = server->get_session_by_id(data->session);
        if (sessionHandler) {
            sessionHandler->push_order(client_id, header->request_id, cmdSetSessionOptions, data->options);
        } else {
            ack(header->request_id, ReturnCode::no_such_session);
        }
        return;
    }

    if (header->command == cmdAddToken) {
        if (header->length != sizeof(cmdAddToken_data)) {
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }
        auto *data = (cmdAddToken_data *) packet_ptr;

        session_wrapper *sessionHandler = server->get_session_by_id(data->session);
        if (sessionHandler) {
            sessionHandler->push_order(client_id, header->request_id, cmdAddToken, data->token_id);
        } else {
            ack(header->request_id, ReturnCode::no_such_session);
        }
        return;
    }

    if (header->command == cmdAddText) {
        if (header->length <= sizeof(cmdAddText_data)) {
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }
        auto *data = (cmdAddText_data *) packet_ptr;

        session_wrapper *sessionHandler = server->get_session_by_id(data->session);
        vector<char> str(header->length - sizeof(cmdAddText_data) + 1);
        memcpy(str.data(), &data->content, header->length - sizeof(cmdAddText_data));
        str.back() = 0;

        if (sessionHandler) {
            string as_s(str.data());
            if (as_s.size() != header->length - sizeof(cmdAddText_data)) {
                cerr << "[?] Null byte in added string" << endl;
            }
            sessionHandler->push_order(client_id, header->request_id, cmdAddText, 0, as_s);
        } else {
            ack(header->request_id, ReturnCode::no_such_session);
        }
        return;
    }

    if ((header->command == cmdSaveFrame) or (header->command == cmdSnapshot) or (header->command == cmdRestore)) {
        assert(header->length > sizeof(cmdSaveFrame_data));
        auto *data = (cmdSaveFrame_data *) packet_ptr;

        session_wrapper *sessionHandler = server->get_session_by_id(data->session);
        if (sessionHandler == nullptr) {
            ack(header->request_id, ReturnCode::no_such_session);
            return;
        }

        vector<char> str(header->length - sizeof(cmdSaveFrame_data) + 1);
        memcpy(str.data(), &data->content, header->length - sizeof(cmdSaveFrame_data));
        str.back() = 0;
        string as_s(str.data());

        if (as_s.size() != header->length - sizeof(cmdSaveFrame_data)) {
            cerr << "[?] Null byte in path" << endl;
            ack(header->request_id, ReturnCode::bad_arguments);
            return;
        }

        sessionHandler->push_order(client_id, header->request_id, header->command, 0, as_s);
        return;
    }

    ack(header->request_id, ReturnCode::unknown_command);
}

void client::add_out_packet(OutCommands packet_type, u32 reply_code, vector<u8> const &packet) {
    // Synchronized or not !
    u32 sz = packet.size();
    vector<u8> output(12 + packet.size());
    memcpy(output.data(), &packet_type, 4);
    memcpy(output.data() + 4, &reply_code, 4);
    memcpy(output.data() + 8, &sz, 4);
    memcpy(output.data() + 12, packet.data(), sz);

    outbound_buffer.emplace_back(std::move(output));
    server->set_write_bit_at(position);
}
