#include "session_wrapper.h"
#include "server.h"
#include "client.h"
#include "../llava_context.h"
#include "../llava_session.h"
#include <sys/poll.h>
#include <cstring>
#include <iostream>
#include <cassert>
#include <csignal>
#include <utility>
#include "../utils.h"

using namespace lsrv;

u32 session_wrapper::ticker = 1U;

// Main thread handlers
void session_wrapper::push_order(u32 client_id, u32 request_id, u32 opcode, u32 arg1, string s) {
    // Main thread
    {
        unique_lock lock(session_mutex);
        orders.emplace_back() = {
                .client_id = client_id,
                .request_id = request_id,
                .opcode = opcode,
                .arg1 = arg1,
                .s = std::move(s)
        };
    }
    cv.notify_one();
}

session_wrapper::~session_wrapper() {
    // SYNC
    thread* _loop_thread = loop_thread;
    if (_loop_thread != nullptr and _loop_thread->joinable()) {
        _loop_thread->join();
    }
    delete _loop_thread;
    loop_thread = nullptr;
}

session_wrapper::session_wrapper(llava_server *_server) : session_id(ticker++), server(_server) {

}

void session_wrapper::start() {
    // SYNC
    assert(loop_thread == nullptr);
    loop_thread = new thread([this](){this->run();});
}


void session_wrapper::run() {
    // ASYNC
    setup_exceptions();

    llava_session session(server->ctx);

    while (not should_die) {
        process_orders(session, not (should_run or should_tick));
        if (should_run or should_tick) {
            if (not session.start_next_token_prediction()) {
                should_run = false;
                cerr << "Error starting token generation" << endl;
                if (should_tick) {
                    add_outbound_ack(tick_client, should_tick, ReturnCode::nok);
                    flush_outbound_packets();
                }
                should_tick = 0;
                tick_client = 0;
                continue;
            }

            // Let's process some events, if any, while token being generated
            process_orders(session, false);

            u32 new_token_id = session.finish_next_token_prediction();

            vector<u8> vec(12);
            memcpy(vec.data(), &session_id, 4);
            memcpy(vec.data() + 4, &new_token_id, 4);
            u32 pos = session.get_token_count();
            memcpy(vec.data() + 8, &pos, 4);

            if (not session.push_token(new_token_id)) {
                cerr << "Out of buffer ?" << endl;
                should_run = false;
            }

            outbound_broadcasts.emplace_back(session_id, outCmdNewToken, vec);

            if (should_tick) {
                outbound_packets.emplace_back(tick_client, outCmdNewToken, should_tick, vec);
            }

            flush_outbound_packets();
            should_tick = 0;
            tick_client = 0;
        }
    }
    process_orders(session, false);
    server->notify_session_death(session_id);
}

void session_wrapper::setup_exceptions() {
    if (this->server->ctx->signal_debug_on()) {
        return;
    }
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGQUIT);
    sigaddset(&mask, SIGHUP);
    sigaddset(&mask, SIGTERM);
    sigprocmask(SIG_BLOCK, &mask, nullptr);
}

void session_wrapper::process_orders(llava_session& session, bool blocking) {
    // ASYNC
    unique_lock lock(session_mutex);
    if (orders.empty() and blocking) {
        cv.wait(lock, [this](){return not this->orders.empty();});
    }

    while (not orders.empty()) {
        session_order order = orders.front();
        ReturnCode return_code = ReturnCode::ok;
        bool should_acknowledge = true; // Must functions need to send ack, some do not. Set this to false in this case
        switch (order.opcode) {
            case cmdStartSessionGeneration:
                if (should_tick) {
                    return_code = ReturnCode::tick_already_requested;
                } else {
                    should_run = true;
                }
                break;
            case cmdTick:
                if (should_run) {
                    return_code = ReturnCode::already_running; // No tick while running
                } else if (should_tick != 0) {
                    return_code = ReturnCode::tick_already_requested;
                } else {
                    should_tick = order.request_id;
                    tick_client = order.client_id;
                    should_acknowledge = false;
                }
                break;
            case cmdStopSessionGeneration:
                should_run = false;
                break;
            case cmdRewind:
                session.rewind(order.arg1);
                break;
            case cmdSetSessionOptions:
                return_code = session.set_options(order.arg1);
                break;
            case cmdKillSession:
                should_die = true;
                break;
            case cmdAddToken:
                if(not session.push_token(order.arg1)) {
                    return_code = ReturnCode::nok;
                }
                break;
            case cmdAddText:
                if(not session.add_text(order.s)) {
                    return_code = ReturnCode::nok;
                }
                break;
            case cmdGetSessionTokens:
                should_acknowledge = false;
                {
                    u32 token_count = session.get_token_count();
                    vector<u8> nb(4 * token_count);
                    memcpy(nb.data(), session.get_token_buffer().data(), 4 * token_count);
                    outbound_packets.emplace_back(order.client_id, outCmdBuffer, order.request_id, nb);
                }
                break;
            case cmdSaveFrame:
                return_code = session.save_frame(order.s);
                break;
            case cmdSnapshot:
                return_code = session.snapshot(order.s);
                break;
            case cmdRestore:
                if (should_run) {
                    return_code = ReturnCode::already_running;
                } else if (should_tick) {
                    return_code = ReturnCode::tick_already_requested;
                } else {
                    return_code = session.restore(order.s);
                }
                break;
            default:
                break;
        }
        if (order.client_id and order.request_id and should_acknowledge) {
            vector<u8> nb(4);
            memcpy(nb.data(), &return_code, 4);
            add_outbound_ack(order.client_id, order.request_id, return_code);
        }
        orders.pop_front();
    }

    lock.unlock();

    flush_outbound_packets();
}

void session_wrapper::join() {
    auto* _thread = this->loop_thread;
    if (_thread) {
        _thread->join();
    }
}

void session_wrapper::flush_outbound_packets() {
    if (not (outbound_packets.empty() and outbound_broadcasts.empty())) {
        server->receive_session_outbound_packets(outbound_packets, outbound_broadcasts);
    }
}

void session_wrapper::add_outbound_ack(u32 client_id, u32 request_code, ReturnCode return_code) {
    vector<u8> packet(4);
    memcpy(packet.data(), &return_code, 4);
    outbound_packets.emplace_back(client_id, outCmdAck, request_code, std::move(packet));
}
