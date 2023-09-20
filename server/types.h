#ifndef VULKAN_LLAMA_SERVER_TYPES_H
#define VULKAN_LLAMA_SERVER_TYPES_H

#include "../types.h"

#define GUARDED_BY(x) \
  THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

namespace lsrv {
    class llava_server;
    class session_wrapper;
    class client;
}

struct session_order {
    u32 client_id;
    u32 request_id;
    u32 opcode;
    u32 arg1;
    std::string s;
};

/* From user:
getTokenMap (0)
newSession (1)
listSessions (2)
getSessionTokens (16, session)
subscribe (17, session)
unsubscribe (18, session)
killSession (19, session)

rewind (32, session, new_pos)
addToken (33, session, token_id)
addText (34, session, [chars])
setSessionOptions (35, session, options)

From session:
newToken (0, token)

Outbound:
newToken (0, req_id, session, index, token) // Subscription
buffer (1, req_id, session, [tokens]) // Response
tokenMap (2, req_id, [nz-str]) // Response
newSession (3, req_id, session) // Response or not
sessionList (4, req_id, [session]) // Response
droppedSession (5, req_id, session) // Broadcast
ack (6, req_id)
*/

enum Commands : u32 {
    // No data
    cmdGetTokenMap = 0,
    cmdNewSession = 1,
    cmdListSessions = 2,

    // Data = {session}
    cmdGetSessionTokens = 16,
    cmdSubscribe = 17,
    cmdUnsubscribe = 18,
    cmdKillSession = 19,
    cmdStartSessionGeneration = 20,
    cmdStopSessionGeneration = 21,
    cmdTick = 22,
    cmdGetSessionStatus = 23,

    cmdRewind = 32,
    cmdAddToken = 33,
    cmdAddText = 34,
    cmdSetSessionOptions = 35,
    cmdSaveFrame = 36,
    cmdSnapshot = 37,
    cmdRestore = 38,
};

enum OutCommands : u32 {
    outCmdNewToken = 0,
    outCmdBuffer = 1,
    outCmdTokenMap = 2,
    outCmdNewSession = 3,
    outCmdSessionList = 4,
    outCmdDroppedSession = 5,
    outCmdAck = 6,
    outCmdSessionStatus = 7,
};

struct cmdRewind_data {
    u32 session;
    u32 new_position;
} __attribute__((packed));

struct cmdSetSessionOptions_data {
    u32 session;
    u32 options;
} __attribute__((packed));

struct cmdAddToken_data {
    u32 session;
    u32 token_id;
} __attribute__((packed));

struct cmdAddText_data {
    u32 session;
    char content[];
} __attribute__((packed));

struct cmdSaveFrame_data {
    u32 session;
    char content[];
} __attribute__((packed));

struct tlv_header {
    u32 command;
    u32 request_id;
    u32 length;
} __attribute__((packed));

#endif //VULKAN_LLAMA_SERVER_TYPES_H
