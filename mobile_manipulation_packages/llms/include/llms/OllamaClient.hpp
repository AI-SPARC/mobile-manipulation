#ifndef OLLAMA_CLIENT_HPP
#define OLLAMA_CLIENT_HPP

#include <string>
#include <iostream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class OllamaClient {
public:
    OllamaClient() {
        curl_global_init(CURL_GLOBAL_ALL);
    }

    ~OllamaClient() {
        curl_global_cleanup();
    }

    json infer(const std::string& user_command) {
        CURL* curl;
        CURLcode res;
        std::string readBuffer;

        curl = curl_easy_init();
        if(curl) {
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            
            json payload = {
                {"model", "phi35_leve"},
                {"prompt", build_prompt(user_command)},
                {"format", "json"},
                {"stream", false},
                {"options", {
                    {"temperature", 0.0},
                    {"top_p", 0.1},
                    {"repeat_penalty", 1.2}
                }}
            };

            std::string payload_str = payload.dump();

            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);

            res = curl_easy_perform(curl);
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if(res != CURLE_OK) {
                std::cerr << "Falha na conexão com Ollama: " << curl_easy_strerror(res) << std::endl;
                return json::array();
            }
        }

        try {
            auto response_wrapper = json::parse(readBuffer);
            std::string actual_json_str = response_wrapper["response"];
            return json::parse(actual_json_str);
        } catch (const std::exception& e) {
            std::cerr << "Erro ao ler JSON da LLM: " << e.what() << std::endl;
            return json::array();
        }
    }

private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    std::string build_prompt(const std::string& cmd) {
        return R"(You are a robot task planner. Convert commands to JSON.

AVAILABLE SKILLS (use EXACTLY these names):
- "pick" = grab/take/pick up an object
- "place" = put/place/store an object somewhere  
- "goto_location" = go/move/navigate to a location

OUTPUT FORMAT (strict JSON):
{"commands": [{"skill": "SKILL_NAME", "params": {"id": "OBJECT_ID"}}]}

RULES:
1. Field name must be exactly "skill" (not "skills", not "skillful", not anything else)
2. Field name must be exactly "params" with "id" inside
3. Output ONLY valid JSON, no explanation
4. Extract object IDs from the command (e.g., "box_01", "storage_02")

EXAMPLES:

Input: "Pick up box_01"
Output: {"commands": [{"skill": "pick", "params": {"id": "box_01"}}]}

Input: "Take box_01 and put it in storage_02"
Output: {"commands": [{"skill": "pick", "params": {"id": "box_01"}}, {"skill": "place", "params": {"id": "storage_02"}}]}

Input: "Go to storage_02"
Output: {"commands": [{"skill": "goto_location", "params": {"id": "storage_02"}}]}

Input: "Grab box_01 and go to storage_02 but don't place it"
Output: {"commands": [{"skill": "pick", "params": {"id": "box_01"}}, {"skill": "goto_location", "params": {"id": "storage_02"}}]}

Input: "Pegue a box_01"
Output: {"commands": [{"skill": "pick", "params": {"id": "box_01"}}]}

Input: "Leve box_01 para storage_02"
Output: {"commands": [{"skill": "pick", "params": {"id": "box_01"}}, {"skill": "goto_location", "params": {"id": "storage_02"}}, {"skill": "place", "params": {"id": "storage_02"}}]}

Now convert this command to JSON:
Input: ")" + cmd + R"("
Output: )";
    }
};

#endif