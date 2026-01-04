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
                {"keep_alive", 0}, 
                {"options", {
                    {"temperature", 0.0},
                    {"top_p", 0.1},
                    {"num_predict", 500}
                }}
            };

            std::string payload_str = payload.dump();

            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 45L); 

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
            
           
            std::cout << "[OllamaClient] Resposta raw: " << actual_json_str << std::endl;
            
            return json::parse(actual_json_str);
        } catch (const std::exception& e) {
            std::cerr << "Erro ao ler JSON da LLM: " << e.what() << std::endl;
            std::cerr << "[OllamaClient] Buffer recebido: " << readBuffer << std::endl;
            return json::array();
        }
    }

private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

    std::string build_prompt(const std::string& cmd) {
        std::string prompt = R"DELIM(Convert to JSON. Skills: pick, place, goto_location.
Coordinates: "x;y;z". Range: "prefix_start:prefix_end". Variables: $item, $dest.

"Pick box_01" = {"commands":[{"skill":"pick","params":{"id":"box_01"}}]}
"Go to (1,2,3)" = {"commands":[{"skill":"goto_location","params":{"id":"1;2;3"}}]}
"Pick box_01 to box_03, place in storage" = {"commands":[{"loop":{"item":"box_01:box_03"},"do":[{"skill":"pick","params":{"id":"$item"}},{"skill":"place","params":{"id":"storage"}}]}]}
"Pick box_01 to box_03, place in storage_01 to storage_03" = {"commands":[{"loop":{"item":"box_01:box_03","dest":"storage_01:storage_03"},"do":[{"skill":"pick","params":{"id":"$item"}},{"skill":"place","params":{"id":"$dest"}}]}]}

)DELIM";
        return prompt + "\"" + cmd + "\" = ";
    }
};

#endif