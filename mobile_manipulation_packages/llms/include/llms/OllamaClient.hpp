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

    // Função que envia o comando e retorna o JSON já parseado
    json infer(const std::string& user_command) {
        CURL* curl;
        CURLcode res;
        std::string readBuffer;

        curl = curl_easy_init();
        if(curl) {
            // Configura Headers
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            
            // Monta o Payload JSON
            json payload = {
                {"model", "phi35_leve"}, // <--- SEU MODELO OTIMIZADO
                {"prompt", build_prompt(user_command)},
                {"format", "json"},      // <--- FORÇA SAÍDA JSON
                {"stream", false},       // <--- Resposta completa de uma vez
                {"options", {
                    {"temperature", 0.0} // <--- Zero criatividade
                }}
            };

            std::string payload_str = payload.dump();

            // Configura o cURL
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L); // Timeout de 10s

            // Executa
            res = curl_easy_perform(curl);
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if(res != CURLE_OK) {
                std::cerr << "Falha na conexão com Ollama: " << curl_easy_strerror(res) << std::endl;
                return json::array();
            }
        }

        // Parse da resposta
        try {
            auto response_wrapper = json::parse(readBuffer);
            // O Ollama devolve o texto dentro de "response". 
            // Esse texto interno é o nosso JSON final.
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

    // Aqui definimos suas Habilidades (Skills)
    std::string build_prompt(const std::string& cmd) {
        return R"(
LISTA DE FERRAMENTAS (SKILLS):
1. {"skill": "pick", "params": {"id": "obj_id"}} 
   - Pegar/Segurar objeto.
2. {"skill": "place", "params": {"location": "loc_name"}} 
   - Guardar/Colocar objeto.
3. {"skill": "goto_location", "params": {"location": "loc_name"}} 
   - Ir/Navegar para local.

USER COMMAND: )" + cmd;
    }
};

#endif