#!/usr/bin/env python3
"""
Cliente Gambiarra Arena usando LangChain para orquestração avançada.
Este cliente usa o módulo orchestration.py para processar prompts
de forma mais sofisticada.
"""
import asyncio
import json
import time

import websockets
from orchestration import create_orchestrator, TokenCollectorCallback

# ============================================
# CONFIGURAÇÕES - Edite aqui conforme necessário
# ============================================
HOST = "192.168.0.212"  # IP do servidor da arena
PIN = "937414"  # PIN da sessão
PARTICIPANT_ID = "meuId"  # ID do participante
NICKNAME = "Python LangChain"  # Apelido do participante

# Configurações do Ollama
OLLAMA_HOST = "localhost"  # Host do servidor Ollama
OLLAMA_MODEL = "qwen3:0.6b"  # Modelo a ser usado

# Configurações do LangChain
USE_ENHANCED = True  # Usar orquestrador aprimorado
TEMPERATURE = 0.7  # Criatividade (0.0-1.0)
MAX_TOKENS = 500  # Máximo de tokens por resposta
STRATEGY = "accuracy_focused"  # speed_focused, accuracy_focused, detailed
# ============================================

# URLs construídas automaticamente
URL = f"ws://{HOST}:3000/ws"


async def handle_challenge_with_langchain(ws, challenge: dict, orchestrator):
    """
    Trata desafio usando o orquestrador LangChain com streaming.
    """
    round_id = challenge.get("round")
    prompt = challenge.get("prompt", "")

    print("\n=== Novo desafio recebido ===")
    print(f"round: {round_id}")
    print(f"prompt: {prompt}")
    print("=============================\n")

    # Definir contexto baseado no tipo de prompt
    context = "Competitive AI Arena Challenge - provide the best possible answer"

    seq = 0
    total_tokens = 0
    start_time = time.perf_counter()
    first_token_time = None

    print(f"Processando com LangChain (modelo: {OLLAMA_MODEL})...")

    try:
        # Criar callback para capturar tokens
        callback = TokenCollectorCallback()

        # Criar chain com callback
        if USE_ENHANCED:
            chain = orchestrator.create_enhanced_chain(
                callback_handler=callback,
                strategy=STRATEGY
            )
        else:
            chain = orchestrator.create_chain(callback_handler=callback)

        # Executar chain em background
        loop = asyncio.get_event_loop()
        inputs = {"user_input": prompt, "context": context}
        if USE_ENHANCED:
            inputs["strategy"] = STRATEGY

        task = loop.run_in_executor(
            None,
            lambda: chain.run(**inputs)
        )

        # Processar tokens conforme chegam (usando queue.Queue thread-safe)
        full_response = ""
        while True:
            # Ler da queue thread-safe de forma não bloqueante
            token = await loop.run_in_executor(None, callback.token_queue.get)

            if token is None:  # Fim da geração
                break

            if first_token_time is None:
                first_token_time = time.perf_counter()

            full_response += token
            total_tokens += 1

            # Enviar token para o servidor
            token_msg = {
                "type": "token",
                "round": round_id,
                "participant_id": PARTICIPANT_ID,
                "seq": seq,
                "content": token,
            }

            await ws.send(json.dumps(token_msg))
            seq += 1

        # Aguardar conclusão
        await task

    except Exception as e:
        print(f"❌ Erro ao processar com LangChain: {e}")
        import traceback
        traceback.print_exc()
        return

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    latency_first_token_ms = int((first_token_time - start_time) * 1000)
    duration_ms = int((end_time - start_time) * 1000)

    print(f"\n✅ Resposta completa ({total_tokens} tokens):")
    print(full_response)
    print()

    # Enviar mensagem de conclusão
    complete_msg = {
        "type": "complete",
        "round": round_id,
        "participant_id": PARTICIPANT_ID,
        "tokens": total_tokens,
        "latency_ms_first_token": latency_first_token_ms,
        "duration_ms": duration_ms,
    }

    print("Enviando mensagem de conclusão com métricas:")
    print(json.dumps(complete_msg, indent=2, ensure_ascii=False))

    await ws.send(json.dumps(complete_msg))


async def client_loop():
    """
    Loop principal do cliente com integração LangChain.
    """
    print(f"Conectando a {URL} ...")

    # Criar orquestrador
    orchestrator = create_orchestrator(
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        enhanced=USE_ENHANCED,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    print(f"Orquestrador LangChain inicializado")
    print(f"  - Modelo: {OLLAMA_MODEL}")
    print(f"  - Enhanced: {USE_ENHANCED}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Max Tokens: {MAX_TOKENS}")
    if USE_ENHANCED:
        print(f"  - Strategy: {STRATEGY}")
    print()

    async with websockets.connect(URL) as ws:
        print("✅ Conectado ao servidor WebSocket.")

        # Mensagem de registro
        register_msg = {
            "type": "register",
            "participant_id": PARTICIPANT_ID,
            "nickname": NICKNAME,
            "pin": PIN,
            "runner": "langchain",
            "model": OLLAMA_MODEL,
        }

        print("Enviando mensagem de registro:")
        print(json.dumps(register_msg, indent=2, ensure_ascii=False))
        await ws.send(json.dumps(register_msg))

        # Loop de recebimento de mensagens
        while True:
            try:
                raw_msg = await ws.recv()
            except websockets.ConnectionClosed as e:
                print(f"Conexão fechada pelo servidor: {e.code} - {e.reason}")
                break

            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                print(f"Mensagem não é JSON válido: {raw_msg}")
                continue

            msg_type = data.get("type")

            if msg_type == "challenge":
                await handle_challenge_with_langchain(ws, data, orchestrator)

            elif msg_type == "heartbeat":
                print("💓 Heartbeat recebido do servidor.")

            else:
                print("📩 Mensagem desconhecida recebida:")
                print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    try:
        asyncio.run(client_loop())
    except KeyboardInterrupt:
        print("\nEncerrando cliente por KeyboardInterrupt.")


if __name__ == "__main__":
    main()
